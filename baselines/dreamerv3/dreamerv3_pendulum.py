"""
DreamerV3-style online baseline for constrained true double-pendulum control.

ALRNN/MTF-SAC is used ONLY to obtain:
  (1) the subject-specific 6x6 action projection matrix;
  (2) reference_model.reward_function.
All dynamics, RSSM, actor, critic, and replay learning below are independent.

The code implements a compact DreamerV3-style agent for vector observations:
  * categorical RSSM (GRU deterministic state + categorical stochastic state),
  * observation/reward/continuation heads, balanced KL with free nats,
  * imagined actor-critic learning with lambda returns and a slow critic,
  * validation-only search over {small, medium} x train_ratio {16, 32},
  * final 5-seed x 16-subject x 5-rollout evaluation loop.

It is a compact pendulum adaptation, not a byte-for-byte copy of the official
DreamerV3 repository (which includes additional implementation details).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import copy, csv, json, math, os, random, sys
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import odeint

Array = np.ndarray
RewardFunction = Callable[[Array, Array], float]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); tc.manual_seed(seed)
    if tc.cuda.is_available(): tc.cuda.manual_seed_all(seed)


def symlog(x: tc.Tensor) -> tc.Tensor:
    return tc.sign(x) * tc.log1p(tc.abs(x))


def symexp(x: tc.Tensor) -> tc.Tensor:
    return tc.sign(x) * tc.expm1(tc.abs(x))


def make_mlp(inp: int, out: int, hid: int, depth: int = 2) -> nn.Sequential:
    layers: List[nn.Module] = []
    cur = inp
    for _ in range(depth):
        layers += [nn.Linear(cur, hid), nn.LayerNorm(hid), nn.SiLU()]
        cur = hid
    layers.append(nn.Linear(cur, out))
    return nn.Sequential(*layers)


@dataclass
class DreamerV3Config:
    observation_dim: int = 6
    action_dim: int = 6
    embed_dim: int = 128
    deter_dim: int = 192
    stoch_classes: int = 12
    stoch_categories: int = 16
    hidden_dim: int = 192

    replay_capacity_episodes: int = 2000
    batch_size: int = 32
    batch_length: int = 32
    seed_steps: int = 2000
    total_environment_steps: int = 100_000
    train_ratio: int = 16
    train_every: int = 16

    world_model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    grad_clip_norm: float = 100.0
    kl_free_nats: float = 1.0
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1
    observation_scale: float = 1.0
    reward_scale: float = 1.0
    continuation_scale: float = 1.0

    discount: float = 0.99
    lambda_return: float = 0.95
    imagination_horizon: int = 15
    slow_target_tau: float = 0.01
    actor_entropy_scale: float = 3e-4
    actor_min_std: float = 0.1
    actor_max_std: float = 1.0
    action_scale: float = 1.0
    action_low: float = -100.0
    action_high: float = 100.0

    seed: int = 0
    device: str = "cpu"

    @property
    def stoch_dim(self) -> int:
        return self.stoch_classes * self.stoch_categories

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_dim


def sized_config(base: DreamerV3Config, size: str) -> DreamerV3Config:
    if size == "very_small":
        return replace(base, embed_dim=64, deter_dim=96,
                       stoch_classes=8, stoch_categories=16, hidden_dim=96)
    if size == "small":
        return replace(base, embed_dim=128, deter_dim=192,
                       stoch_classes=12, stoch_categories=16, hidden_dim=192)
    if size == "medium":
        return replace(base, embed_dim=192, deter_dim=256,
                       stoch_classes=16, stoch_categories=16, hidden_dim=256)
    if size == "large":
        return replace(base, embed_dim=256, deter_dim=384,
                       stoch_classes=20, stoch_categories=16, hidden_dim=384)
    raise ValueError("size must be 'very_small', 'small', 'medium', or 'large'")


class ProjectedActionSpace:
    """Same order as CEM/TD-MPC: clip raw action, then apply P @ action."""
    def __init__(self, low: float, high: float, projection: Optional[Array], device: tc.device):
        self.low, self.high = float(low), float(high)
        self.matrix = None if projection is None else tc.as_tensor(
            np.asarray(projection, dtype=np.float32), device=device)
        self.matrix_np = None if projection is None else np.asarray(projection, dtype=np.float32)
        if self.matrix is not None and tuple(self.matrix.shape) != (6, 6):
            raise ValueError(f"Projection must have shape (6,6), got {tuple(self.matrix.shape)}")

    def tensor(self, action: tc.Tensor) -> tc.Tensor:
        action = action.clamp(self.low, self.high)
        return action if self.matrix is None else action @ self.matrix.T

    def numpy(self, action: Array) -> Array:
        action = np.clip(np.asarray(action, dtype=np.float64), self.low, self.high)
        return action if self.matrix_np is None else action @ self.matrix_np.T


class TrueDoublePendulumEnv:
    def __init__(self, pendulum_params, observation_to_ode_state, ode_state_to_observation,
                 double_pendulum_derivatives, reward_fn: RewardFunction,
                 action_space: ProjectedActionSpace, dt: float = .01, episode_length: int = 500):
        self.params = pendulum_params
        self.obs_to_ode = observation_to_ode_state
        self.ode_to_obs = ode_state_to_observation
        self.derivatives = double_pendulum_derivatives
        self.reward_fn = reward_fn
        self.action_space = action_space
        self.dt, self.episode_length = float(dt), int(episode_length)
        self.ode_args = (self.params['m1'], self.params['m2'], self.params['l1'], self.params['l2'], 9.81)
        self.state = None; self.obs = None; self.time = 0.; self.steps = 0

    @property
    def physical_state(self) -> Array:
        if self.state is None: raise RuntimeError("reset first")
        return self.state.copy()

    def reset(self, x0: Array) -> Array:
        obs = np.asarray(x0, dtype=np.float64)
        if obs.shape == (1, 6): obs = obs[0]
        if obs.shape != (6,): raise ValueError(f"Expected 6D observation, got {obs.shape}")
        self.state = np.asarray(self.obs_to_ode(obs), dtype=np.float64)
        if self.state.shape != (4,): raise ValueError("obs_to_ode must return 4D state")
        self.obs = obs.copy(); self.time = 0.; self.steps = 0
        return self.obs.copy()

    def _rk4_step(self, state: Array) -> Array:
        dt = self.dt
        t = self.time
        args = self.ode_args
        k1 = self.derivatives(state, t, *args)
        k2 = self.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, *args)
        k3 = self.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, *args)
        k4 = self.derivatives(state + dt * k3, t + dt, *args)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, raw_action: Array):
        if self.state is None: raise RuntimeError("reset first")
        action = self.action_space.numpy(raw_action)
        state = self.state.copy()
        state[1] += action[4]
        state[3] += action[5]
        next_state = odeint(self.derivatives, state, [self.time, self.time + self.dt], args=self.ode_args)[1]
        next_state = np.asarray(next_state, dtype=np.float64)
        failure = not np.all(np.isfinite(next_state))
        if failure: next_state = np.nan_to_num(next_state, nan=0., posinf=1e6, neginf=-1e6)
        next_obs = np.asarray(self.ode_to_obs(next_state), dtype=np.float64)
        reward = float(self.reward_fn(next_obs, action))
        self.state, self.obs = next_state, next_obs
        self.time += self.dt; self.steps += 1
        done = failure or self.steps >= self.episode_length
        return next_obs.copy(), reward, done, {'executed_action': action.copy(), 'physical_state': next_state.copy(), 'numerical_failure': failure}


def make_reward_adapter(reference_model: Any, measurement_index: int) -> RewardFunction:
    device = getattr(reference_model, 'device', tc.device('cpu'))
    def reward(obs: Array, action: Array) -> float:
        with tc.no_grad():
            value = reference_model.reward_function(
                tc.as_tensor(obs, dtype=tc.float32, device=device).view(1, 1, 6),
                tc.as_tensor(action, dtype=tc.float32, device=device).view(1, 1, 6),
                measurement_index)
        return float(value.squeeze().cpu().item())
    return reward


@dataclass
class Episode:
    observations: Array
    actions: Array
    rewards: Array
    continues: Array


class EpisodeReplay:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity; self.episodes = deque(); self.rng = np.random.default_rng(seed)
        self.transition_count = 0; self.max_episode_len = 0

    def __len__(self): return len(self.episodes)
    @property
    def transitions(self): return self.transition_count

    def add(self, obs: Sequence[Array], acts: Sequence[Array], rews: Sequence[float], cont: Sequence[float]):
        if len(acts) == 0: return
        if len(obs) != len(acts) + 1: raise ValueError("obs must have one extra time point")
        episode = Episode(np.asarray(obs, np.float32), np.asarray(acts, np.float32),
                          np.asarray(rews, np.float32)[:, None], np.asarray(cont, np.float32)[:, None])
        self.episodes.append(episode)
        self.transition_count += len(episode.actions)
        self.max_episode_len = max(self.max_episode_len, len(episode.actions))
        if len(self.episodes) > self.capacity:
            removed = self.episodes.popleft()
            self.transition_count -= len(removed.actions)
            if len(removed.actions) == self.max_episode_len:
                self.max_episode_len = max((len(ep.actions) for ep in self.episodes), default=0)

    def can_sample(self, batch: int, length: int) -> bool:
        return self.max_episode_len >= length and self.transition_count >= batch * length

    def sample(self, batch: int, length: int, device: tc.device):
        eligible = [ep for ep in self.episodes if len(ep.actions) >= length]
        if not eligible: raise RuntimeError("No sufficiently long episode")
        O=[]; A=[]; R=[]; C=[]
        for _ in range(batch):
            ep = eligible[int(self.rng.integers(len(eligible)))]
            start = int(self.rng.integers(len(ep.actions)-length+1)); stop=start+length
            O.append(ep.observations[start:stop+1]); A.append(ep.actions[start:stop]); R.append(ep.rewards[start:stop]); C.append(ep.continues[start:stop])
        return tuple(tc.as_tensor(np.asarray(x), device=device) for x in (O,A,R,C))


@dataclass
class RSSMState:
    deter: tc.Tensor
    stoch: tc.Tensor
    logits: tc.Tensor
    @property
    def feat(self): return tc.cat([self.deter, self.stoch], -1)


class RSSM(nn.Module):
    def __init__(self, cfg: DreamerV3Config):
        super().__init__(); self.cfg=cfg
        ld = cfg.stoch_dim
        self.aembed=make_mlp(cfg.action_dim, cfg.hidden_dim, cfg.hidden_dim, 1)
        self.gru=nn.GRUCell(ld+cfg.hidden_dim, cfg.deter_dim)
        self.prior=make_mlp(cfg.deter_dim, ld, cfg.hidden_dim, 2)
        self.post=make_mlp(cfg.deter_dim+cfg.embed_dim, ld, cfg.hidden_dim, 2)

    def initial(self, batch: int, device: tc.device) -> RSSMState:
        deter=tc.zeros(batch,self.cfg.deter_dim,device=device)
        logits=tc.zeros(batch,self.cfg.stoch_classes,self.cfg.stoch_categories,device=device)
        stoch=F.one_hot(tc.zeros(batch,self.cfg.stoch_classes,dtype=tc.long,device=device), self.cfg.stoch_categories).float().reshape(batch,-1)
        return RSSMState(deter,stoch,logits)

    def sample(self, logits: tc.Tensor, deterministic=False):
        if deterministic:
            x=F.one_hot(logits.argmax(-1),self.cfg.stoch_categories).float()
        elif self.training:
            x=F.gumbel_softmax(logits,tau=1.,hard=True,dim=-1)
        else:
            x=F.one_hot(tc.distributions.Categorical(logits=logits).sample(),self.cfg.stoch_categories).float()
        return x.reshape(logits.shape[0],-1)

    def deter_step(self, prev: RSSMState, action: tc.Tensor):
        return self.gru(tc.cat([prev.stoch,self.aembed(action)],-1),prev.deter)

    def observe(self, prev: RSSMState, action: tc.Tensor, embed: tc.Tensor, deterministic=False):
        deter=self.deter_step(prev,action)
        logits=self.post(tc.cat([deter,embed],-1)).reshape(-1,self.cfg.stoch_classes,self.cfg.stoch_categories)
        return RSSMState(deter,self.sample(logits,deterministic),logits)

    def imagine(self, prev: RSSMState, action: tc.Tensor, deterministic=False):
        deter=self.deter_step(prev,action)
        logits=self.prior(deter).reshape(-1,self.cfg.stoch_classes,self.cfg.stoch_categories)
        return RSSMState(deter,self.sample(logits,deterministic),logits)

    def prior_logits(self,deter):
        return self.prior(deter).reshape(-1,self.cfg.stoch_classes,self.cfg.stoch_categories)


class Networks(nn.Module):
    def __init__(self,cfg:DreamerV3Config):
        super().__init__(); self.cfg=cfg
        self.encoder=make_mlp(6,cfg.embed_dim,cfg.hidden_dim,2)
        self.rssm=RSSM(cfg)
        self.decoder=make_mlp(cfg.feature_dim,6,cfg.hidden_dim,2)
        self.reward=make_mlp(cfg.feature_dim,1,cfg.hidden_dim,2)
        self.continue_head=make_mlp(cfg.feature_dim,1,cfg.hidden_dim,2)
        self.actor=make_mlp(cfg.feature_dim,12,cfg.hidden_dim,3)
        self.critic=make_mlp(cfg.feature_dim,1,cfg.hidden_dim,3)
    def dist(self,feat):
        mean, rawstd=self.actor(feat).chunk(2,-1)
        mean=tc.tanh(mean)
        std=self.cfg.actor_min_std+(self.cfg.actor_max_std-self.cfg.actor_min_std)*tc.sigmoid(rawstd+2.)
        return tc.distributions.Normal(mean,std)


def cat_kl(p_logits:tc.Tensor,q_logits:tc.Tensor):
    pl=F.log_softmax(p_logits,-1); p=pl.exp(); ql=F.log_softmax(q_logits,-1)
    return (p*(pl-ql)).sum(-1).sum(-1)


def lambda_return(rewards, values, continues, bootstrap, discount, lam):
    out=tc.zeros_like(rewards); last=bootstrap
    for i in reversed(range(len(rewards))):
        nxt=bootstrap if i==len(rewards)-1 else values[i+1]
        last=rewards[i]+discount*continues[i]*((1-lam)*nxt+lam*last)
        out[i]=last
    return out


class DreamerV3Agent:
    def __init__(self,cfg:DreamerV3Config,projection:Optional[Array]):
        self.cfg=cfg; self.device=tc.device(cfg.device); set_seed(cfg.seed); self.rng=np.random.default_rng(cfg.seed)
        self.actions=ProjectedActionSpace(cfg.action_low,cfg.action_high,projection,self.device)
        self.net=Networks(cfg).to(self.device); self.slow=copy.deepcopy(self.net.critic).to(self.device); self.slow.requires_grad_(False)
        self.world_opt=tc.optim.Adam(list(self.net.encoder.parameters())+list(self.net.rssm.parameters())+list(self.net.decoder.parameters())+list(self.net.reward.parameters())+list(self.net.continue_head.parameters()),lr=cfg.world_model_lr)
        self.actor_opt=tc.optim.Adam(self.net.actor.parameters(),lr=cfg.actor_lr)
        self.critic_opt=tc.optim.Adam(self.net.critic.parameters(),lr=cfg.critic_lr)
        self.replay=EpisodeReplay(cfg.replay_capacity_episodes,cfg.seed)
        self.belief=None; self.prev_action=None; self.updates=0

    def reset_belief(self): self.belief=None; self.prev_action=None

    @tc.no_grad()
    def prime_belief(self,obs:Array,deterministic:bool=False):
        self.net.eval()
        x=tc.as_tensor(obs,dtype=tc.float32,device=self.device).view(1,self.cfg.observation_dim)
        emb=self.net.encoder(x)
        if self.belief is None:
            init=self.net.rssm.initial(1,self.device)
            logits=self.net.rssm.post(tc.cat([init.deter,emb],-1)).reshape(1,self.cfg.stoch_classes,self.cfg.stoch_categories)
            self.belief=RSSMState(init.deter,self.net.rssm.sample(logits,deterministic),logits)
        else:
            self.belief=self.net.rssm.observe(self.belief,self.prev_action,emb,deterministic)
        self.net.train()

    @tc.no_grad()
    def sample_action_from_belief(self,deterministic:bool=False):
        if self.belief is None:
            raise RuntimeError("belief must be initialized before sampling actions")
        self.net.eval()
        d=self.net.dist(self.belief.feat); norm=d.mean if deterministic else d.sample()
        raw=(norm.clamp(-1,1)*self.cfg.action_scale)
        self.prev_action=self.actions.tensor(raw)
        self.net.train(); return raw[0].cpu().numpy().astype(np.float64)

    @tc.no_grad()
    def act(self,obs:Array,deterministic:bool=False):
        self.net.eval(); x=tc.as_tensor(obs,dtype=tc.float32,device=self.device).view(1,6); emb=self.net.encoder(x)
        if self.belief is None:
            init=self.net.rssm.initial(1,self.device)
            logits=self.net.rssm.post(tc.cat([init.deter,emb],-1)).reshape(1,self.cfg.stoch_classes,self.cfg.stoch_categories)
            self.belief=RSSMState(init.deter,self.net.rssm.sample(logits,deterministic),logits)
        else:
            self.belief=self.net.rssm.observe(self.belief,self.prev_action,emb,deterministic)
        d=self.net.dist(self.belief.feat); norm=d.mean if deterministic else d.sample()
        raw=(norm.clamp(-1,1)*self.cfg.action_scale)
        self.prev_action=self.actions.tensor(raw)
        self.net.train(); return raw[0].cpu().numpy().astype(np.float64)

    def observe_seq(self,obs,acts):
        B,L1,_=obs.shape; emb=self.net.encoder(obs.reshape(-1,6)).reshape(B,L1,-1)
        init=self.net.rssm.initial(B,self.device)
        logits=self.net.rssm.post(tc.cat([init.deter,emb[:,0]],-1)).reshape(B,self.cfg.stoch_classes,self.cfg.stoch_categories)
        states=[RSSMState(init.deter,self.net.rssm.sample(logits),logits)]
        for t in range(acts.shape[1]): states.append(self.net.rssm.observe(states[-1],acts[:,t],emb[:,t+1]))
        return states

    def world_loss(self,obs,acts,rews,cont):
        states=self.observe_seq(obs,acts); feat=tc.stack([s.feat for s in states],1)
        oloss=F.mse_loss(self.net.decoder(feat),obs)
        rloss=F.mse_loss(self.net.reward(feat[:,1:]),symlog(rews))
        closs=F.binary_cross_entropy_with_logits(self.net.continue_head(feat[:,1:]),cont)
        post=tc.stack([s.logits for s in states[1:]],1); prior=tc.stack([self.net.rssm.prior_logits(s.deter) for s in states[1:]],1)
        shape=(-1,self.cfg.stoch_classes,self.cfg.stoch_categories)
        dyn=tc.clamp(cat_kl(post.detach().reshape(shape),prior.reshape(shape)),min=self.cfg.kl_free_nats).mean()
        rep=tc.clamp(cat_kl(post.reshape(shape),prior.detach().reshape(shape)),min=self.cfg.kl_free_nats).mean()
        total=self.cfg.observation_scale*oloss+self.cfg.reward_scale*rloss+self.cfg.continuation_scale*closs+self.cfg.kl_dyn_scale*dyn+self.cfg.kl_rep_scale*rep
        return total, {'world_loss':total.detach(),'observation_loss':oloss.detach(),'reward_loss':rloss.detach(),'continuation_loss':closs.detach(),'dyn_kl':dyn.detach(),'rep_kl':rep.detach()}, states

    def imagine(self,start:RSSMState,H:int):
        state=RSSMState(start.deter.detach(),start.stoch.detach(),start.logits.detach())
        mods=[self.net.rssm,self.net.reward,self.net.continue_head]; flags=[[p.requires_grad for p in m.parameters()] for m in mods]
        for m in mods:
            for p in m.parameters(): p.requires_grad_(False)
        feats=[]; rews=[]; conts=[]; logps=[]
        try:
            for _ in range(H):
                d=self.net.dist(state.feat); norm=d.rsample().clamp(-1,1); raw=norm*self.cfg.action_scale; act=self.actions.tensor(raw)
                nxt=self.net.rssm.imagine(state,act)
                feats.append(state.feat); rews.append(symexp(self.net.reward(nxt.feat))); conts.append(tc.sigmoid(self.net.continue_head(nxt.feat))); logps.append(d.log_prob(norm).sum(-1,keepdim=True)); state=nxt
        finally:
            for m,fs in zip(mods,flags):
                for p,f in zip(m.parameters(),fs): p.requires_grad_(f)
        return tc.stack(feats),tc.stack(rews),tc.stack(conts),tc.stack(logps)

    def update(self):
        c=self.cfg
        obs,acts,rews,cont=self.replay.sample(c.batch_size,c.batch_length,self.device)
        loss,metrics,states=self.world_loss(obs,acts,rews,cont)
        self.world_opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(list(self.net.encoder.parameters())+list(self.net.rssm.parameters())+list(self.net.decoder.parameters())+list(self.net.reward.parameters())+list(self.net.continue_head.parameters()),c.grad_clip_norm); self.world_opt.step()
        start=RSSMState(tc.cat([s.deter for s in states[:-1]],0),tc.cat([s.stoch for s in states[:-1]],0),tc.cat([s.logits for s in states[:-1]],0))
        feats,irew,icont,logp=self.imagine(start,c.imagination_horizon)
        # Actor receives analytic gradients through the frozen RSSM/reward
        # model and the slow critic's dependence on imagined features. The slow
        # critic parameters remain frozen; only the actor is updated here.
        vals=self.slow(feats.reshape(-1,c.feature_dim)).reshape(c.imagination_horizon,-1,1)
        actor_targets=lambda_return(irew,vals,icont,vals[-1],c.discount,c.lambda_return)
        actor_loss=-actor_targets.mean()-c.actor_entropy_scale*logp.mean()
        self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); nn.utils.clip_grad_norm_(self.net.actor.parameters(),c.grad_clip_norm); self.actor_opt.step()

        # The critic regresses to a stop-gradient version of the same imagined
        # lambda-return targets.
        pred=self.net.critic(feats.detach().reshape(-1,c.feature_dim)).reshape(c.imagination_horizon,-1,1)
        critic_loss=F.mse_loss(pred,actor_targets.detach())
        self.critic_opt.zero_grad(set_to_none=True); critic_loss.backward(); nn.utils.clip_grad_norm_(self.net.critic.parameters(),c.grad_clip_norm); self.critic_opt.step()
        with tc.no_grad():
            for p,q in zip(self.net.critic.parameters(),self.slow.parameters()): q.lerp_(p,c.slow_target_tau)
        self.updates+=1
        out={k:float(v.cpu().item()) for k,v in metrics.items()}; out.update(actor_loss=float(actor_loss.detach().cpu()),critic_loss=float(critic_loss.detach().cpu()))
        return out

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        tc.save({'config':asdict(self.cfg),'net':self.net.state_dict(),'slow':self.slow.state_dict(),'world_opt':self.world_opt.state_dict(),'actor_opt':self.actor_opt.state_dict(),'critic_opt':self.critic_opt.state_dict(),'updates':self.updates},path)


def init_obs(rng):
    t1,t2=rng.uniform(-np.pi,np.pi,2)
    return np.array([np.sin(t1),np.cos(t1),np.sin(t2),np.cos(t2),rng.normal(),rng.normal()],dtype=np.float64)


def train(agent:DreamerV3Agent,env:TrueDoublePendulumEnv,total_steps:Optional[int]=None,log_every:int=1000):
    c=agent.cfg; total=total_steps or c.total_environment_steps; rng=np.random.default_rng(c.seed+999)
    obs=env.reset(init_obs(rng)); agent.reset_belief(); O=[obs.copy()];A=[];R=[];C=[]; ret=0.; logs={k:[] for k in ['step','episode_return','world_loss','observation_loss','reward_loss','continuation_loss','dyn_kl','rep_kl','actor_loss','critic_loss']}; latest=None
    for step in range(total):
        raw=rng.uniform(-c.action_scale,c.action_scale,6) if step<c.seed_steps else agent.act(obs,False)
        nxt,r,done,info=env.step(raw); A.append(info['executed_action']);R.append(r);C.append(0. if done else 1.);O.append(nxt.copy());ret+=r;obs=nxt
        if done:
            agent.replay.add(O,A,R,C); logs['step'].append(step+1);logs['episode_return'].append(ret)
            if latest:
                for k in ['world_loss','observation_loss','reward_loss','continuation_loss','dyn_kl','rep_kl','actor_loss','critic_loss']: logs[k].append(latest[k])
            obs=env.reset(init_obs(rng));agent.reset_belief();O=[obs.copy()];A=[];R=[];C=[];ret=0.
        if step>=c.seed_steps and (step+1)%c.train_every==0 and agent.replay.can_sample(c.batch_size,c.batch_length):
            for _ in range(max(1,round(c.train_ratio*c.train_every/c.batch_size))): latest=agent.update()
        if log_every and (step+1)%log_every==0:
            x=np.mean(logs['episode_return'][-10:]) if logs['episode_return'] else float('nan')
            print(f'[DreamerV3] step={step+1:7d} episodes={len(agent.replay):4d} transitions={agent.replay.transitions:7d} mean_return={x:.6f}')
    if A: agent.replay.add(O,A,R,C)
    return logs


@tc.no_grad()
def evaluate(agent:DreamerV3Agent,env:TrueDoublePendulumEnv,x0:Array,T:int=500):
    obs=env.reset(x0);agent.reset_belief();xs=tc.zeros((1,T+1,6));acts=tc.zeros((1,T,6));ode=np.zeros((T+1,4));rewards=np.zeros(T);ode[0]=env.physical_state;xs[0,0]=tc.as_tensor(obs)
    for t in range(T):
        raw=agent.act(obs,True);nxt,r,done,info=env.step(raw);xs[0,t+1]=tc.as_tensor(nxt);acts[0,t]=tc.as_tensor(info['executed_action']);ode[t+1]=env.physical_state;rewards[t]=r;obs=nxt
        if done:
            for j in range(t+1,T):xs[0,j+1]=tc.as_tensor(obs);ode[j+1]=env.physical_state
            break
    return {'observations':xs,'actions':acts,'ode_states':ode,'rewards':rewards}


@tc.no_grad()
def evaluate_world_model(agent:DreamerV3Agent,x0:Array,T:int=500):
    obs=np.asarray(x0,dtype=np.float64)
    if obs.ndim==2:
        if obs.shape[0]!=1:
            raise ValueError(f"x0 must have batch size 1; received {obs.shape}.")
        obs=obs[0]
    if obs.shape!=(6,):
        raise ValueError(f"x0 must resolve to a 6D observation; got {obs.shape}.")

    agent.reset_belief()
    agent.prime_belief(obs,deterministic=True)

    xs=tc.zeros((1,T+1,6),dtype=tc.float32)
    acts=tc.zeros((1,T,6),dtype=tc.float32)
    latent_states=tc.zeros((T+1,agent.cfg.feature_dim),dtype=tc.float32)
    xs[0,0]=tc.as_tensor(obs,dtype=tc.float32)
    latent_states[0]=agent.belief.feat[0].detach().cpu().float()

    for t in range(T):
        raw=agent.sample_action_from_belief(True)
        acts[0,t]=tc.as_tensor(agent.prev_action[0].detach().cpu(),dtype=tc.float32)
        next_belief=agent.net.rssm.imagine(agent.belief,agent.prev_action,True)
        next_obs=agent.net.decoder(next_belief.feat)
        agent.belief=next_belief
        xs[0,t+1]=next_obs[0].detach().cpu().float()
        latent_states[t+1]=next_belief.feat[0].detach().cpu().float()

    return {'observations':xs,'actions':acts,'latent_states':latent_states.unsqueeze(0)}


def projection_for(reference_model,idx):
    layers=reference_model.dsr_sac_model.projection_matrix[idx]
    return (layers.weight).detach().cpu().numpy()


def build(reference_model,idx,params,obs_to_ode,ode_to_obs,derivatives,cfg,dt=.01,horizon=500):
    agent=DreamerV3Agent(cfg,projection_for(reference_model,idx))
    env=TrueDoublePendulumEnv(params,obs_to_ode,ode_to_obs,derivatives,make_reward_adapter(reference_model,idx),agent.actions,dt,horizon)
    return agent,env


def fixed_conditions(seed,count):
    rng=np.random.default_rng(seed);return [init_obs(rng) for _ in range(count)]


def validation_score(agent,env,conditions,reference_model,idx,horizon=500):
    scores=[]
    for x0 in conditions:
        res=evaluate(agent,env,x0,horizon)
        with tc.no_grad(): r=reference_model.reward_function(res['observations'][:,1:],res['actions'],idx).squeeze()
        scores.append(float(r[15:].mean().cpu()))
    return float(np.mean(scores))


def run_search(reference_model,all_params,obs_to_ode,ode_to_obs,derivatives,base_cfg,output,dt=.01,horizon=500,
               subjects=(0,5,10,15),sizes=('very_small','small','medium','large'),ratios=(16,32),tuning_steps=25_000):
    conditions=fixed_conditions(20260629,5); rows=[]
    for size in sizes:
        for ratio in ratios:
            vals=[];print(f'\n[Tuning] size={size}, train_ratio={ratio}')
            for idx in subjects:
                cfg=replace(sized_config(base_cfg,size),train_ratio=ratio,total_environment_steps=tuning_steps,seed=901_000+1000*idx+17*ratios.index(ratio)+sizes.index(size))
                agent,env=build(reference_model,idx,all_params[idx],obs_to_ode,ode_to_obs,derivatives,cfg,dt,horizon)
                train(agent,env,tuning_steps,0);score=validation_score(agent,env,conditions,reference_model,idx,horizon);vals.append(score);print(f' subject={idx}, validation_reward={score:.6f}')
            rows.append({'model_size':size,'train_ratio':ratio,'mean_validation_reward':float(np.mean(vals)),'std_validation_reward':float(np.std(vals)),'tuning_environment_steps':tuning_steps})
    rows.sort(key=lambda x:x['mean_validation_reward'],reverse=True);out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    out.with_suffix('.json').write_text(json.dumps({'best':rows[0],'all_results':rows},indent=2),encoding='utf-8')
    return rows[0],rows


if __name__ == '__main__':
    tc.set_num_threads(2); os.environ['OMP_NUM_THREADS']='2'
    sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)),'../..'))
    from src.models.mtf_sac_model import MTF_SAC_Model
    from src.main import init_dataset
    from src.environments.pendulum import ode_state_to_observation,double_pendulum_derivatives,observation_to_ode_state
    from src.paths import get_results_dir
    RESULTS_DIR, DATA_DIR = get_results_dir()
    RESULTS_PATH = os.path.join(RESULTS_DIR, "baselines", "dreamerv3")

    params_file= DATA_DIR + '/multi_task_double_pendulum/default/params.npy'
    alrnn_path=os.path.join(RESULTS_DIR + '/pendulum_sac_actuated/','params_subjects=16_q_hid1=93_pol_hid1=73_steps=29_init_temp=1p65_polyak_tau=0p00_noise=0p18_q_lr=0p03_policy_lr=0p00_q_lr_gamma=1p00_policy_lr_gamma=1p00_aq=16p22_apolicy=116p45_q_hid2=85_pol_hid2=66/001')
    with open(f'{alrnn_path}/config.json') as f: parameters=json.load(f)
    with open(f"{parameters['world_model']['path']}/config.json") as f: mtf_parameters=json.load(f)
    dataset=init_dataset(mtf_parameters,tc.device('cpu'))
    reference_model=MTF_SAC_Model(dataset,model_path=alrnn_path,device=tc.device('cpu'))
    all_params=np.load(params_file,allow_pickle=True)

    model_size = "very_small" #small
    OUT=Path(RESULTS_PATH + '/dreamerv3_pendulum_results_'+model_size);OUT.mkdir(exist_ok=True)
    TEST_HORIZON=500; DT=.01; TUNING_STEPS=25_000; FINAL_STEPS=50_000
    base=DreamerV3Config(seed=0,device='cuda:0',total_environment_steps=FINAL_STEPS,action_low=-100.,action_high=100.,action_scale=1.,discount=.99,batch_size=32,batch_length=32,imagination_horizon=15,seed_steps=2000,train_every=16)

    # Limited validation-only search; final test seeds and final test x0s are never used here.
    #best,_=run_search(reference_model,all_params,observation_to_ode_state,ode_state_to_observation,double_pendulum_derivatives,base,OUT/'dreamerv3_validation_search.csv',DT,TEST_HORIZON,(0,5,10,15),('very_small','small','medium','large'),(16,32),TUNING_STEPS)
    best = {"model_size": model_size, "train_ratio": 32} 
    print('\nSelected configuration:',best)
    final_base=replace(sized_config(base,best['model_size']),train_ratio=int(best['train_ratio']),total_environment_steps=FINAL_STEPS)
    rewards_all=np.full((5,16,5),np.nan,dtype=np.float64)
    world_model_rewards_all=np.full((5,16,5),np.nan,dtype=np.float64)

    for seed in range(5):
        print(f'\n========== DreamerV3 seed {seed} ==========')
        for idx in range(16):
            cfg=replace(final_base,seed=seed*10_000+idx)
            print(f'--- Training measurement index {idx}, seed {seed} ---')
            agent,env=build(reference_model,idx,all_params[idx],observation_to_ode_state,ode_state_to_observation,double_pendulum_derivatives,cfg,DT,TEST_HORIZON)
            logs=train(agent,env,FINAL_STEPS,5000)
            agent.save(OUT/f'dreamerv3_seed={seed}_measurement_index={idx}.pt')
            #np.savez(OUT/f'dreamerv3_training_seed={seed}_measurement_index={idx}.npz',**{k:np.asarray(v,dtype=np.float64) for k,v in logs.items()})
            rng=np.random.default_rng(1_000_000+seed*10_000+idx)
            for k in range(5):
                x0 = init_obs(rng)
                res=evaluate(agent,env,x0,TEST_HORIZON)
                applied=res['actions'][0].numpy(); assert np.allclose(applied[:,0],0.,atol=1e-6)
                den=applied[:,-1]; nz=np.abs(den)>1e-8
                #if np.any(nz): assert np.max(np.abs(np.diff(applied[nz,-2]/den[nz])))<.01
                with tc.no_grad(): r=reference_model.reward_function(res['observations'][:,1:],res['actions'],idx).squeeze()
                rewards_all[seed,idx,k]=float(r[15:].mean().cpu())
                print(f' test rollout {k+1}/5: reward after burn-in={rewards_all[seed,idx,k]:.6f}')
                world_model_res=evaluate_world_model(agent,x0,TEST_HORIZON)
                world_model_applied=world_model_res['actions'][0].numpy(); assert np.allclose(world_model_applied[:,0],0.,atol=1e-6)
                with tc.no_grad(): world_model_r=reference_model.reward_function(world_model_res['observations'][:,1:],world_model_res['actions'],idx).squeeze()
                world_model_rewards_all[seed,idx,k]=float(world_model_r[15:].mean().cpu())
                print(f' world model rollout {k+1}/5: reward after burn-in={world_model_rewards_all[seed,idx,k]:.6f}')
            np.save(OUT/'dreamerv3_pendulum_rewards_all.npy',rewards_all)
            np.save(OUT/'dreamerv3_pendulum_world_model_rewards_all.npy',world_model_rewards_all)
    np.save(OUT/'dreamerv3_pendulum_rewards_all.npy',rewards_all)
    np.save(OUT/'dreamerv3_pendulum_world_model_rewards_all.npy',world_model_rewards_all)
    print('\nDreamerV3 experiment complete.')

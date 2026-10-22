import os
from typing import List, Literal, Tuple, Union

import torch as tc
from tensorboardX import SummaryWriter
from torch.utils._foreach_utils import (
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)
from src.hyperparameter_search.terminate_run_error import TerminateRunError
from src.dataset import TimeseriesDataset
from src.evaluation.evaluator import Evaluator
from src.evaluation.sac_evaluator import SAC_Evaluator
from src.loss_handler import LossHandler
from src.models.mtf_model import MTFModel
from src.parameters import Parameters, SAC_Parameters
from src.scheduler.composite_scheduler import CompositeScheduler


class Saver:
    def __init__(
        self,
        writer: SummaryWriter,
        save_path: str,
        params: Union[Parameters, SAC_Parameters],
        dataset: TimeseriesDataset,
        model: MTFModel,
        loss_handler: LossHandler,
        device=tc.device("cpu"),
        evaluator: Union["Evaluator", "SAC_Evaluator"] = Evaluator,
    ):
        self.writer = writer
        self.save_path = save_path
        self.params = params
        self.dataset = dataset
        self.model = model
        self.loss_handler = loss_handler

        self.evaluator = evaluator(
            dataset=dataset,
            model=self.model,
            device=device,
            modality_evaluators=params["modality_specific_evaluators"],
            general_evaluators=params["general_evaluators"],
            no_modality_evaluators=params["modality_independent_evaluators"],
            loss_handler=self.loss_handler,
            pe_args=self.params["PE_args"] if evaluator == Evaluator else None,
        )

        self.evaluation_set_for_metrics = params["evaluation_set_for_metrics"]
        self.evaluation_set_for_plots = params["evaluation_set_for_plots"]

    def cheap_saving(
        self,
        epoch: int,
        loss: float,
        loss_terms,
        learning_rates: list[dict[Literal["name", "lr"], str | float]],
        value_scheduler: "CompositeScheduler",
    ):
        was_training = self.model.training
        if was_training:
            self.model.eval()

        self.save_learning_rate(learning_rates, epoch)

        self.save_value_scheduling(value_scheduler, epoch)
        self.save_loss(loss, loss_terms, epoch)

        self.save_current_grad_norms(epoch)
        self.save_max_eig_A(epoch)

        self.writer.flush()

        if was_training:
            self.model.train()

    def expensive_saving(self, epoch: int):
        was_training = self.model.training
        if was_training:
            self.model.eval()

        self.save_metrics(epoch)
        self.save_simulation_plot(epoch)
        self.save_test_loss(epoch)
        self.save_current_model(epoch)
        self.save_general_evaluators(epoch)

        self.writer.flush()

        if was_training:
            self.model.train()

    def save_learning_rate(
        self,
        learning_rates: list[dict[Literal["name", "lr"], str | float]],
        epoch: int,
    ):
        for learning_rate in learning_rates:
            self.writer.add_scalar(
                tag=f"Scheduling/Learning_Rate_{learning_rate['name']}",
                scalar_value=learning_rate["lr"],
                global_step=epoch,
            )

    def save_loss(
        self,
        loss: float,
        loss_terms: dict[str, float | dict[str, float]],
        epoch: int,
    ):
        self.writer.add_scalar(
            tag="Loss/Total",
            scalar_value=loss,
            global_step=epoch,
        )
        for loss_term_name, loss_term_value in loss_terms.items():
            self.writer.add_scalar(
                tag="Loss/" + loss_term_name,
                scalar_value=loss_term_value,
                global_step=epoch,
            )

    def save_test_loss(self, epoch: int):
        test_loss, test_loss_terms, _ = self.evaluator.compute_test_loss(epoch)
        self.writer.add_scalar(
            tag="Loss Test/Total",
            scalar_value=test_loss,
            global_step=epoch,
        )
        for loss_term_name, loss_term_value in test_loss_terms.items():
            self.writer.add_scalar(
                tag="Loss Test/" + loss_term_name,
                scalar_value=loss_term_value,
                global_step=epoch,
            )

    def save_current_model(self, epoch: int):
        """save parameters of current model (state_dict)"""
        state_dict_path = os.path.join(self.save_path, f"model_{epoch}.pt")
        current_state_dict = self.model.state_dict()
        if not os.path.exists(self.save_path):
            raise TerminateRunError(f"Save path {self.save_path} does not exist")
        tc.save(current_state_dict, state_dict_path)

    def save_current_grad_norms(self, epoch: int):
        """save norm of gradients of current model to tensorboard

        Credits: PyTorch.nn.utils.clip_grad_norm_ code
        """
        # this copies nn.utils.clip_grad_norm_ code without the actual clipping
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        first_device = grads[0].device
        grouped_grads: dict[Tuple[tc.device, tc.dtype], List[List[tc.Tensor]]] = (
            _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
        )  # type: ignore[assignment]

        norms = []
        for (device, _), ([grads], _) in grouped_grads.items():  # type: ignore[assignment]
            if _has_foreach_support(grads, device=device):
                norms.extend(tc._foreach_norm(grads, 2.0))
            else:
                norms.extend([tc.linalg.vector_norm(g, 2.0) for g in grads])

        total_gradient_norm = tc.linalg.vector_norm(
            tc.stack([norm.to(first_device) for norm in norms]), 2.0
        )

        # save total norm of gradient
        self.writer.add_scalar(
            tag="Norm/total_grad_norm",
            scalar_value=total_gradient_norm.item(),
            global_step=epoch,
        )

    def save_max_eig_A(self, epoch: int):
        """save median of max eigenvalues of A to tensorboard"""
        As = self.model.dsr_model.construct_params(
            subject_index=tc.arange(0, self.model.dsr_model.num_subjects)
            if self.model.dsr_model.num_subjects is not None
            else None,
        )["A"]
        max_eigenvalues, _ = tc.max(As, dim=1)
        self.writer.add_scalar(
            tag="Norm/median_max_eigenvalue(A)",
            scalar_value=tc.median(max_eigenvalues),
            global_step=epoch,
        )

    def save_value_scheduling(self, scheduler: "CompositeScheduler", epoch: int):
        for (
            name,
            value_scheduler,
        ) in scheduler.value_scheduler.schedulers.items():
            if value_scheduler.is_constant:
                continue
            self.writer.add_scalar(
                tag=f"Scheduling/{name}",
                scalar_value=value_scheduler.value,
                global_step=epoch,
            )

    def save_metrics(self, epoch: int):
        if self.evaluation_set_for_metrics == "test" or self.evaluation_set_for_metrics == "both":
            self.evaluator.compute_metrics(
                summary_writer=self.writer,
                evaluate_on="test",
                epoch=epoch,
            )
        if self.evaluation_set_for_metrics == "train" or self.evaluation_set_for_metrics == "both":
            self.evaluator.compute_metrics(
                summary_writer=self.writer,
                evaluate_on="train",
                epoch=epoch,
            )

    def save_simulation_plot(self, epoch: int):
        if self.evaluation_set_for_plots == "test" or self.evaluation_set_for_plots == "both":
            self.evaluator.get_tensorboard_figures(self.writer, "test", epoch)

        if self.evaluation_set_for_plots == "train" or self.evaluation_set_for_plots == "both":
            self.evaluator.get_tensorboard_figures(self.writer, "train", epoch)

    def save_general_evaluators(self, epoch: int):
        self.evaluator.get_general_evaluator_outputs(self.writer, epoch)

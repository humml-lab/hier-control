import torch as tc
import torch.nn.functional as F

from src.models.dsr_model.models.shallow_plrnn.base import ShallowPLRNNStep


class ClippedShallowPLRNNStep(ShallowPLRNNStep):
    @classmethod
    def forward(cls, z, s, params, hyperparameters, **kwargs):
        zw = tc.einsum("bij,bj->bi", params["W2"], z)
        next_z = (
            z * params["A"]
            + tc.einsum("bij,bj->bi", params["W1"], F.relu(zw + params["h2"]) - F.relu(zw))
            + params["h1"]
        )

        if params["C"] is not None:
            next_z += tc.einsum("bij,bj->bi", params["C"], s)

        return next_z

    @classmethod
    def jacobian(cls, z, params, hyperparameters):
        # A: (B, M)
        # W1: (B, M, L)
        # W2: (B, L, M)
        # h2: (B, L)
        A, W1, W2, h2 = (
            params["A"],
            params["W1"],
            params["W2"],
            params["h2"],
        )

        W2z = (W2 @ z.unsqueeze(2)).squeeze(2)  # B x L
        D = (W2z > -h2).float() - (W2z > 0).float()  # B x L
        # W1 * D * W2
        J = (W1 * D.unsqueeze(1)) @ W2

        # add A in-place, J = A + W1 * D * W2
        J.diagonal(dim1=-2, dim2=-1).add_(A)
        return J

import torch as tc
import torch.nn as nn


class CommonLinearProjectionSchemeUtils:
    @staticmethod
    def init_projection_matrix(
        *dims: int,
        gain: float = 0.1,
    ) -> tc.Tensor:
        return nn.init.xavier_uniform_(
            tc.empty(*dims),
            gain=gain,
        )

    @staticmethod
    def init_subject_vector(
        dim: int,
    ) -> tc.Tensor:
        return nn.init.uniform_(tc.empty(1, dim), -1, 1)

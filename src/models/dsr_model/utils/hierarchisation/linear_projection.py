import torch as tc

from .common_linear_projection import CommonLinearProjectionSchemeUtils


class LinearProjectionSchemeUtils:
    @staticmethod
    def init_projection_matrix(
        *dims: int,
        gain: float = 0.1,
    ) -> tc.Tensor:
        return CommonLinearProjectionSchemeUtils.init_projection_matrix(
            *dims,
            gain=gain,
        )

    @staticmethod
    def init_subject_vector(
        dim: int,
    ) -> tc.Tensor:
        return CommonLinearProjectionSchemeUtils.init_subject_vector(dim)

    @staticmethod
    def construct_param(
        projection_matrix: tc.Tensor,
        subject_vector: tc.Tensor,
    ) -> tc.Tensor:
        return tc.einsum("sp,p...->s...", subject_vector, projection_matrix)

from typing import Optional

import torch as tc

from .common_linear_projection import CommonLinearProjectionSchemeUtils


class OuterProductSchemeUtils:
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
        P: Optional[tc.Tensor],
        Q: tc.Tensor,
        R: tc.Tensor,
        subject_vector: tc.Tensor,
    ) -> tc.Tensor:
        if P is None:
            return tc.einsum(
                "sp,p...->s...",
                tc.einsum("bp,pr->br", subject_vector, R),
                Q,
            )
        else:
            return tc.einsum(
                "mr,br,rk->bmk",
                P,
                tc.einsum("bp,pr->br", subject_vector, R),
                Q,
            )

import math

import torch as tc
import torch.nn as nn


class CommonInitializationUtils:
    @staticmethod
    def init_weight_matrix(dim_1: int, dim_2: int) -> tc.Tensor:
        r = 1 / math.sqrt(dim_2)
        return nn.init.uniform_(tc.empty(dim_1, dim_2), -r, r)

    @staticmethod
    def init_external_matrix(external_dim: int, latent_dim: int) -> tc.Tensor:
        if external_dim > latent_dim:
            print(
                "Warning: external_dim > latent_dim. External input matrix will be initialized as a weight matrix. See dsr_hierarchisation_utils.py for more details."
            )
            return CommonInitializationUtils.init_weight_matrix(
                latent_dim,
                external_dim,
            )
        return tc.cat(
            [
                tc.zeros(
                    latent_dim - external_dim,
                    external_dim,
                ),
                tc.eye(
                    external_dim,
                ),
            ]
        )

    @staticmethod
    def init_AW_values(
        dim: int,
    ) -> tuple[tc.Tensor, tc.Tensor]:
        """
        Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network
        with ReLU Nonlinearity https://arxiv.org/abs/1511.03771.
        """
        matrix_random = tc.randn(dim, dim)
        matrix_positive_normal = (1 / dim) * matrix_random.T @ matrix_random
        matrix = tc.eye(dim) + matrix_positive_normal
        max_ev = tc.max(tc.abs(tc.linalg.eigvals(matrix)))
        matrix_spectral_norm_one = matrix / max_ev
        A = matrix_spectral_norm_one.diag()
        W = matrix_spectral_norm_one - tc.diag(A)
        return A, W

    @staticmethod
    def init_bias(dim: int) -> nn.Parameter:
        r = 1.0 / (dim**0.5)
        return nn.init.uniform_(tc.empty(dim), -r, r)

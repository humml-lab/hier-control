import torch as tc


class ALRNNInitializationUtils:
    @staticmethod
    def generate_normalized_positive_definite_matrix(dim: int) -> tc.Tensor:
        R = tc.randn(dim, dim)
        K = (R.T @ R) / dim + tc.eye(dim)
        eigenvalues = tc.linalg.eigvals(K)
        lambda_max = tc.max(tc.abs(eigenvalues))
        return K / lambda_max

    @staticmethod
    def init_AWh(dim: int) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
        PosDefMatrix = ALRNNInitializationUtils.generate_normalized_positive_definite_matrix(dim)
        A = tc.diagonal(PosDefMatrix)
        W = tc.randn(dim, dim) * 0.01
        h = tc.zeros(dim)
        return A, W, h

from typing import Optional


class TerminateRunError(Exception):
    """
    Exception to terminate a hyperparameter search run.

    Useful as this error is caught when searching using optuna, so the training is not stopped but the run is marked as bad.
    """

    def __init__(self, message: Optional[str] = None):
        self.message = message or "Hyperparameter search run terminated."
        super().__init__(self.message)

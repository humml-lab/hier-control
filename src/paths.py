import os
from pathlib import Path


def get_results_dir():
    return (str(Path(
        os.environ.get(
            "HIER_CONTROL_RESULTS_DIR",
            Path.cwd() / "results",
        )
    ).expanduser().resolve()),
    str(Path(
        os.environ.get(
            "HIER_CONTROL_DATA_DIR",
            Path.cwd() / "data",
        )
    ).expanduser().resolve()))
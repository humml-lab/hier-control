import optuna
import sqlite3
import shutil
import sys, os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_rel
from scipy.stats import pearsonr
import shutil
import torch as tc
import torch.nn as nn
import numpy as np
import json
from sklearn.decomposition import PCA
sys.path.append(os.path.dirname(os.path.abspath('')))
from src.hyperparameter_search.value_formatter import ValueFormatter
from src.evaluation.metrics.dstsp.klx_bin import klx_bin
from src.evaluation.metrics.dstsp.klx_gmm import state_space_divergence_gmm
from src.evaluation.metrics.pse import power_spectrum_error

import math
import re
import string
from pathlib import Path
from typing import Sequence
import fitz  # PyMuPDF
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import Bbox
from src.paths import get_results_dir

RESULTS_DIR, DATA_DIR = get_results_dir()
_SIZE_RE = re.compile(r"_width_(?P<width>\d+(?:\.\d+)?)_height_(?P<height>\d+(?:\.\d+)?)")
FONT_SIZE = 12
FIG_DIR = "figures"

def save_fig(fig, name, dpi=300):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get the tight bounding box in figure coordinates (inches)
    bbox = fig.get_tightbbox(renderer).transformed(
        fig.dpi_scale_trans.inverted())
    width_inch, height_inch = bbox.width, bbox.height

    path = "figures_new/" + name + "_width_{}_height_{}.pdf".format(
        int(width_inch * fig.dpi * 10),
        int(height_inch * fig.dpi * 10))
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print("Figure saved to {}".format(path))


def get_trial_by_number(trials: list[optuna.trial.Trial], trial_number: int) -> optuna.trial.Trial:
    for trial in trials:
        if trial.number == trial_number:
            return trial
    raise ValueError(f"Trial with number {trial_number} not found.")

# Identify failed trials
def remove_failed_trials(study: optuna.study.Study, storage_url: str):
    failed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.FAIL
    ]

    # Remove them from the database
    conn = sqlite3.connect(storage_url.replace('sqlite:///', ''))
    cursor = conn.cursor()

    for trial in failed_trials:
        # Delete trial records from related tables
        cursor.execute(
            "DELETE FROM trial_params WHERE trial_id = ?", (trial._trial_id,))
        cursor.execute(
            "DELETE FROM trial_values WHERE trial_id = ?", (trial._trial_id,))
        cursor.execute(
            "DELETE FROM trial_intermediate_values WHERE trial_id = ?", (trial._trial_id,))
        cursor.execute(
            "DELETE FROM trial_system_attributes WHERE trial_id = ?", (trial._trial_id,))
        cursor.execute(
            "DELETE FROM trial_user_attributes WHERE trial_id = ?", (trial._trial_id,))
        # Finally delete the trial itself
        cursor.execute("DELETE FROM trials WHERE trial_id = ?",
                       (trial._trial_id,))

    conn.commit()
    conn.close()

    print(f"Removed {len(failed_trials)} failed trials")

# Create a custom visualization for pruning
def plot_pruning_history(study):
    pruned_x, pruned_y = [], []
    complete_x, complete_y = [], []

    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.PRUNED and trial.intermediate_values:
            last_step = max(trial.intermediate_values.keys())
            last_value = trial.intermediate_values[last_step]
            pruned_x.append(trial.number)
            pruned_y.append(last_value)
        elif trial.state == optuna.trial.TrialState.COMPLETE:
            complete_x.append(trial.number)
            complete_y.append(trial.value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=complete_x, y=complete_y,
                             mode='markers', name='Completed',
                             marker=dict(color='green', size=8)))
    fig.add_trace(go.Scatter(x=pruned_x, y=pruned_y,
                             mode='markers', name='Pruned',
                             marker=dict(color='red', size=8, symbol='x')))

    fig.update_layout(title="Trial Outcomes: Completed vs Pruned",
                      xaxis_title="Trial Number",
                      yaxis_title="Objective Value")
    return fig

def remove_pruned_trial_folders(study: optuna.study.Study, study_path: str):
    pruned_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]

    # Remove folders associated with pruned trials
    for trial in pruned_trials:
        folder_name = "params_" + "_".join([(f"{key}={ValueFormatter._formatted_value(value)}")
                                            for key, value in trial.params.items()])
        folder_path = os.path.join(study_path, folder_name)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Removed folder: {folder_path}")
            except Exception as e:
                print(f"Error removing folder {folder_path}: {e}")
        else:
            print(f"Folder not found, skipping: {folder_path}")


def optuna_overview(storage_url: str, plot=True, remove_trials=False) -> tuple[optuna.trial.Trial, optuna.trial.Trial]:

    study_names = optuna.study.get_all_study_names(storage_url)
    print("Available studies in the database:", study_names)

    study = optuna.load_study(
        # Use the name from the list above, or your custom name
        study_name=study_names[0],
        storage=storage_url
    )
    
    # find the smallest intemediate value of all trials
    idx_best_intermediate_trial = None
    min_intermediate_value = float('inf')
    for trial in study.trials:
        if trial.intermediate_values:
            if trial.value is None:
                value = tc.inf
            else:
                value = trial.value
            trial_min = min(min(trial.intermediate_values.values()), value)
            if trial_min < min_intermediate_value:
                min_intermediate_value = trial_min
                idx_best_intermediate_trial = trial.number
                best_intermediate_trial = trial
                
    
    if plot:
        try:
            optuna.visualization.plot_optimization_history(study).show()
        except Exception as e:
            print(f"Could not plot optimization history: {e}")
        try:
            optuna.visualization.plot_param_importances(study).show()
        except Exception as e:
            print(f"Could not plot parameter importances: {e}")
        optuna.visualization.plot_parallel_coordinate(study).show()
        #optuna.visualization.plot_contour(study).show()
        optuna.visualization.plot_slice(study).show()
        plot_pruning_history(study).show()
    
    print(f"Best intermediate value: {min_intermediate_value:.4f}, Number: {idx_best_intermediate_trial}, epoch: {list(best_intermediate_trial.intermediate_values.keys())[tc.argmin(tc.as_tensor(list(best_intermediate_trial.intermediate_values.values()))).item()]}")
    try:
        idx_best_trial = study.best_trial.number
    except ValueError:
        return None, study, None, None
        
    print(f"Number of trials: {len(study.trials)}")


    try:
        param_importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        param_importances = None


    if remove_trials:
        remove_failed_trials(study, storage_url)
    print(
        f"Best Value: {study.best_trial.value:.4f}, Number: {idx_best_trial}")

    return param_importances, study, idx_best_trial, idx_best_intermediate_trial

def compute_value_function(model, state, measurement_index, cumulative_timeseries_index, external_inputs=None, n_samples=1):
    policy_params = model.dsr_sac_model.sac_construct_params(cumulative_timeseries_index, model.dsr_sac_model.policy_params, "policy")
    with tc.no_grad():
        if external_inputs is not None:
            state = tc.cat([state, external_inputs], dim=-1)
        action, ll = model.dsr_sac_model.sample_action(state.tile(n_samples, 1), policy_params, measurement_index)
        q1, q2 = model.dsr_sac_model.compute_q_values(state.tile(n_samples, 1), action, cumulative_timeseries_index)
        q = tc.min(tc.stack([q1, q2], dim=0), dim=0).values
        if model.dsr_sac_model.hyperparameters["policy_hierarchisation_scheme"]["scheme"] == "none":
            log_temperature = model.dsr_sac_model.log_temperature
        else:
            log_temperature = model.dsr_sac_model.log_temperature[measurement_index]
        v = (q- log_temperature.exp()*ll).reshape((n_samples, -1)).mean(dim=0)
    return v

def compute_q(model, state, action, cumulative_timeseries_index):
    with tc.no_grad():
        q1, q2 = model.dsr_sac_model.compute_q_values(state, action, cumulative_timeseries_index)
        q = tc.min(q1, q2)
    return q

def compute_actived_linear_subregions(latent_trajectory, relu_units):
    activated_units = tc.nn.functional.relu(latent_trajectory[:, -relu_units:]).detach()>0
    all_linear_subregions = tc.unique(activated_units, dim=0)
    sorting_indices = tc.argsort(all_linear_subregions.sum(dim=1), descending=False)
    all_linear_subregions = all_linear_subregions[sorting_indices]

    all_idx = []
    for linear_subregion in activated_units:
        # get index of linear subregion
        idx = tc.argwhere((linear_subregion==all_linear_subregions).all(dim=1)).item()
        all_idx.append(idx)

    return tc.as_tensor(all_idx), all_linear_subregions

def repeat_trial(study: optuna.study.Study, trial_folder_name: str = None, trial_number: int=None, n_repeats: int=1):
    assert (trial_folder_name is not None) ^ (trial_number is not None), "Provide either trial_folder_name or trial_number, not both."
    if trial_folder_name is not None:
        for trial in study.trials:
            folder_name = "params_" + "_".join([(f"{key}={ValueFormatter._formatted_value(value)}")
                                                for key, value in trial.params.items()])
            if folder_name == trial_folder_name:
                break
    elif trial_number is not None:
        trial = get_trial_by_number(study.trials, trial_number)
        
    for i in range(n_repeats):
        study.enqueue_trial(trial.params)
        
        
def comp_lyapunov_spectrum(jacobians: tc.Tensor) -> tc.Tensor:
    # apply qr decomposition method
    Q = tc.eye(jacobians.shape[1])
    spectrum = tc.zeros(jacobians.shape[1])
    for i in range(jacobians.shape[0]):
        Q = jacobians[i] @ Q
        Q, R = tc.linalg.qr(Q)
        spectrum += tc.log(tc.abs(tc.diag(R)))
    return spectrum / jacobians.shape[0]


def _read_size_from_name(path: str | Path) -> tuple[float, float]:
    match = _SIZE_RE.search(Path(path).stem)
    if not match:
        raise ValueError(
            f"Could not parse '_width_<number>_height_<number>' from filename: {path}"
        )
    return float(match.group("width")), float(match.group("height"))


def _pdf_page_size_pt(path: str | Path) -> tuple[float, float]:
    """Return the visible page width and height in PDF points."""
    with fitz.open(path) as doc:
        if len(doc) == 0:
            raise ValueError(f"PDF has no pages: {path}")
        page = doc[0]
        rect = page.rect
        return float(rect.width), float(rect.height)


def align_pdf_figures(
    pdf_files: Sequence[str | Path],
    output_pdf: str | Path,
    *,
    rows: int = 2,
    gap: float = 0.10,
    label_gap: float = 0.035,
    label_fontsize: float = 11,
    page_width: float = 7.0,
    max_page_height: float | None = None,
    order: str = "input",
) -> Path:
    """
    Arrange PDF figures into a paper-ready multi-panel PDF.

    Parameters
    ----------
    pdf_files:
        Input PDF filenames. Names must contain e.g. '_width_57_height_47'.
        Each PDF's first page is used.
    output_pdf:
        Output PDF path.
    rows:
        Desired number of rows. The function chooses the number of columns
        and scales all panels by one common factor, preserving aspect ratios.
    gap:
        Horizontal and vertical gap between panels, in inches.
    label_gap:
        Gap between a panel's top edge and its label, in inches.
    label_fontsize:
        Size of panel labels in points.
    page_width:
        Output page width in inches.
    max_page_height:
        Optional maximum output page height in inches. If set, all panels are
        uniformly scaled down as needed to fit.
    order:
        "input" preserves the supplied order; "name" sorts by filename.

    Returns
    -------
    Path to the created PDF.

    Notes
    -----
    - The dimensions encoded in the filenames are used to choose the layout.
    - The actual PDF page dimensions determine each panel's displayed aspect ratio.
    - All panels receive exactly the same scale factor. This is what preserves
      the relative size of the original 14 pt text.
    - Labels are added as vector text; the original PDF page is inserted as
      vector content, not rasterized.
    """
    if not pdf_files:
        raise ValueError("pdf_files must contain at least one PDF.")
    if rows < 1:
        raise ValueError("rows must be at least 1.")
    if gap < 0 or label_gap < 0:
        raise ValueError("gap and label_gap must be non-negative.")
    if page_width <= 0:
        raise ValueError("page_width must be positive.")
    if max_page_height is not None and max_page_height <= 0:
        raise ValueError("max_page_height must be positive.")

    files = [Path(p) for p in pdf_files]
    if order == "name":
        files.sort(key=lambda p: p.name)
    elif order != "input":
        raise ValueError("order must be 'input' or 'name'.")

    if any(not p.exists() for p in files):
        missing = [str(p) for p in files if not p.exists()]
        raise FileNotFoundError(f"Input PDF(s) not found: {missing}")

    # Encoded dimensions are used for layout estimation; PDF page geometry
    # is used for the actual aspect ratio.
    encoded_sizes = [_read_size_from_name(p) for p in files]
    page_sizes = [_pdf_page_size_pt(p) for p in files]
    aspect_ratios = [w / h for w, h in page_sizes]

    n = len(files)
    rows = min(rows, n)
    cols = math.ceil(n / rows)

    # Keep the same scale for every original PDF page.
    # First estimate the scale needed to fit the requested page width.
    label_h = (label_fontsize / 72.0) + label_gap
    available_width = page_width - (cols - 1) * gap
    if available_width <= 0:
        raise ValueError("The requested gaps leave no room for the panels.")

    # Choose a common panel height scale from width constraints.
    # Each panel width = scale * original page width in inches.
    widths_in = [w / 72.0 for w, _ in page_sizes]
    heights_in = [h / 72.0 for _, h in page_sizes]
    row_heights = []
    idx = 0
    for r in range(rows):
        row_count = min(cols, n - idx)
        row_heights.append(max(heights_in[idx:idx + row_count]) + label_h)
        idx += row_count

    # The width limit is based on the sum of the widest panels in each row.
    # Evaluate candidate scale factors for a near-full-width layout.
    row_widths = []
    idx = 0
    for r in range(rows):
        row_count = min(cols, n - idx)
        row_widths.append(sum(widths_in[idx:idx + row_count]) + gap * (row_count - 1))
        idx += row_count
    scale_w = min(available_width / w for w in row_widths)

    # Scale by the same factor vertically; labels sit just above each panel.
    total_height_at_scale = sum(
        max(heights_in[sum(min(cols, n - j) for j in range(r)):sum(min(cols, n - j) for j in range(r + 1))])
        for r in range(rows)
    ) if rows else 0
    # Simpler exact row-height calculation in original-inch units.
    row_panel_heights = []
    idx = 0
    for r in range(rows):
        row_count = min(cols, n - idx)
        row_panel_heights.append(max(heights_in[idx:idx + row_count]))
        idx += row_count

    # Label height is not scaled with the PDF, since it is a separate caption.
    total_h_at_scale = sum(row_panel_heights) * scale_w + rows * label_h + (rows - 1) * gap
    if max_page_height is not None and total_h_at_scale > max_page_height:
        scale_h = (max_page_height - rows * label_h - (rows - 1) * gap) / sum(row_panel_heights)
        if scale_h <= 0:
            raise ValueError("max_page_height is too small for the labels and gaps.")
        scale = min(scale_w, scale_h)
    else:
        scale = scale_w

    panel_widths = [w * scale for w in widths_in]
    panel_heights = [h * scale for h in heights_in]

    # Output height follows the actual layout.
    row_panel_heights_scaled = []
    idx = 0
    for r in range(rows):
        row_count = min(cols, n - idx)
        row_panel_heights_scaled.append(max(panel_heights[idx:idx + row_count]))
        idx += row_count
    page_height = sum(row_panel_heights_scaled) + rows * label_h + (rows - 1) * gap
    page_height = max(page_height, 1.0)

    output_pdf = Path(os.path.join(FIG_DIR, output_pdf))
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary Matplotlib PDF page for the labels, then import each
    # original PDF page into the final PDF at its calculated vector rectangle.
    # PyMuPDF's show_pdf_page preserves vector paths and text.
    final_doc = fitz.open()
    out_page = final_doc.new_page(width=page_width * 72, height=page_height * 72)

    labels = [f"{string.ascii_lowercase[i]}" if i < 26 else f"{i + 1}"
              for i in range(n)]

    x_positions = []
    y_positions = []
    idx = 0
    y_top = 0.0
    for r in range(rows):
        row_count = min(cols, n - idx)
        # Center each row within the page, allowing a shorter last row.
        row_w = sum(panel_widths[idx:idx + row_count]) + gap * (row_count - 1)
        x = (page_width - row_w) / 2
        row_h = row_panel_heights_scaled[r]
        for j in range(row_count):
            panel_i = idx + j
            x_positions.append(x)
            # Align the bottom edges of all panels in this row.
            y_positions.append(y_top + label_h + row_h - panel_heights[panel_i])
            x += panel_widths[panel_i] + gap
        y_top += label_h + row_h + gap
        idx += row_count

    for i, path in enumerate(files):
        # PDF coordinates start at top-left in PyMuPDF.
        x0 = x_positions[i] * 72
        y0 = y_positions[i] * 72
        w = panel_widths[i] * 72
        h = panel_heights[i] * 72
        target = fitz.Rect(x0, y0, x0 + w, y0 + h)
        with fitz.open(path) as src:
            out_page.show_pdf_page(target, src, 0, keep_proportion=True)

        # Labels share a common top baseline within each row, regardless
        # of each panel's height. Panels themselves remain bottom-aligned.
        row_index = next(
            r for r, start in enumerate(
                [sum(min(cols, n - j) for j in range(k)) for k in range(rows)]
            )
            if start <= i < start + min(cols, n - r * cols)
        )
        label_y = (sum(row_panel_heights_scaled[:row_index])
                   + row_index * (label_h + gap)
                   + label_fontsize / 72.0) * 72
        out_page.insert_text(
            fitz.Point(x0, label_y),
            labels[i],
            fontsize=label_fontsize,
            fontname="heit",
            color=(0, 0, 0),
        )

    final_doc.save(output_pdf, garbage=4, deflate=True)
    final_doc.close()
    return output_pdf
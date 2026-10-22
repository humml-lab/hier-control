# Hier-Control

`Hier-Control` is a research codebase for learning hierarchical latent dynamical
systems from multiple modalities. It contains the main Hierarchical
Multimodal Teacher Forcing (MTF) models, supervised and Soft Actor-Critic (SAC)
training pipelines, evaluation utilities, preprocessing notebooks, and
baseline controllers.

The experiments cover synthetic and control-oriented settings including
hierarchical Lorenz systems, pendulum data, multi-task data, and macaque
behavioral recordings from maze experiments. The repository is organized around
reproducible experiment configurations
rather than a command-line interface: the `ubermain_*.py` files define the dataset, model and hyperparameters.

## Repository layout

```text
src/
  Training loops, datasets, replay buffers, models, schedulers, and evaluators
ubermain*.py
  Experiment entry points and configurations for MTF and SAC experiments
neuro_preprocessing/
  Data-preparation notebooks and preprocessed example datasets, including
  macaque recordings
evaluation/
  Evaluation notebooks, metric helpers, and generated figures and videos
baselines/
  CEM-MPC, DreamerV3, and TD-MPC baseline implementations
environment.yml
  Conda environment definition
```

`src/` includes:

- `src/models/`: MTF, SAC, encoder, decoder, and DSR model components.
- `src/bptt.py`: training with backpropagation through time.
- `src/sac.py`: Soft Actor-Critic training.
- `src/evaluation/`: modality-specific, modality-independent, latent-space,
  prediction-error, and control evaluation.
- `src/hyperparameter_search/`: grid-search and Optuna-based experiment
  orchestration.

## Requirements

- Python 3.13, as specified by `.python-version`.
- Conda or another environment manager capable of creating the environment
  from `environment.yml`.
- PyTorch 2.8 or newer. GPU runs additionally require a compatible NVIDIA
  driver and CUDA runtime.

Create and activate the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate hier-control
```

The repository uses the environment variables
`HIER_CONTROL_RESULTS_DIR` and `HIER_CONTROL_DATA_DIR` for machine-specific
locations. They should point to the root directory for experiment outputs and
the root directory containing the prepared datasets, respectively:

```bash
export HIER_CONTROL_RESULTS_DIR=/path/to/results
export HIER_CONTROL_DATA_DIR=/path/to/data
```

If they are not set, the current path helper falls back to `results/` and
`data/` below the current working directory. Set these variables explicitly
when running the supplied configurations so that data and previous checkpoints
are resolved consistently.

## Training pipeline

The control experiments use a two-stage training pipeline:

1. **Train a DSR world model.** Use an MTF launcher such as
   `ubermain_hier_lorenz_.py`, `ubermain_pendulum.py`, or
   `ubermain_mc_multitask.py`. These experiments learn a latent dynamical
   system and its modality-specific decoders from the prepared measurements.
   The run saves the model checkpoints and its configuration in the results
   directory.
2. **Train SAC using the learned world model.** Use the corresponding
   `ubermain_sac_*.py` launcher. Its configuration points to the pretrained DSR
   run through the `world_model.path` setting. SAC then uses the learned
   latent dynamics for control and saves its policy, value functions, and
   evaluation outputs as a new run.

The fine-tuning launchers extend an existing DSR or SAC run rather than
starting from scratch. They require the path to the previous run and its
configuration, so they should only be used after the relevant first-stage
experiment has completed successfully.

## Running an experiment

Run launchers from the repository root. Each launcher has a configuration
section near the top of the file where you can select the data, modalities,
model dimensions, loss coefficients, number of epochs, device settings, and
hyperparameter-search options.

For example, a typical hierarchical Lorenz control workflow is:

```bash
python3 ubermain_hier_lorenz_.py
python3 ubermain_sac_hier_lorenz.py
```

The same pattern applies to the pendulum and macaque experiments:

```bash
python3 ubermain_pendulum.py
python3 ubermain_sac_pendulum.py

python3 ubermain_mc_multitask.py
python3 ubermain_sac_mc_multitask.py
```

Before running an SAC launcher, update its `world_model.path` to the DSR
checkpoint directory produced by the first-stage run. The launcher settings
also determine whether training uses a grid search or Optuna, how many runs
are started, and whether GPU acceleration is enabled.

Before starting a run:

1. Set `HIER_CONTROL_DATA_DIR` and `HIER_CONTROL_RESULTS_DIR`.
2. Check the `base_path` and `measurement_folders` in the selected launcher.
3. For GPU training, set `use_gpu`, `device_ids`, and
   `num_processes_per_gpu` appropriately.
4. Choose whether to use a grid search or Optuna by setting `use_optuna`.

Training writes checkpoints, configuration files, TensorBoard data, and
evaluation outputs below the configured results directory. The exact directory
layout is generated from the experiment name and parameter configuration.

## Preprocessing and evaluation

The notebooks in `neuro_preprocessing/` prepare the supported datasets,
including hierarchical Lorenz, double pendulum, macaque maze, and multi-task
data. Run them in dependency order when starting from raw data;
otherwise point the experiment launcher at an existing prepared dataset.

The notebooks in `evaluation/` load saved runs and calculate reconstruction,
latent-space, prediction-error, dynamical-system, and control metrics. The
`evaluate_sac_*.ipynb` notebooks are the SAC-specific counterparts of the
standard evaluation notebooks. `evaluation/evaluation_utils.py` provides
shared result aggregation, statistical summaries, and figure-generation
helpers.

## Baselines

The `baselines/` directory contains independent comparison implementations:

- `cem_mpc/`: Cross-Entropy Method model-predictive control.
- `dreamerv3/`: DreamerV3-style latent world-model control.
- `td_mpc/`: TD-MPC-style model-predictive control.

These scripts use the environments in `src/environments/` and may require
their own pretrained models or prepared data paths in addition to the common
configuration above.

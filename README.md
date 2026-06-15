# Hybrid Biophysical-Neural ODE Models

Code accompanying "Learning Hybrid Biophysical Neuron Models with Neural ODEs". This repository contains the implementation of hybrid models that combine biophysical Hodgkin-Huxley (HH) ion channel dynamics with neural ODEs (NODEs), along with all experiments and figures from the paper.

## Setup

Requires Python 3.12+. We recommend [uv](https://docs.astral.sh/uv/) for environment management:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

Key dependencies: `jax`, `equinox`, `diffrax`, `optax`, `jaxley`.

## Repository Structure

```
├── hybrid_models/          # Core library
│   ├── hh/
│   │   ├── base.py         # HH model, integration, compartment logic
│   │   └── channels.py     # Channel implementations (Na, K, Leak, NODE, Omni, ...)
│   ├── optimizers.py       # Custom optimizers (hybrid ODE+NN optimizer)
│   ├── transforms.py       # Parameter transforms (logistic, sigmoid)
│   └── utils.py            # Data loading, training utilities
│
├── scripts/
│   ├── configs/
│   │   ├── hh_configs.py   # Experiment configs for HH channel fitting
│   │   └── gates_configs.py # Experiment configs for ICG gate fitting
│   ├── train_channel.py    # Train hybrid NODE channel on HH voltage traces
│   ├── train_biophysics.py # Train biophysics-only models
│   ├── train_gates.py      # Train NODE/Omni on ICG channel kinetics data
│   ├── simulate_voltage.py # Generate synthetic HH voltage trace data
│   ├── simulate_gates.py   # Generate synthetic ion channel gate data
│   └── evaluate_runtimes.py # Benchmark integration runtimes
│
├── notebooks/
│   ├── 29_figure2.ipynb    # Figure 2: ICG ion channel fitting
│   ├── 30_figure3.ipynb    # Figure 3: HH K-channel fitting, noise robustness
│   ├── 31_figure4.ipynb    # Figure 4: Multicompartment neuron fitting
│   ├── 32_figure1.ipynb    # Figure 1: Schematic overview
│   ├── 33_figure3_app.ipynb # Appendix Figure: Na-channel architecture sweep
│   └── 34_tables.ipynb     # Tables: Architecture comparison & runtime benchmarks
│
├── data/
│   ├── hh_synth/           # Synthetic HH voltage trace data
│   └── icg_channels/       # Ion channel kinetics data (ICG database)
│
├── results/
│   ├── sweeps/             # Pre-computed experiment sweep results
│   │   ├── icg11/                      # ICG channel fitting sweep (Figure 2)
│   │   ├── hh_channel_obs_noise_sweep4/ # K-channel obs. noise sweep (Figure 3)
│   │   ├── hh_channel_init_noise_sweep5/ # K-channel init noise sweep (Table)
│   │   ├── hh_channel_na_arch_sweep/   # Na-channel architecture sweep (App. Figure)
│   │   ├── hh_channel_arch_sweep3/     # K-channel architecture sweep (Table)
│   │   └── multicomp_sweep2/           # Multicomp NODE sweep (Figure 4)
│   ├── runtimes_hh_k.csv   # K-channel runtime benchmarks (Table)
│   └── runtimes_multicomp.csv # Multicomp runtime benchmarks (Table)
│
└── other/
    ├── swc_files/          # Neuron morphology files
    └── load_icg_data/      # Scripts for loading ICG channel data
```

## Data and Pre-computed Results

Training data and pre-computed experiment results are hosted on Zenodo and must be downloaded before running the notebooks:

- **[DATA.md](DATA.md)** — download instructions for `data/` (synthetic HH traces + ICG channel kinetics)
- **[RESULTS.md](RESULTS.md)** — download instructions for `results/` (trained model weights + sweep outputs)

After downloading and extracting both archives to the project root, reproduce any figure with:

## Reproducing Figures

```bash
jupyter notebook notebooks/
```

| Notebook | Output |
|---|---|
| `29_figure2.ipynb` | Figure 2 (ICG channel fitting) |
| `30_figure3.ipynb` | Figure 3 (K-channel noise robustness) |
| `31_figure4.ipynb` | Figure 4 (Multicompartment fitting) |
| `32_figure1.ipynb` | Figure 1 (Schematic) |
| `33_figure3_app.ipynb` | Appendix Figure (Na-channel architectures) |
| `34_tables.ipynb` | Tables (architectures + runtimes) |

## Running Experiments

### Generate synthetic training data

```bash
cd scripts
python simulate_voltage.py --config generate_hh_multi_spike
python simulate_voltage.py --config generate_hh_multi_spike_batch
```

### Train a hybrid channel model (NODE replacing K+ channel)

```bash
cd scripts
python train_channel.py --config train_hybrid_on_hh_k -v
```

### Train on the multicompartment neuron

```bash
cd scripts
python train_channel.py --config train_hybrid_on_multicomp -v
```

### Train on ICG ion channel kinetics data

```bash
cd scripts
python train_gates.py --config train_omni_on_gates \
    --train_fpath data/icg_channels/icg-channels-K/279_hh2.json -v
```

### Fit a biophysics-only (single-compartment soma) model

```bash
cd scripts
python train_biophysics.py --config train_soma_on_multicomp -v
```

All training scripts accept `--config <ConfigName>` to select the experiment configuration defined in `scripts/configs/`. Run with `-v` for verbose console output.

## Model Overview

The `hybrid_models` library provides:

- **`HH`**: A modular Hodgkin-Huxley neuron model supporting arbitrary channel combinations and batched/vmapped integration via `diffrax`.
- **`NODE`**: A neural ODE channel that replaces one or more biophysical channels with a learned vector field, sharing voltage as the observed state.
- **`BioPhysicsNODE`**: A NODE with biophysically-structured gating (power-law conductance × driving force).
- **`Omni`**: A parameterized biophysical channel with flexible gate kinetics, fitted to ion channel databases.
- **`scaled_integrate`**: Differentiable ODE integration with adjoint-based gradients for training.

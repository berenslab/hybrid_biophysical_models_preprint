# Pre-computed Results

Pre-computed experiment results (trained model weights and sweep outputs) are hosted on Zenodo. Downloading them is required to reproduce the figures without rerunning training.

## Download

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20699289.svg)](https://doi.org/10.5281/zenodo.20699289)

```bash
# Download and extract to the project root
wget https://zenodo.org/records/20699289/files/results.zip
unzip results.zip
```

The archive must be extracted so that the `results/` directory sits at the project root (next to `notebooks/`, `scripts/`, etc.):

```
hybrid_models/
├── results/            ← extract here
│   ├── sweeps/
│   ├── runtimes_hh_k.csv
│   ├── runtimes_multicomp.csv
│   └── train_soma_on_multicomp_*.{json,eqx}
├── notebooks/
├── scripts/
└── ...
```

## Contents

### `results/sweeps/`

Each sweep directory contains one `.json` (config + metrics), `.eqx` (model weights), and `.log` (training log) file per run.

| Directory | Used in | Description |
|---|---|---|
| `icg11/` | Figure 2 | ICG ion channel fitting sweep (NODE and Omni on Na, K, Ca, IH, KCa channels) |
| `hh_channel_obs_noise_sweep4/` | Figure 3, Table | K-channel fitting at varying observation noise levels |
| `hh_channel_init_noise_sweep5/` | Table | K-channel fitting at varying parameter initialisation noise |
| `hh_channel_na_arch_sweep/` | Appendix Figure | Na-channel fitting across network architectures and activation functions |
| `hh_channel_arch_sweep3/` | Table | K-channel fitting across network architectures |
| `multicomp_sweep2/` | Figure 4, Table | NODE fitting on multicompartment neuron across architectures |

### `results/runtimes_hh_k.csv` / `results/runtimes_multicomp.csv`

Pre-benchmarked integration runtimes used in the runtime comparison tables. Can be regenerated with `scripts/evaluate_runtimes.py`.

### `results/train_soma_on_multicomp_*.{json,eqx}`

Two fitted single-compartment soma baseline models referenced directly in Figure 4 (notebook `31_figure4.ipynb`).

## Rerunning experiments

To rerun a sweep instead of using pre-computed results, use the training scripts in `scripts/`. For example, to retrain the K-channel observation noise sweep:

```bash
cd scripts
for seed in 0 1 2 3 4; do
    for noise in 0.0 0.01 0.05 0.1 0.2; do
        python train_channel.py --config train_hybrid_on_hh_k \
            --output_dir results/sweeps/hh_channel_obs_noise_sweep4 \
            --obs_noise $noise --seed $seed
    done
done
```

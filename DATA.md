# Data

The training data used in this work is hosted on Zenodo and must be downloaded separately.

## Download

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20699289.svg)](https://doi.org/10.5281/zenodo.20699289)

```bash
# Download and extract to the project root
wget https://zenodo.org/records/20699289/files/data.zip
unzip data.zip
```

The archive must be extracted so that the `data/` directory sits at the project root (next to `notebooks/`, `scripts/`, etc.):

```
hybrid_models/
├── data/               ← extract here
│   ├── hh_synth/
│   └── icg_channels/
├── notebooks/
├── scripts/
└── ...
```

## Contents

### `data/hh_synth/`

Synthetic voltage traces generated from the Hodgkin-Huxley model via `scripts/simulate_voltage.py`. Used as training and validation data for the HH channel-fitting experiments (Figures 3, 4).

| File | Description |
|---|---|
| `hh_multi_ap_v_only_{train,val}.json` | Single-stimulus multi-spike trace (K- and Na-channel experiments) |
| `hh_multi_ap_batch_v_only_{train,val}.json` | Batch of 21 traces at varying stimulus amplitudes |
| `pospischil_mutlicomp_single_pt_soma.json` | Multicompartment soma voltage trace (Figure 4) |

### `data/icg_channels/`

Ion channel kinetics data (steady-state activation/inactivation curves and time constants) from the [ICG ion channel database](https://icg.neurotheory.ox.ac.uk/). Used for the ICG channel-fitting experiments (Figure 2).

Subdirectories: `icg-channels-Na/`, `icg-channels-K/`, `icg-channels-Ca/`, `icg-channels-IH/`, `icg-channels-KCa/`, `icg_pickles/`.

## Regenerating the data

The synthetic HH data can be regenerated from scratch:

```bash
cd scripts
python simulate_voltage.py --config generate_hh_multi_spike
python simulate_voltage.py --config generate_hh_multi_spike_batch
```

ICG data must be downloaded from the ICG database; see `other/load_icg_data/` for loading scripts.

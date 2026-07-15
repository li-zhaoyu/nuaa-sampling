# NUAA: Non-Uniform Adaptive Acquisition

Code and paper artifacts for the MDPI *Photonics* manuscript

**Non-Uniform Adaptive Acquisition for Microwave Photonic Analog-to-Digital Conversion**  
Manuscript ID: `photonics-4466249`

This repository contains the algorithms, experiment entry points, pretrained checkpoints, and numerical/waveform data that appear in the manuscript (Sections 3–4).

## Layout

```
nuaa/              Core library (layout, forward operator A(τ), SOMP/FISTA, streaming)
models/            NUAA-MU + GRU evolution prior
experiments/       §4 entry points + figure scripts (+ required helpers)
tests/             Mapping / protocol regression tests
outputs/           Raw JSON metrics + checkpoints used in the paper
data/metrics/      Extracted arrays for Figs. 3, 5, 6 and Table 2
data/waveforms/    All §4 signal waveforms (chirplet / radar / THz)
figures/           Publication figures (Fig. 1–7)
docs/              Data-availability notes
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repository root.

## Reproduce Section 4

### §4.1 Broadband chirplet (Fig. 3–4, Table 2)

```bash
bash experiments/run_prior_accumulation_200pt.sh   # Fig. 3
python experiments/plot_wideband_waveform_best.py  # Fig. 4 (+ saves data/waveforms/fig4_*.npz)
```

Table 2 numbers are in `outputs/streaming_wideband_nuaa_table2_fig3slice_*.json`  
and summarized in `data/metrics/table2_broadband_chirplet.csv`.

### §4.2 Quasi-periodic radar (Fig. 5)

```bash
python experiments/exp_streaming_radar_complex_nuaa_mu.py --duration-s 2 --trials 8
python experiments/plot_radar_quasiperiodic_nmse.py
```

### §4.3 THz 16-QAM EVM (Fig. 6–7)

```bash
python experiments/exp_streaming_thz_nuaa_mu.py --tag table4_evm0to22 --plot \
  --model-path outputs/thz_nuaa_mu_pretrained_large_m200.pt \
  --window-periods 40 --ticks 20 --control-dt-ms 100 --trials 8 --K 7 \
  --cap large --evm-list 0 2 4 6 8 10 12 14 15 16 17 18 19 20 21 22 \
  --plot-evm 15 --multipath-gains 0
```

## Data map

| Manuscript              | Files                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| Fig. 3                  | `data/metrics/fig3_*.npz`, `outputs/*nosp_accum.json`, `*withprior_cal20*.json`          |
| Table 2                 | `data/metrics/table2_broadband_chirplet.csv`, `outputs/*table2_fig3slice_*.json`         |
| Fig. 4 / §4.1 waveforms | `data/waveforms/chirplet_broadband_waveforms.npz`                                        |
| Fig. 5 / §4.2 waveforms | `data/waveforms/radar_quasiperiodic_waveforms.npz` (+ NMSE in `data/metrics/fig5_*.npz`) |
| Fig. 6                  | `data/metrics/fig6_thz_evm_curve.npz`, `*table4_evm0to22_med.json`                       |
| Fig. 7 / §4.3 waveforms | `data/waveforms/thz_16qam_waveforms.npz`                                                 |

Regenerate all waveform packs: `python experiments/export_paper_waveforms.py`

See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) for details.

# 

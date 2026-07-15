# Data Availability

Companion data for MDPI *Photonics* manuscript **photonics-4466249**.

## Algorithms (code)

| Component                                    | Location                                                   |
| -------------------------------------------- | ---------------------------------------------------------- |
| Cascaded-EDL layout / multi-coset operator   | `nuaa/layout.py`, `nuaa/measurement.py`, `nuaa/forward.py` |
| SOMP / FISTA / interference nulling          | `nuaa/reconstruct.py`                                      |
| Streaming belief / EDL control               | `nuaa/streaming.py`, `nuaa/policy.py`                      |
| Signal families (chirplet, radar codes, THz) | `nuaa/signals.py`, `nuaa/signals_thz_comm.py`              |
| NUAA-MU (Bi-Mamba unfolding)                 | `models/nuaa_mu.py`                                        |
| GRU evolution prior                          | `models/evolution_prior.py`                                |

## Experiment entry points

| Section                 | Script                                               |
| ----------------------- | ---------------------------------------------------- |
| §4.1 broadband chirplet | `experiments/exp_streaming_wideband_nuaa.py`         |
| §4.1 Fig. 3 driver      | `experiments/run_prior_accumulation_200pt.sh`        |
| §4.1 Fig. 4 plot        | `experiments/plot_wideband_waveform_best.py`         |
| §4.2 radar tracking     | `experiments/exp_streaming_radar_complex_nuaa_mu.py` |
| §4.3 THz EVM            | `experiments/exp_streaming_thz_nuaa_mu.py`           |

Helper modules kept solely as imports: `exp_structured_nuaa_mu.py`, `exp_e4_evolution.py`, `exp_e5_radar_complex.py`, `exp_streaming_radar_complex.py`, `train_nuaa_mu.py`.

## Numerical results

Raw run dumps: `outputs/`.

Extracted arrays for plotting / reuse:

| File                                                | Content                                           |
| --------------------------------------------------- | ------------------------------------------------- |
| `data/metrics/fig3_prior_accumulation_nmse.npz`     | median NMSE / F1 vs time                          |
| `data/metrics/fig3_prior_accumulation_tick_iqr.npz` | median + IQR tick curves                          |
| `data/metrics/table2_broadband_chirplet.csv`        | Table 2 summary                                   |
| `data/metrics/fig5_radar_quasiperiodic_nmse.npz`    | NLFM / polyphase-LFM / Costas / Frank NMSE curves |
| `data/metrics/fig6_thz_evm_curve.npz`               | residual EVM vs injected EVM                      |
| `data/waveforms/chirplet_broadband_waveforms.npz`   | §4.1 chirplet: true / recon / jammer / observed   |
| `data/waveforms/radar_quasiperiodic_waveforms.npz`  | §4.2 NLFM, polyphase-LFM, Costas, Frank pulses    |
| `data/waveforms/thz_16qam_waveforms.npz`            | §4.3 clean / impaired / reconstructed symbols     |

# 

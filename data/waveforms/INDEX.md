# Paper signal waveform packs

All manuscript signal families (§4.1–§4.3):

| File | Content |
|---|---|
| `chirplet_broadband_waveforms.npz` | Broadband chirplet: full-pulse complex `x_true_useful`, `x_reconstructed`, `x_jammer`, `x_observed_jam_noise`, plus Fig. 4 zoom amplitudes |
| `fig4_wideband_chirp_waveforms.npz` | Fig. 4 convenience alias (zoomed amplitudes only) |
| `radar_quasiperiodic_waveforms.npz` | NLFM / polyphase-LFM (`phase_lfm`) / Costas / Frank: `*_x_true`, `*_x_reconstructed`, `*_y_measurement` |
| `thz_16qam_waveforms.npz` | THz 16-QAM: `sym_clean`, `sym_impaired`, `sym_reconstructed`, `sym_format_projected` |
| `fig7_thz_constellation_evm15.npz` | Fig. 7 convenience alias |

Regenerate:

```bash
python experiments/export_paper_waveforms.py
```

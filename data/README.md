# Extracted paper data

## `metrics/`

NMSE / EVM / Table 2 arrays derived from `outputs/*.json`.

## `waveforms/`

Ground-truth and reconstructed waveforms for **every signal family in the manuscript**:

| Signal (§) | File | Arrays |
|---|---|---|
| Broadband chirplet (§4.1) | `chirplet_broadband_waveforms.npz` | `x_true_useful`, `x_reconstructed`, `x_jammer`, `x_observed_jam_noise` (+ Fig.4 zoom amps) |
| NLFM / polyphase-LFM / Costas / Frank (§4.2) | `radar_quasiperiodic_waveforms.npz` | `{family}_x_true`, `{family}_x_reconstructed`, `{family}_y_measurement` |
| THz 16-QAM (§4.3) | `thz_16qam_waveforms.npz` | `sym_clean`, `sym_impaired`, `sym_reconstructed`, `sym_format_projected` |

```python
import numpy as np

chirp = np.load("data/waveforms/chirplet_broadband_waveforms.npz")
t, xt, xh = chirp["t_ns"], chirp["x_true_useful"], chirp["x_reconstructed"]

radar = np.load("data/waveforms/radar_quasiperiodic_waveforms.npz")
xt = radar["nlfm_x_true"]; xh = radar["nlfm_x_reconstructed"]

thz = np.load("data/waveforms/thz_16qam_waveforms.npz")
clean, imp, rec = thz["sym_clean"], thz["sym_impaired"], thz["sym_reconstructed"]
```

Regenerate all packs: `python experiments/export_paper_waveforms.py`

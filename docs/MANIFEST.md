# Manifest

## `outputs/`

- prior_accumulation_200pt_metrics.json
- streaming_wideband_nuaa_n5000_wb_tau16_wp40_dt100_t20_nosp_accum.json
- streaming_wideband_nuaa_n5000_wb_tau16_wp40_dt100_t20_withprior_cal20_gain0.1_ema0.3_fwcoef.json
- streaming_wideband_nuaa_table2_fig3slice_baselines.json
- streaming_wideband_nuaa_table2_fig3slice_nosp.json
- streaming_wideband_nuaa_table2_fig3slice_withprior.json
- streaming_wideband_nuaa_n5000_wb_tau16_wp40_withprior_sat.pt
- streaming_radar_complex_nuaa_mu_2s.json
- radar_complex_nuaa_mu_pretrained.pt
- radar_complex_evolution_prior.pt
- streaming_thz_nuaa_mu_table4_evm0to22_med.json
- streaming_thz_nuaa_mu_fig7_evm15_one_error.json
- thz_nuaa_mu_pretrained_large_m200.pt

## `data/metrics/`

- fig3_prior_accumulation_nmse.npz
- fig3_prior_accumulation_tick_iqr.npz
- fig5_radar_quasiperiodic_nmse.npz
- fig6_thz_evm_curve.npz
- table2_broadband_chirplet.csv

## `data/waveforms/`

- chirplet_broadband_waveforms.npz          # §4.1 useful / jammer / recon / observed
- fig4_wideband_chirp_waveforms.npz         # Fig. 4 zoom alias
- radar_quasiperiodic_waveforms.npz         # §4.2 NLFM / phase_lfm / Costas / Frank
- thz_16qam_waveforms.npz                   # §4.3 16-QAM symbols
- fig7_thz_constellation_evm15.npz          # Fig. 7 alias
- INDEX.md

## `figures/`

- hardware_signal_flow.svg/.pdf
- nuaa_algo_closed_loop.svg/.pdf
- prior_accumulation_nmse_200pt.svg
- wideband_waveform_best.svg/.pdf
- radar_quasiperiodic_nmse_2x2.svg/.pdf
- thz_evm_reconstruction_curve.svg/.pdf
- thz_constellation_evm.svg/.pdf

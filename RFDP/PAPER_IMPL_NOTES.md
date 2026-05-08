# Paper-Accurate RFDP Additions

This repository now contains a parallel NumPy-first implementation of the core
methods from:

- Zheng et al., 2021, *Hybrid Regularization of Diffusion Process for Visual Re-Ranking*

The new code lives under `MODEL/Paper_RFDP/` and is intentionally separate from
the older RFDP classes so the previous behavior remains unchanged.

## Main entry points

- `HyRdpPaper`
- `GmfptPaper`
- `IHyRdpPaper`
- `IGmfptPaper`

## Scope of this implementation

- Paper-style `A/B` split and local problem construction
- HyRDP closed-form and iterative solvers
- GMFPT closed-form solver
- Iterative re-ranking with temporary connections on disconnected graphs
- Unit tests and a synthetic demo without external `.mat` files

## Deliberate limitations

- NumPy is the reference backend for the paper-accurate path
- Existing `HyDiffFC` / `RfdpCtofOtcm` code paths are preserved and not redefined
- Real dataset experiment reproduction is not bundled in this stage

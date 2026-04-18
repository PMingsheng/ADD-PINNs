# ADD-PINNs-Possion-3D

3D Poisson interface problem on `Omega=[0,1]^3` with one sphere:

- `c1=(0.4,0.5,0.5)`, `r=0.1`
- inside: `u=exp(xyz)`
- outside: `u=sin(x+y+z)`
- `beta_inside=100`, `beta_outside=2+cos(x+y)`
- reference setting: `alpha1=alpha2=0`, interface `beta1=beta2=0`

## Files

- `problem_3d.py`: geometry, exact solution, coefficients and source term
- `data_3d.py`: 3D sampling for data/interior points
- `pinn3d_*.py`: single-network 3D PINN baseline
- `apinn3d_*.py`: APINN baseline (shared trunk + soft gate + two experts)
- `main.py`, `config.py`, `model.py`, `loss.py`: ADD-PINNs (partition + two experts) 3D solver
- `level_set_3d.py`: smoothed Heaviside/Dirac used by ADD-PINNs loss
- `utils.py`: seed utility

## Run

```bash
cd ADD-PINNs-Possion-3D
python main.py
```

Baseline:

```bash
cd ADD-PINNs-Possion-3D
python pinn3d_main.py
```

APINN:

```bash
cd ADD-PINNs-Possion-3D
python apinn3d_main.py
```

## Training And Visualization

- `pinn3d_config.py` and `apinn3d_config.py` now reuse the same label sampling grid, interior sampling budget, and RAR settings as `config.py`.
- Default total training budget is aligned across all three solvers:
  - ADD-PINNs: `5000 + 95000 = 100000` epochs
  - PINN: `5000 + 95000 = 100000` epochs
  - APINN: `1000` gate pretrain + `5000 + 94000 = 100000` total epochs
- Parameter counts are kept on the same scale:
  - ADD-PINNs 3D: `85251`
  - PINN 3D: `85004`
  - APINN 3D: `85252`

- Stage-wise learning rates are configured by `TrainBlock` lists in:
  - `config.py -> PIMOE3DTrainConfig.blocks`
  - `pinn3d_config.py -> PINN3DTrainConfig.blocks`
  - `apinn3d_config.py -> APINN3DTrainConfig.blocks`
- Every `snapshot_every` epochs (default `5000`), the code saves periodic visuals (with middle slices and `c1` slices):
  - ADD-PINNs: `u_heatmaps/`, `f_heatmaps/`, `phi_heatmaps/`
- PINN: `u_heatmaps/`, `f_heatmaps/`
- APINN: `u_heatmaps/`, `f_heatmaps/`, `gate_heatmaps/`
- Raw (unweighted) loss terms are written to CSV:
  - ADD-PINNs: `loss_list_global.csv`
  - PINN: `loss_list_global.csv`
  - APINN: `loss_list_global.csv`

For ADD-PINNs (`main.py`), output layout now follows the flower-style organization:

- `viz/`: summary figures (`Loss.png`, slice figures)
- `data_output/`: `final_fields.npz`, `loss_list_global.csv`
- root periodic folders: `u_heatmaps/`, `f_heatmaps/`, `phi_heatmaps/`, `snapshots/`

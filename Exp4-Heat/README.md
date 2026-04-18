# Possion Project

Poisson inverse problem with a partitioned PINN and level-set evolution.

## Files and responsibilities

- `main.py`: main entry point; loads data, trains, and generates plots.
- `config.py`: dataclass configs for data, model, training, and visualization.
- `data.py`: load Poisson temperature field data and sample training points.
- `model.py`: PartitionPINN architecture and material parameterization.
- `pinn_main.py`: single PINN baseline for the Poisson inverse problem.
- `apinn_main.py`: APINN baseline with two experts and a gating network.
- `pinn_model.py`, `apinn_model.py`: baseline model definitions.
- `pinn_loss.py`, `apinn_loss.py`: baseline loss functions.
- `problem.py`: circle geometry helpers and full-field reference loading.
- `pde.py`: Poisson PDE residuals and derivative helpers.
- `loss.py`: data/PDE/interface/eikonal loss composition.
- `level_set.py`: level-set evolution and RAR helpers.
- `train.py`: training loop, logging, and snapshot management.
- `visualization.py`: plotting utilities for loss, phi, and residuals.
- `utils.py`: random seed and small helpers.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib

## Quick start

```bash
python main.py
python pinn_main.py
python apinn_main.py
```

## Inputs and outputs

- Input data file: configured by `DataConfig.ttxt_filename` in `config.py`.
  - The current folder includes `Possion_6-10-cross.txt`; update `ttxt_filename` if you want to use it.
- Outputs:
  - `loss_hist_global.txt`
  - `phi_snapshots/`
  - `viz_scatter/`
  - `Loss.png`, `f.png`, `phi_cross_circle.png`

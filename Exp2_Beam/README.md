# LS-PINN Beam Project

Physics-Informed Neural Network with Level-Set method for Euler-Bernoulli beam problems with material discontinuities.

## Files and responsibilities

- `main.py`: training entry point; runs the staged schedule and saves snapshots.
- `main_pinn.py`: standard single-network PINN baseline for the beam problem.
- `config.py`: hyperparameters, device, data ranges, and loss weights.
- `data.py`: load `Beam.txt` and create training/label grids.
- `model.py`: neural network and partitioned material field definitions.
- `pinn_model.py`: standard PINN model that predicts `u` and `EI`.
- `pinn_loss.py`: loss terms for the standard PINN baseline.
- `pde.py`: Euler-Bernoulli beam residuals and derivative helpers.
- `loss.py`: composite loss terms for data, PDE, and interface constraints.
- `level_set.py`: level-set style update utilities for `phi`.
- `train.py`: training loop, optimizer setup, logging, and snapshot export.
- `visualization.py`: plotting helpers for sampling points and beam panels.
- `plot_panels_from_npz.py`: plot `phi_*.npz` panels without retraining.
- `plot_loss_from_csv.py`: plot loss curves from `loss_list_global.csv`.
- `utils.py`: seeding and small helpers.
- `__init__.py`: package marker.

## Usage

```bash
cd ls_pinn_beam
python main.py
```

## Requirements

- PyTorch
- NumPy
- SciPy
- Matplotlib

## Configuration

Edit `config.py` to modify hyperparameters:
- Network architecture (width, depth)
- Training parameters (learning rate, epochs)
- Loss weights
- Data sampling parameters

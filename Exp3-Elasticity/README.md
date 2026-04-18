# Ellipse Project

Physics-informed neural network with a partitioned material field on an ellipse dataset.

## Files and responsibilities

- `main.py`: loads ellipse data, trains the model, and saves plots.
- `config.py`: data sampling, model size, and training hyperparameters.
- `data.py`: loads `Ellipse.txt` and prepares training/fit samples.
- `model.py`: partitioned PINN model and material parameters.
- `pde.py`: PDE residuals and derivative utilities.
- `loss.py`: loss terms for data fit, PDE, and interface penalties.
- `level_set.py`: level-set style updates for the partition field.
- `train.py`: training loop, optimizers, and logging.
- `visualization.py`: plotting utilities for sampling, phi, and residuals.
- `utils.py`: seeding and small helpers.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib

## Quick start

```bash
python main.py
```

## Inputs and outputs

- Input data: `Ellipse.txt` (path configured in `config.py`).
- Outputs:
  - `viz/sampling_points.png`
  - `viz/phi_compare.png`
  - `viz/scatter_final.png`

## Configuration

- `config.py`: data sampling, model size, training hyperparameters, and ellipse geometry.

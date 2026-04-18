# ADD-PINNs-Possion-flower

Poisson interface inverse problem on `[-1, 1]^2` with a circular interface (flower petal count n=0).

## Problem setup

- Interface (polar): `r(theta) = 0.4 + 0.1 sin(0 theta) = 0.4`.
- Level-set: `phi(x,y) = tanh(x^2 + y^2 - r(theta)^2)`, `theta = atan2(x, y)`.
- Piecewise exact solution:
  - outside (`phi >= 0`):
    `u_out = sqrt(0.1 (x^2 + y^2)^2 - 0.01 log(2 sqrt(x^2 + y^2)))`
  - inside (`phi < 0`):
    `u_in = exp(x^2 + y^2)`
- `f, w, v, g` are analytically constructed from the exact solution in `problem.py`.

## Key files

- `problem.py`: flower geometry, exact solution, PDE source terms, jump terms.
- `main.py`: training entry for solving and interface inversion.
- `loss.py`: data + PDE + boundary + interface-jump + eikonal loss.
- `visualization.py`: scatter/heatmap and flower interface comparison plots.

## Run

```bash
cd ADD-PINNs-Possion-flower
python main.py
```

## Single PINN (one network)

Added a plain PINN pipeline that uses:

- one neural network for `u(x,y)`
- hard Dirichlet BC by lifting:
  `u(x,y) = g(x,y) + (x^2-1)(y^2-1)N(x,y)`
- loss terms: `data + PDE` only

Run:

```bash
cd ADD-PINNs-Possion-flower
python pinn_main.py
```

Main files:

- `pinn_config.py`: config for data/model/train/output
- `pinn_model.py`: single-network PINN with hard BC
- `pinn_loss.py`: `loss_data + loss_pde`
- `pinn_main.py`: training/evaluation entry

## APINN (soft-gated experts)

Added an APINN-style pipeline with:

- 2 experts + shared network + soft gate
- gate initialized by `g1(x) ~= sigmoid(-phi0(x)/tau)`, `g2 = 1 - g1`
- PDE enforced on final mixed output:
  `u(x) = g1(x)u1(x) + g2(x)u2(x)`

Run:

```bash
cd ADD-PINNs-Possion-flower
python apinn_main.py
```

Main files:

- `apinn_config.py`: APINN config
- `apinn_model.py`: APINN model (shared + experts + gate)
- `apinn_loss.py`: APINN loss (`data + PDE`)
- `apinn_main.py`: APINN training/evaluation entry

Output includes `u/f` true-pred-residual snapshots and gate visualizations (`gate_g1`, `gate_g2`, `gate_diff`, `gate_partition`).

## Notes

- By default, labels are generated synthetically from the analytic solution (`DataConfig.use_synthetic=True`).
- Output is written to `outputs_flower/<sampling_mode>/run_YYYYMMDD_HHMMSS/` (auto non-overwrite).

## 3D Code Location

All 3D-related code and outputs have been moved to sibling directory:

- `../ADD-PINNs-Possion-3D`

"""
Main entry point for LS-PINN Beam training.
"""
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path for data file access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, EPOCHS, LOSS_WEIGHTS,
    NX_GRID, DENSE_FACTOR, LABEL_COLUMN, DATA_RANGES,
    N_COLLOCATION, CORNER_TOL, EI_REAL,
)
from utils import set_seed
from data import load_uniform_grid_fit, sample_xy_no_corners
from model import PartitionPINN
from train import train_main
from visualization import plot_sampling_points, save_beam_phi_snapshot


def main():
    """Main training entry point."""
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    print(f"Using device: {DEVICE}")
    
    # Set random seed for reproducibility
    set_seed()

    # Initialize model
    model = PartitionPINN().to(DEVICE)
    print("Model initialized")

    # Load training data
    data_file = project_root / "Beam.txt"
    data_file_str = str(data_file)
    xy_fit, label_fit = load_uniform_grid_fit(
        nx=NX_GRID,
        filename=data_file_str,
        device=DEVICE,
        ranges=DATA_RANGES,
        dense_factor=DENSE_FACTOR,
        label_column=LABEL_COLUMN,
    )
    _, label_fit_disp = load_uniform_grid_fit(
        nx=NX_GRID,
        filename=data_file_str,
        device=DEVICE,
        ranges=DATA_RANGES,
        dense_factor=DENSE_FACTOR,
        label_column=2,
    )

    # Plot sampling points
    plot_sampling_points(
        xy_fit,
        label_fit,
        ranges=DATA_RANGES,
        title="Uniform grid + dense defect",
        show=False,
    )

    # Generate collocation points
    xy_int_const = sample_xy_no_corners(N_COLLOCATION, device=DEVICE, corner_tol=CORNER_TOL)
    print(f"Generated {len(xy_int_const)} collocation points")

    # Match notebook training sequence (cells executed in order)
    base_weights = LOSS_WEIGHTS.copy()
    high_weights = LOSS_WEIGHTS.copy()
    high_weights['dEI'] = 1e-6
    # high_weights['interface'] = 1e5

    data_output = project_root / "data_output"
    data_output.mkdir(parents=True, exist_ok=True)
    loss_csv_path = project_root / "loss_list_global.csv"

    epoch_offset = 0
    history = None
    xy_train = xy_int_const
    panels_live = {}

    def run_train(epochs: int, lr: float, lam: dict, save_snapshot: bool) -> None:
        nonlocal model, xy_train, epoch_offset, history

        print(f"\n=== Training: epochs={epochs}, lr={lr} ===")
        print(
            "Weights: "
            f"data_strain={lam.get('data_strain', lam.get('data', 0))}, "
            f"data_disp={lam.get('data_disp', lam.get('data', 0))}, "
            f"dEI={lam.get('dEI', 0)}, "
            f"interface={lam.get('interface', 0)}"
        )

        model, _, _, xy_train, epoch_offset, history = train_main(
            model,
            epochs=epochs,
            lr=lr,
            xy_fit=xy_fit,
            label_fit=label_fit,
            label_fit_disp=label_fit_disp,
            xy_int_const=xy_train,
            opt=None,
            opt_phi=None,
            lam=lam,
            loss_save_path=str(loss_csv_path),
            epoch_offset_global=epoch_offset,
            history=history,
            live_phi=False,
            live_panels=False,
            panels_live=panels_live,
            panels_label_path=data_file_str,
            panels_out_dir=str(project_root / "beam_bechmark_viz"),
            panels_EI_real=EI_REAL,
            save_snapshots=save_snapshot,
            snapshot_plot_panels=True,
        )

        if save_snapshot:
            save_beam_phi_snapshot(
                model,
                label_path=data_file_str,
                out_dir=str(project_root / "beam_bechmark_viz"),
                plot_panels=True,
                suffix=epoch_offset,
                EI_real=EI_REAL,
            )

    # 1) 5000 epochs @ 1e-3, base weights
    run_train(epochs=EPOCHS, lr=1e-4, lam=base_weights, save_snapshot=True)

    # 2) 5000 epochs @ 1e-5, base weights
    run_train(epochs=EPOCHS, lr=1e-4, lam=base_weights, save_snapshot=True)

    # # 3) Repeat 4x: 5000 epochs @ 1e-5, high weights (save each time)
    for _ in range(4):
        run_train(epochs=EPOCHS, lr=1e-5, lam=high_weights, save_snapshot=True)

    loss_src = project_root / "loss_list_global.csv"
    loss_dst = data_output / "loss_list_global.csv"
    if loss_src.exists():
        shutil.copy2(loss_src, loss_dst)
    elif history and history.get("loss_list_global_item"):
        header = "epoch,total,data,data_strain,data_disp,fai,dfai,weight,M,V,Q,dEI,interface,eik,area"
        np.savetxt(
            loss_dst,
            np.asarray(history["loss_list_global_item"], dtype=float),
            delimiter=",",
            header=header,
            comments="",
        )

    save_beam_phi_snapshot(
        model,
        label_path=data_file_str,
        out_dir=str(data_output),
        plot_panels=False,
        suffix="final",
        EI_real=EI_REAL,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    main()

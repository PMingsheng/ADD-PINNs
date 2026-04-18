import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def relative_l2_error(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    true_flat = np.asarray(true_vals, dtype=np.float64).reshape(-1)
    pred_flat = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(pred_flat - true_flat, ord=2) / (np.linalg.norm(true_flat, ord=2) + 1e-15))


def compute_rel_l2_rows(
    model_npz_paths: Dict[str, Path],
    quantity_pairs: List[Tuple[str, str, str]],
) -> List[Dict[str, str]]:
    loaded = {}
    for model_name, npz_path in model_npz_paths.items():
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ not found for {model_name}: {npz_path}")
        loaded[model_name] = np.load(npz_path)

    rows: List[Dict[str, str]] = []
    for quantity_name, pred_key, true_key in quantity_pairs:
        row: Dict[str, str] = {"Quantity": quantity_name}
        for model_name, data in loaded.items():
            if pred_key not in data or true_key not in data:
                row[model_name] = "NA"
                continue
            pred = np.asarray(data[pred_key]).reshape(-1)
            true = np.asarray(data[true_key]).reshape(-1)
            if pred.size == 0 or true.size == 0 or pred.shape != true.shape:
                row[model_name] = "NA"
                continue
            row[model_name] = f"{relative_l2_error(true, pred):.8e}"
        rows.append(row)
    return rows


def write_csv(out_csv: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_output = base_dir.parent / "Figure" / "Beam_Physical_Quantities_Relative_L2_Error.csv"

    parser = argparse.ArgumentParser(
        description="Export relative L2 errors of beam physical quantities to CSV."
    )
    parser.add_argument(
        "--pimoe-npz",
        type=str,
        default=str(base_dir / "data_output" / "phi_final.npz"),
        help="Path to ADD-PINNs npz (default: data_output/phi_final.npz).",
    )
    parser.add_argument(
        "--pinn-npz",
        type=str,
        default=str(base_dir / "data_output_reduced" / "phi_final.npz"),
        help="Path to PINN npz (default: data_output_reduced/phi_final.npz).",
    )
    parser.add_argument(
        "--apinn-npz",
        type=str,
        default=str(base_dir / "data_output_apinn" / "phi_final.npz"),
        help="Path to APINN npz (default: data_output_apinn/phi_final.npz).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(default_output),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    model_npz_paths = {
        "ADD-PINNs": Path(args.pimoe_npz).expanduser().resolve(),
        "PINN": Path(args.pinn_npz).expanduser().resolve(),
        "APINN": Path(args.apinn_npz).expanduser().resolve(),
    }

    quantity_pairs = [
        ("Displacement_u", "u_NN", "u_lab"),
        ("Rotation_theta", "theta_NN", "theta_lab"),
        ("Curvature_kappa", "kappa_NN", "kappa_lab"),
        ("Bending_Moment_M", "M_NN", "M_lab"),
        ("Shear_Force_V", "V_NN", "V_lab"),
        ("Stiffness_EI", "EI_pred", "EI_true"),
    ]

    fieldnames = ["Quantity", "ADD-PINNs", "PINN", "APINN"]
    rows = compute_rel_l2_rows(model_npz_paths, quantity_pairs)
    out_csv = Path(args.out).expanduser().resolve()
    write_csv(out_csv, fieldnames, rows)
    print(f"[Done] CSV exported to: {out_csv}")


if __name__ == "__main__":
    main()

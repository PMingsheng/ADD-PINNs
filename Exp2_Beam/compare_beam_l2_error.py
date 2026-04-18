import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def relative_l2_error(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    diff = (pred_vals - true_vals).reshape(-1)
    true_flat = true_vals.reshape(-1)
    true_l2 = float(np.linalg.norm(true_flat, ord=2))
    return float(np.linalg.norm(diff, ord=2) / (true_l2 + 1e-15))


def _flat(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def compute_quantity_errors(
    data: np.lib.npyio.NpzFile,
    pairs: List[Tuple[str, str, str]],
) -> Dict[str, float]:
    errors: Dict[str, float] = {}
    for quantity, pred_key, true_key in pairs:
        if pred_key not in data or true_key not in data:
            continue

        pred = _flat(data[pred_key])
        true = _flat(data[true_key])

        if pred.size == 0 or true.size == 0:
            continue
        if pred.shape != true.shape:
            raise ValueError(
                f"Shape mismatch for {quantity}: {pred_key}={pred.shape}, {true_key}={true.shape}"
            )

        errors[quantity] = relative_l2_error(true, pred)
    return errors


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export relative L2 errors of ADD-PINNs, PINN, reduced PINN, and APINN to CSV."
    )
    parser.add_argument(
        "--pi-moe-npz",
        type=str,
        default=str(project_root / "data_output" / "phi_final.npz"),
        help="Path to ADD-PINNs npz file.",
    )
    parser.add_argument(
        "--pinn-npz",
        type=str,
        default=str(project_root / "data_output_pinn" / "phi_final.npz"),
        help="Path to standard PINN npz file.",
    )
    parser.add_argument(
        "--reduced-pinn-npz",
        type=str,
        default=str(project_root / "data_output_reduced" / "phi_final.npz"),
        help="Path to reduced PINN npz file.",
    )
    parser.add_argument(
        "--apinn-npz",
        type=str,
        default=str(project_root / "data_output_apinn" / "phi_final.npz"),
        help="Path to APINN npz file.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(project_root / "beam_relative_l2_errors.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    model_paths = {
        "ADD-PINNs": Path(args.pi_moe_npz).expanduser().resolve(),
        "PINN": Path(args.pinn_npz).expanduser().resolve(),
        "Reduced PINN": Path(args.reduced_pinn_npz).expanduser().resolve(),
        "APINN": Path(args.apinn_npz).expanduser().resolve(),
    }
    for model_name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{model_name} NPZ not found: {path}")

    quantity_pairs = [
        (r"u", "u_NN", "u_lab"),
        (r"theta = u'", "theta_NN", "theta_lab"),
        (r"kappa = u''", "kappa_NN", "kappa_lab"),
        (r"M = EIu''", "M_NN", "M_lab"),
        (r"V = (EIu'')'", "V_NN", "V_lab"),
        (r"EI", "EI_pred", "EI_true"),
    ]

    model_errors: Dict[str, Dict[str, float]] = {}
    for model_name, path in model_paths.items():
        with np.load(path) as data:
            model_errors[model_name] = compute_quantity_errors(data, quantity_pairs)

    rows: List[List[str]] = []
    header = ["Quantity", "ADD-PINNs", "PINN", "Reduced PINN", "APINN"]
    for quantity, _, _ in quantity_pairs:
        rows.append([
            quantity,
            f"{model_errors['ADD-PINNs'].get(quantity, np.nan):.8e}",
            f"{model_errors['PINN'].get(quantity, np.nan):.8e}",
            f"{model_errors['Reduced PINN'].get(quantity, np.nan):.8e}",
            f"{model_errors['APINN'].get(quantity, np.nan):.8e}",
        ])

    csv_path = Path(args.csv).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("=== Beam Relative L2 Error Summary ===")
    print(f"ADD-PINNs : {model_paths['ADD-PINNs']}")
    print(f"PINN   : {model_paths['PINN']}  (standard PINN)")
    print(f"Reduced: {model_paths['Reduced PINN']}")
    print(f"APINN  : {model_paths['APINN']}")
    print(f"CSV    : {csv_path}")
    print("")
    print(f"{'Quantity':<18} {'ADD-PINNs':>14} {'PINN':>14} {'Reduced':>14} {'APINN':>14}")
    for quantity, pi_moe, pinn, reduced_pinn, apinn in rows:
        print(f"{quantity:<18} {pi_moe:>14} {pinn:>14} {reduced_pinn:>14} {apinn:>14}")


if __name__ == "__main__":
    main()

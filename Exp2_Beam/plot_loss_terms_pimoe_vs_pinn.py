import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


# Nature-like style (aligned with existing figure scripts)
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["mathtext.fontset"] = "stixsans"
rcParams["font.size"] = 7
rcParams["axes.labelsize"] = 7
rcParams["axes.titlesize"] = 7
rcParams["xtick.labelsize"] = 6
rcParams["ytick.labelsize"] = 6
rcParams["legend.fontsize"] = 6
rcParams["axes.linewidth"] = 0.6
rcParams["grid.linewidth"] = 0.5
rcParams["lines.linewidth"] = 1.2


C_PIMOE = "#D55E00"
C_PINN = "#0072B2"

FIG_WIDTH = 7.08
FIG_HEIGHT = 3.8

TERMS = [
    ("data", "Data Loss"),
    ("fai", "Rotation Residual ($R_{\\theta}$)"),
    ("dfai", "Curvature Residual ($R_{\\kappa}$)"),
    ("M", "Moment Residual ($R_M$)"),
    ("V", "Shear Residual ($R_V$)"),
    ("Q", "Load Residual ($R_Q$)"),
]

PANEL_TAGS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "epoch" not in df.columns:
        raise ValueError(f"'epoch' column missing in {path}")
    return df


def get_numeric_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def plot_terms(
    pimoe_csv: Path,
    pinn_csv: Path,
    output_pdf: Path,
    show: bool = False,
) -> None:
    df_pimoe = load_csv(pimoe_csv)
    df_pinn = load_csv(pinn_csv)

    ep_pimoe = pd.to_numeric(df_pimoe["epoch"], errors="coerce").to_numpy(dtype=float)
    ep_pinn = pd.to_numeric(df_pinn["epoch"], errors="coerce").to_numpy(dtype=float)

    fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
    axs = axs.ravel()

    for ax, (term, title), tag in zip(axs, TERMS, PANEL_TAGS):
        y_pimoe = get_numeric_series(df_pimoe, term)
        y_pinn = get_numeric_series(df_pinn, term)

        mask_pimoe = np.isfinite(ep_pimoe) & np.isfinite(y_pimoe) & (y_pimoe > 0)
        mask_pinn = np.isfinite(ep_pinn) & np.isfinite(y_pinn) & (y_pinn > 0)

        if np.any(mask_pimoe):
            ax.semilogy(
                ep_pimoe[mask_pimoe],
                y_pimoe[mask_pimoe],
                color=C_PIMOE,
                linestyle="-",
                linewidth=1.3,
                label="ADD-PINNs",
            )
        if np.any(mask_pinn):
            ax.semilogy(
                ep_pinn[mask_pinn],
                y_pinn[mask_pinn],
                color=C_PINN,
                linestyle="--",
                linewidth=1.3,
                label="PINN",
            )

        ax.text(-0.14, 1.04, tag, transform=ax.transAxes, fontsize=8, fontweight="bold")
        ax.set_title(title, pad=4)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.55)

        x_max = np.nanmax([np.nanmax(ep_pimoe), np.nanmax(ep_pinn)])
        ax.set_xlim(left=0, right=x_max)

        if term == "data":
            ax.legend(frameon=False, loc="upper right")

        if not np.any(mask_pimoe) and not np.any(mask_pinn):
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes, color="#666666")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")

    print(f"[Info] Saving PDF to: {output_pdf}")
    plt.savefig(output_pdf, dpi=300, format="pdf")
    print(f"[Info] Saving PNG to: {output_png}")
    plt.savefig(output_png, dpi=300, format="png")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_pimoe = base_dir / "data_output" / "loss_list_global.csv"
    default_pinn = base_dir / "data_output_reduced" / "loss_list_global.csv"
    default_out = base_dir.parent / "Figure" / "LossTerms_ADD-PINNs_vs_PINN.pdf"

    parser = argparse.ArgumentParser(
        description="Plot per-term loss histories (data/fai/dfai/M/V/Q) for ADD-PINNs vs PINN."
    )
    parser.add_argument("--pimoe-csv", type=str, default=str(default_pimoe), help="ADD-PINNs loss csv path")
    parser.add_argument("--pinn-csv", type=str, default=str(default_pinn), help="PINN loss csv path")
    parser.add_argument("--out", type=str, default=str(default_out), help="Output PDF path")
    parser.add_argument("--show", action="store_true", help="Show figure interactively")
    args = parser.parse_args()

    plot_terms(
        pimoe_csv=Path(args.pimoe_csv).expanduser().resolve(),
        pinn_csv=Path(args.pinn_csv).expanduser().resolve(),
        output_pdf=Path(args.out).expanduser().resolve(),
        show=args.show,
    )


if __name__ == "__main__":
    main()

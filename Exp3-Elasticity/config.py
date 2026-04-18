from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEED = 1234

ELLIPSE_PARAMS = (0.05, 0.10, 0.35, 0.15, np.deg2rad(-30.0))

DATA_FILE = str(DATA_ROOT / "Ellipse.txt")

DATA_CONFIG = {
    "txt_filename": DATA_FILE,
    "nx": 7,
    "ny": 7,
    "ellipse": ELLIPSE_PARAMS,
    "tau_for_strain": 0,
    "use_dense": True,
    "dense_factor": 0.21,
    "rect_params": ((0.07, 0.09), 1.0, 0.4, -30.0),
    "target_total": 100,
    "random_state": 123,
}

ROI_OFF_CONFIG = {
    "nx": 10,
    "ny": 10,
    "txt_filename": DATA_FILE,
    "tau_for_strain": 0.001,
    "use_dense": False,
    "dense_factor": 1.0,
    "rect_corners": None,
    "xlim": (-1.0, 1.0),
    "ylim": (-1.0, 1.0),
}

# ROI_RECT_CONFIG = {
#     "nx": 7,
#     "ny": 7,
#     "txt_filename": DATA_FILE,
#     "tau_for_strain": 0.001,
#     "dense_factor": 0.7,
#     "rect_corners": ((-0.5, -0.2), (0.7, 0.4)),
#     "target_total": 100,
#     "random_state": 2,
# }
ROI_RECT_CONFIG = {
    "nx": 7,
    "ny": 7,
    "txt_filename": DATA_FILE,
    "tau_for_strain": 0.001,
    "dense_factor": 0.3,
    "rect_corners": ((-0.5, -0.2), (0.7, 0.4)),
    "target_total": 100,
    "random_state": 2,
}
MODEL_CONFIG = {
    "width": 50,
    "depth": 4,
    "E1_init": 1,
    "E2_init": 1,
    "learn_E1": True,
    "learn_E2": False,
}

E_SCALE = 1.0

NU = 0.30
EPS0 = 3.0e-3

LAM_WEIGHTS = {
    "data": 1e5,
    "pde": 1e3,
    "bc": 1e2,
    "interface": 1e2,
    "eik": 1e-4,
    "area": 1e-4,
}

TRAIN_CONFIG = {
    "epochs": 20000,
    "lr": 1e-3,
    "phi_lr": 1e-4,
    "xy_int_n": 8000,
    "xy_int_xlim": (-1.0, 1.0),
    "xy_int_ylim": (-1.0, 1.0),
    "rar_every": 5000,
    "rar_n_cand": 4096,
    "rar_n_new": 256,
    "log_every": 500,
    "history_every": 50,
    "phi_snapshot_every": 250,
    "phi_snapshot_n": 200,
    "phi_snapshot_bbox": (-1.0, 1.0, -1.0, 1.0),
    "scatter_every": 2500,
    "phi_evolve_every": 100,
    "phi_evolve_stage1": {
        "start": 2000,
        "end": 10000,
        "dt": 1.0,
        "n_inner": 10,
        "band_eps": 0.05,
        "h": 0.05,
        "tau": 1e-3,
        "typeVn": "CV",
    },
    "phi_evolve_stage2": {
        "start": 10000,
        "dt": 1e-3,
        "n_inner": 100,
        "band_eps": 0.01,
        "h": 0.05,
        "tau": 1e-8,
        "typeVn": "PDE",
    },
}

VIZ_SCATTER_CONFIGS = [
    (
        "PDE",
        {
            "dt_next": 1e-3,
            "band_eps_vel": 0.01,
            "h_vel": 0.05,
            "tau_vel": 1e-12,
            "clip_q_vel": 0.99,
        },
    ),
    (
        "GRAD",
        {
            "dt_next": 0.05,
            "band_eps_vel": 0.05,
            "h_vel": 0.05,
            "tau_vel": 1e-12,
            "clip_q_vel": 0.99,
        },
    ),
    (
        "CV",
        {
            "dt_next": 1.0,
            "band_eps_vel": 0.05,
            "h_vel": 0.05,
            "tau_vel": 1e-3,
            "clip_q_vel": 0.99,
        },
    ),
]

PHI_SNAPSHOT_DIR = PROJECT_ROOT / "phi_snapshots"
VIZ_SCATTER_DIR = PROJECT_ROOT / "viz_scatter"
DISP_STRAIN_SNAPSHOT_DIR = PROJECT_ROOT / "disp_strain_snapshots"
SLICE_SNAPSHOT_DIR = PROJECT_ROOT / "u_slice_with_phi"

E_HISTORY_PATH = PROJECT_ROOT / "E_history.txt"
LOSS_HIST_GLOBAL_PATH = PROJECT_ROOT / "loss_hist_global.txt"
LOSS_CSV_PATH = PROJECT_ROOT / "loss_list_global.csv"

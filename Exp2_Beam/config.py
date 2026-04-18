"""
Configuration and hyperparameters for LS-PINN Beam project.
"""
import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Beam geometry parameters
L = 1.0          # Beam length
H = 0.05         # Beam height
B = 0.05         # Beam width
I = (B * H**3) / 12  # Second moment of area

# Material properties
E_BG = 8e9       # Background elastic modulus
E_DEF_1 = 6e9    # Defect region 1 elastic modulus
E_DEF_2 = 10e9   # Defect region 2 elastic modulus
EI_REAL = 3e9 * 0.05 * 0.05**3 / 12  # Reference EI value

# Network architecture
PHI_WIDTH = 20
PHI_DEPTH = 3
VAR_WIDTH = 150
VAR_DEPTH = 5

# Training hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 5000
RAR_ENABLED = False
RAR_INTERVAL = 2500


LOSS_WEIGHTS = {
    'data_strain': 1e6,
    'data_disp': 1e6,
    'fai': 1e2,
    'dfai': 1e2,
    'M': 1,
    'V': 1,
    'Q': 1,
    'weight': 0,
    'interface': 1e-4,
    'eik': 1e-8,
    'area': 1e-8,
    'dEI': 0,
}
# Data sampling parameters
NX_GRID = 100
DENSE_FACTOR = 20
LABEL_COLUMN = 6  # 6 for strain
DATA_RANGES = [(0.25, 0.35), (0.55, 0.65)]  # Local refinement regions

# Sampling parameters
N_COLLOCATION = 1000
CORNER_TOL = 0.00001

# Interface loss parameters
INTERFACE_SAMPLE_EACH = 10
INTERFACE_SAMPLE_RADIUS = 1e-3
MIN_ROOT_GAP = 0.03

# Level-set evolution parameters
BAND_EPS = 0.05
EIKONAL_DELTA_EPS = 1e-4
H_KNN = 0.05
TAU = 1e-3
DT = 1.0
N_INNER = 100

# Visualization parameters
PLOT_INTERVAL = 100
SAVE_INTERVAL = 500
SAVE_INTERVAL_EARLY = 500
SAVE_INTERVAL_LATE = 500
SAVE_EPOCH_CUTOFF = 10000
TRUE_JUMPS = (0.30, 0.60)

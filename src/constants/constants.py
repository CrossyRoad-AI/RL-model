# Shared memory
SHARED_MEMORY_FILENAME = "Global\\crossyroad-memorymapt"
SHARED_MEMORY_SIZE = 1200

FIELDS_TO_QUERY = ["grass", "trees", "water", "water-lily"]
PAD_PER_FIELDS = [0, 0, 0, 3 * 20 * 2] # [600, 120, 200, 80]

# Game window infos
GAME_WINDOW_NAME = "Crossy road"

# Hex codes for keyboard keys
# Full list available at: https://msdn.microsoft.com/en-us/library/dd375731
KEY_W = 0x57
KEY_S = 0x53
KEY_A = 0x41
KEY_D = 0x44

KEY_ENTER = 0x0D

# RL model hyperparameteres
LR = 0.0002
NB_GAMES = 1000
GAMMA = 0.99
EPSILON = 1.0
BATCH_SIZE = 64
EPS_MIN = 0.01
EPS_DEC = 1e-3
NB_ACTIONS = 3
INPUT_DIMS = PAD_PER_FIELDS[0] + PAD_PER_FIELDS[1] + PAD_PER_FIELDS[2] + PAD_PER_FIELDS[3] + 2
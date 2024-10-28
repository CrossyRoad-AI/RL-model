# Shared memory
SHARED_MEMORY_FILENAME = "Global\\crossyroad-memorymap"
SHARED_MEMORY_SIZE = 5000

FIELDS_TO_QUERY = ["grass", "trees", "water", "water-lily"]
PAD_PER_FIELDS = [600, 120, 200, 80]

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
LR = 0.001
NB_GAMES = 10000
GAMMA = 0.99
EPSILON = 1.0
BATCH_SIZE = 64
EPS_MIN = 0.01
EPS_DEC = 1e-4
NB_ACTIONS = 4
INPUT_DIMS = 1004
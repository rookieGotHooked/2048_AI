from enum import Enum

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
GAME_VIEW_WIDTH = 400
GAME_VIEW_HEIGHT = 400
DEFAULT_MARGIN = 8
DEFAULT_BUTTON_WIDTH = 16
DEFAULT_BUTTON_HEIGHT = 1
FPS = 30
FONT = 'freesansbold.tff'
TK_FONT = 'Arial'
FONT_SIZE = 24
TK_FONT_SIZE = 18
BOARD_INIT_VALUES = [[0 for _ in range(4)] for _ in range(4)]
# BOARD_INIT_VALUES = [[2, 0, 4, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
# BOARD_INIT_VALUES = [[2, 0, 4, 0], [2, 8, 4, 0], [4, 0, 0, 0], [4, 0, 0, 0]]
# BOARD_INIT_VALUES = [[32, 8, 4, 2], [2, 32, 8, 4], [2, 0, 4, 2], [0, 2, 4, 8]]
# BOARD_INIT_VALUES = [[32, 32, 64, 64], [8, 16, 16, 8], [4, 2, 4, 2], [2, 16, 2, 0]]
# BOARD_INIT_VALUES = [[2, 4, 8, 16],
#                 [32, 64, 128, 256],
#                 [512, 1024, 2048, 4096],
#                 [8192, 16384, 32768, 65536]]

COLORS = {0: (204, 192, 179),
          2: (238, 228, 218),
          4: (237, 224, 200),
          8: (242, 177, 121),
          16: (245, 149, 99),
          32: (246, 124, 95),
          64: (246, 94, 59),
          128: (237, 207, 114),
          256: (237, 204, 97),
          512: (237, 200, 80),
          1024: (237, 197, 63),
          2048: (237, 194, 46),
          'light_text': (249, 246, 242),
          'dark_text': (119, 110, 101),
          'other': (0, 0, 0),
          'bg': (187, 173, 160)}


class Directions(Enum):
    Right = 'Right'
    Left = 'Left'
    Up = 'Up'
    Down = 'Down'


class States(Enum):
    WIN = 'WIN'
    LOOSE = 'LOOSE'


TARGET = 2048
DATA_DIR_NAME = 'data'
TRAIN_DIR_NAME = 'train_logs'
NEURAL_NET_TRAINING_RATE = 0.3
NEURAL_NET_MAX_EPOCHS = 2000


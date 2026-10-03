# ---- Map size ----
GRID_HEIGHT = 30          # number of rows
GRID_WIDTH = 44           # number of columns
CELL_SIZE_M = 2.0         # each cell is 2 m x 2 m (modelling choice)

# ---- Terrain classes (the numbers stored in the map) ----
FREE = 0
DUST = 1
ROCK = 2
CRATER = 3
STEEP = 4

TERRAIN_NAMES = {
    FREE: "free ground",
    DUST: "dust",
    ROCK: "rock",
    CRATER: "crater",
    STEEP: "steep slope",
}

# One colour per class, in the same order as the numbers above
TERRAIN_COLORS = ["#b0664a", "#d6b084", "#8b8d99", "#1b1512", "#6a3a35"]

# ---- Planet generation settings ----
ROCK_FRACTION = 0.05      # about 5% of cells contain a rock
NUM_CRATERS = 4
NUM_HILLS = 3
NUM_DUST_PATCHES = 5
STEEP_SLOPE_DEG = 24      # slopes above this are "steep"

# ---- Rover start and destination, as (row, column) ----
START = (GRID_HEIGHT - 3, 2)
GOAL = (2, GRID_WIDTH - 3)         
# ---- Rover movement ----
# Each move is (change in row, change in column). Row 0 is the top, so North is -1.
MOVES = {
    "N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
    "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1),
}

# Terrain the rover cannot enter
HAZARDS = (ROCK, CRATER, STEEP)   
# ---- Camera simulation ----
PIXELS_PER_CELL = 4       # each map cell becomes a 4x4 block of pixels
CAMERA_RANGE = 7          # the rover sees 7 cells in each direction

# Typical brightness of each terrain (0 = black, 1 = white).
# The ranges overlap on purpose, so the AI has a real problem to solve.
BASE_BRIGHTNESS = {FREE: 0.55, DUST: 0.70, ROCK: 0.45, CRATER: 0.20, STEEP: 0.40}

# How grainy each terrain looks. Rocks are rough, dust is smooth.
TEXTURE_NOISE = {FREE: 0.05, DUST: 0.03, ROCK: 0.15, CRATER: 0.06, STEEP: 0.08}

SENSOR_NOISE = 0.03       # random noise added by the camera itself
HAZE_BRIGHTNESS = 0.65    # the grey that dust haze washes things toward  
# ---- Path planning costs ----
COST_FREE = 1.0
COST_DUST = 3.0
COST_NEAR_HAZARD_1 = 2.5
COST_NEAR_HAZARD_2 = 1.0
# ---- Belief / unknown cells ----
UNKNOWN = -1
COST_UNKNOWN = 1.6

                                                                          
                                                                          
                                                                          
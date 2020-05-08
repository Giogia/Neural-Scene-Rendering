# MODEL SETTINGS
MODELS = ['Fox']
EXTENSION = 'fbx'
POLY_NUMBER = 250
SCALE = 0.1

# CAMERAS SETTINGS
CAMERAS_NUMBER = 10
DISTANCE = 2
HEIGHT = 2
FOV = 35
NEAR_PLANE = 0.1
FAR_PLANE = 5 * DISTANCE

# CAMERA POSITION SETTINGS
DISTANCE_NOISE: float = DISTANCE * 0.5  # meters
HEIGHT_NOISE: float = HEIGHT  # meters
YAW_NOISE: float = 10  # degrees
FOV_NOISE: float = 10  # degrees

# RENDERING SETTINGS
RESOLUTION_X = 667
RESOLUTION_Y = 1024
OUTPUT_RESOLUTION = 100  # percentage

# ANIMATION SETTINGS
START_FRAME = 150
END_FRAME = 250

# FILE SETTINGS
MODEL_FILE_HEADER = ['Name', 'Location', 'Rotation', 'Scale', 'Dimensions']

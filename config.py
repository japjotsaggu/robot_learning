import time


MODE = 'development'
SEED = 1

# Set the size of the window.
WINDOW_SIZE = 300

# Set the frame rate for pygame, which determines how quickly the program runs.
# In our evaluation of your code, this will be set at 30.
FRAME_RATE = 30

ALGORITHM = "bc"   # "bc" | "dagger" | "residual_rl"
SAFETY_BUFFER = 6.0     # never spend below this (covers 1 reset worth of cost)
import subprocess
from itertools import product

# Specify the path to the child Python script
config_file = 'config.py'



# Define binary values
binary_values = [0, 1]

# Generate all combinations
combinations = list(product(binary_values, repeat=3))

# Filter out the combination (0, 0, 0)
filtered_combinations = [comb for comb in combinations if any(comb)]

# Print or use the combinations as needed
for combination in filtered_combinations:

    audio_flag, depth_flag, video_flag = combination

    # Convert integers to strings because subprocess.run() expects strings
    arguments = [str(arg) for arg in [audio_flag, depth_flag, video_flag]]

    # Call the child Python script with integer arguments
    subprocess.run(['python3', config_file] + arguments)
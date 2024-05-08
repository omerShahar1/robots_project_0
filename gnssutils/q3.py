import numpy as np
import pandas as pd

# Read the data from CSV file into a DataFrame
data = pd.read_csv('gps_data.csv')

# Function to calculate distance between satellite and point
def distance(satellite_pos, point_pos):
    return np.linalg.norm(satellite_pos - point_pos)

# Function to calculate error
def error(point_pos, satellites_pos, pseudo_ranges):
    return np.array([distance(satellites_pos[i], point_pos) - pseudo_ranges[i] for i in range(len(satellites_pos))])

# Least squares algorithm
def least_squares(satellites_pos, pseudo_ranges):
    initial_guess = np.zeros(3)  # Initial guess for the point position
    result = optimize.least_squares(error, initial_guess, args=(satellites_pos, pseudo_ranges))
    return result.x

# Extract relevant columns from the DataFrame
satellite_data = data[['Sat.X', 'Sat.Y', 'Sat.Z']].values
pseudo_ranges = data['Pseudo-Range'].values

# Initial guess for the point position
initial_guess = np.zeros(3)

# Compute the position using least squares
position = least_squares(satellite_data, pseudo_ranges)
print("Estimated Position (X, Y, Z):", position)
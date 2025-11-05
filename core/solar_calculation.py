from pysolar.solar import *
import numpy as np

def calculate_solar(date, latitude, longitude):
    alpha = get_altitude(latitude, longitude, date)
    azimuth = get_azimuth(latitude, longitude, date)

    # Convert degrees to radians for numpy trigonometric functions
    alpha_rad = np.radians(alpha)
    azimuth_rad = np.radians(azimuth)

    # Correct conversion from spherical (azimuth, altitude) to Cartesian coordinates
    # Using ENU (East-North-Up) convention
    x1 = np.cos(alpha_rad) * np.sin(azimuth_rad)  # East
    x2 = np.cos(alpha_rad) * np.cos(azimuth_rad)  # North
    x3 = np.sin(alpha_rad)                        # Up

    s_w = np.array([x1, x2, x3])

    return alpha, azimuth, s_w

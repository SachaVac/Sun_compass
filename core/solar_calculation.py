from pysolar.solar import *
import numpy as np

def calculate_solar(date, latitude, longitude):
    alpha = get_altitude(latitude, longitude, date)
    beta = get_azimuth(latitude, longitude, date)


    x1 = np.cos(alpha)*np.sin(beta)
    x2 = np.cos(alpha)*np.sin(beta)
    x3 = np.sin(alpha)

    s_w = [x1, x2, x3]

    return alpha, beta, s_w



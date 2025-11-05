from pysolar.solar import *
import datetime
import numpy as np

latitude = 50.73235690065898 # set your coordinates
longitude = 15.782290488599608

date = datetime.datetime(2025, 10, 22, 8, 00, 1, tzinfo=datetime.timezone.utc) # set time and date - datetime.datetime( YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, mikrosecond, tzinfo=...)
alpha = get_altitude(latitude, longitude, date)
beta = get_azimuth(latitude, longitude, date)


x1 = np.cos(alpha)*np.sin(beta)
x2 = np.cos(alpha)*np.sin(beta)
x3 = np.sin(alpha)

s_w = np.array([x1, x2, x3])
s_w/= np.linalg.norm(s_w)

print(s_w)
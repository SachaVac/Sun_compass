# Sun_compass
- semestral project Winter 2025 by Vaclav Sacha

# Task
Create real-time system to be installed on mobile robot. System (camera) should be able to follow sun and set an exact location of north. 

# Inputs
- real time info.
- GPS location
- accelerometer, gyroscope - rotation of body
- camera input of sky (180 deg.) + camera placement on robot body

# Hardware needed
- Control board
- 180 deg fisheye cam
- GPS sensor
- IMU - accelerometer unit
# Process
## Computations
### Incomes
- latitude, longitude
- time, date

### Outcomes
- sun altitude + sun azimuth 

```python
from pysolar.solar import *
import datetime

latitude = 42.206
longitude = -71.382

date = datetime.datetime(2007, 2, 18, 15, 13, 1, 130320, tzinfo=datetime.timezone.utc)
alpha = get_altitude(latitude, longitude, date)
beta = get_azimuth(latitude, longitude, date)
```
- vector direction from exact place on earth to the sun

```python
x1 = cos(alpha)*sin(beta)
x2 = cos(alpha)*sin(beta)
x3 = sin(alpha)

s_w = [x1, x2, x3]
```



[Pysolar library](https://pysolar.readthedocs.io/en/latest/)
[PVLib - solarposition](https://pvlib-python.readthedocs.io/en/stable/reference/solarposition.html)
## Vision


## Vectors up

## Direction
[Triad method](https://en.wikipedia.org/wiki/Triad_method)

## Connection

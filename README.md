# Sun_compass
- CTU in Prague - Winter 2025 semestral project by Vaclav Sacha

# Task
Create real-time system to be installed on mobile robot. System (camera) should be able to follow sun and set an exact location of north. 

# Inputs
- real time info
- GPS location
- accelerometer
- camera input of sky (180 deg) + camera placement on robot body

# Hardware needed
- Control board
- 180 deg fisheye cam
- GPS sensor
- IMU - accelerometer unit
# Process
## Computations
### Inputs
- latitude, longitude
- time, date

### Outputs
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

[Pysolar library](https://pysolar.readthedocs.io/en/latest/)<br />
[PVLib - solarposition](https://pvlib-python.readthedocs.io/en/stable/reference/solarposition.html)
## Vision

### Inputs
- fisheye cam captured image
- translation matrix robot_body->camera
### Outputs
- vector of direction s_r (robot->sun)
### Tetsing set
[e-con cam](https://www.e-consystems.com/industrial-cameras/ar0234-usb3-global-shutter-camera.asp)<br />
[Objektiv](https://rpishop.cz/sirokouhle-objektivy/2808-arducam-17mm-f2-m12-objektiv-m25170h12.html?utm_source=google&utm_medium=cpc&utm_campaign=CZ-GS-No%2Funder%20index%20produkty&utm_id=21746766956&gad_source=1&gad_campaignid=21746766956&gbraid=0AAAAApAQKp1C6CxnpAR_PiTUvx5-ZkFq8&gclid=CjwKCAjwgeLHBhBuEiwAL5gNEWODSiVjbud9-QqEhCNyx2jn57uVTvGaEGmp_8_mULWh2CRNvubSbBoCBoIQAvD_BwE)<br />

[Raspberry HQ cam - M12](https://rpishop.cz/mipi-kamerove-moduly/5603-raspberry-pi-hq-camera-m12-mount.html?utm_source=google&utm_medium=cpc&utm_campaign=CZ-PMax-Raspberry%20Pi&utm_id=19691368073&gad_source=1&gad_campaignid=19691725348&gbraid=0AAAAApAQKp0wLMGZbLberkrktuDELLmqT&gclid=CjwKCAjw3tzHBhBREiwAlMJoUi85UexceyijH-CBFHmzGrSzuKT_8Cs40828wSok_U4as8vut44m9RoCqgQQAvD_BwE)<br />
[Fish eye lens 180 deg](https://botland.cz/objektivy-fotoaparatu-pro-raspberry-pi/17066-objektiv-fisheye-m12-156-mm-s-adapterem-pro-fotoaparat-raspberry-pi-arducam-ln031-5904422378349.html)<br />
[Rasdpberry pi 5](https://rpishop.cz/raspberry-pi-5/6498-raspberry-pi-5-8gb-ram.html) 

## Vectors up
purpose: get 2nd vector for Triad method
### World
ENU (East North Up) coordinate system
```bash
u_w = (0, 0, 0)
```
### Robot
u_r from accelerometer unit - needed to measure while robot standing

### Tetsing set
[ISM330DHCX](https://botland.cz/9dof-imu-senzory/16458-ism330dhcx-6dof-imu-3osy-akcelerometr-a-gyroskop-adafruit-4502-5904422344528.html)<br />
or <br />
[BNO085](https://botland.cz/9dof-imu-senzory/22113-bno085-9-dof-imu-fusion-breakout-3osy-akcelerometr-gyroskop-a-magnetometr-adafruit-4754.html)
## Direction
### Inputs 
- (s_w s_r u_w u_r) - 4 vectors in 2 coordinate systems (robot/world)
### Outputs
- T_wr - transformation matrix between robot body and ENU system<br />
[Triad method](https://en.wikipedia.org/wiki/Triad_method)<br />
### Example 
robot moving straight p_r = (1 0 0)
```bash
p_w = T_wr*p_r
```
p_w giving robot movement in ENU coordinates



    
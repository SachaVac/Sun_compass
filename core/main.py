import sys, os, datetime
import solar_calculation, image_processor, orientation_solver

#inputs for pysolar
latitude = 50.73235690065898 # set your coordinates
longitude = 15.782290488599608
date = datetime.datetime(2025, 10, 22, 8, 00, 1, tzinfo=datetime.timezone.utc)

#path to cam calibration
path = "see3cam_fisheye_calib.npz"


if __name__ == "__main__":
    al_w, el_w, s_w = solar_calculation.calculate_solar(date, latitude, longitude)    
    al_c, el_c, s_c = image_processor.calculate_camsolar(path, True)
    
    u_w = [0, 0, 1]
    u_c = [0, 0, 1]

    print(al_w, el_w, s_w, al_c, el_c, s_c)



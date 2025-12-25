from pysolar.solar import *
import datetime
import numpy as np
import cv2
import sys
from pathlib import Path
import time

def get_sun_in_world(latitude, longitude, date):

    alpha = get_altitude(latitude, longitude, date)
    beta = get_azimuth(latitude, longitude, date)

    x1 = np.cos(alpha)*np.sin(beta)
    x2 = np.cos(alpha)*np.cos(beta)
    x3 = np.sin(alpha)

    s_w = np.array([x1, x2, x3])
    s_w/= np.linalg.norm(s_w)
    return s_w

def open_calibrationIMG (path): #load calibration of FISHEYE
    data = np.load(path)
    
    return data["K"], data["D"], data["size"]

def check_resolution(act_w, act_h, cal_w, cal_h):
    if act_w != cal_w or act_h != cal_h:
        print("ERROR: cam resolution not fitting to calib resolution {act_w} x {act_h} vs {calib_w} x {calib_h}")
    return

def find_sun_pose(frame):
    gamma_correction = 2.2 #2.2 default
    highest_percentile = 99.6 #99.0
    median_filter = 3 #default 5

    #prepare image
    grayIMG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #transformation of color into gray
    norm = np.clip((grayIMG/255.0)** gamma_correction, 0,1) #normalized values
    threshold = np.percentile(norm, highest_percentile)

    #setup mask
    mask = (norm >= threshold).astype(np.uint8)*255
    mask = cv2.medianBlur(mask, median_filter)
    
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if num < 2: #at least one component for every time (background = index 1)
        return None, mask
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    max = int(np.argmax(areas))+1   
    x,y = centroids[max]


    return (float(x), float(y)), mask

def triad_method(s_w, u_w, s_r, u_r):
    z_w = s_w
    z_r = s_r

    base_w = np.cross(u_w, s_w)
    base_r = np.cross(u_r, s_r)

    y_w = base_w/np.linalg.norm(base_w)
    y_r = base_r/np.linalg.norm(base_r)

    x_w = np.cross(z_w, y_w)
    x_r = np.cross(z_r, y_r)

    M_w = np.column_stack((x_w, y_w, z_w))
    M_r = np.column_stack((x_r, y_r, z_r))

    T_wr = M_w @ M_r.T



    return T_wr

def project_vector_to_fisheye(vector_3d, K, D):
    X, Y, Z = vector_3d
    
    
    theta = np.arctan2(np.sqrt(X**2 + Y**2), Z)
    phi = np.arctan2(Y, X) # azimuth xy


    f = (K[0, 0] + K[1, 1]) / 2.0 
    r_distorted = f * theta


    c_x = K[0, 2]
    c_y = K[1, 2]
    
    x_pixel = r_distorted * np.cos(phi) + c_x
    y_pixel = r_distorted * np.sin(phi) + c_y
    
    return int(round(x_pixel)), int(round(y_pixel))

def main():

    
    # INPUTS NEEDED
    latitude = 50.0761783
    longitude = 14.4187586

    
    # HW DEPENDENCIES
    calib_path = "fisheye_calib.npz" 
    cam_index = 2 
    
    # IMPORT CALIBRATION
    K, D, (w, h)  = open_calibrationIMG(calib_path)
    c_x, c_y = K[0, 2], K[1, 2]
    center_point = (int(round(c_x)), int(round(c_y)))

    # CAMERA INIT
    cam = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cam.isOpened():
        print("ERROR: NO CAM OPENED")        
        return
    print(f"INFO: CAM RUNNING WITH RESOLUTION {w}x{h}.")    
    
    # VECTOR UP - WORLD COORDS
    u_w = np.array([0.0, 0.0, 1.0]) 

    # MAIN LOOP
    while True:
        ret, frame = cam.read()
        if not ret:
            print("ERROR: NO IMAGE LOADED")
            break

        # INPUTS TO LOOP
        date = datetime.datetime.now(datetime.timezone.utc)
        u_r = np.array([0.0, 0.0, 1.0]) # for now just vector up 
        
        s_w = get_sun_in_world(latitude, longitude, date)
        xy, mask = find_sun_pose(frame)
        
        if xy is None:
            text_info = "ERROR: NO SUN FOUND"
            cv2.putText(frame, text_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            comb_vis = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
            cv2.imshow("Sun_Compass", comb_vis)
        
        else:
            sun_pixel_pos = np.array([[[xy[0], xy[1]]]], dtype=np.float32)
            
            s_r_projection = cv2.fisheye.undistortPoints(sun_pixel_pos, K, D)
            x_n, y_n = s_r_projection[0, 0]
            

            s_r_unnorm = np.array([x_n, y_n, 1.0]) 
            s_r = s_r_unnorm / np.linalg.norm(s_r_unnorm) 
            
            T_wr = triad_method(s_w, u_w, s_r, u_r)
            n_r = robot_north_direction(T_wr) 
            
            # VISUALIZE
            s_pixel_x, s_pixel_y = project_vector_to_fisheye(s_r, K, D)
            n_pixel_x, n_pixel_y = project_vector_to_fisheye(n_r, K, D)
            sun_point = (s_pixel_x, s_pixel_y)
            north_point = (n_pixel_x, n_pixel_y)

            # sun direction
            cv2.drawMarker(frame, sun_point, (0,200,0), cv2.MARKER_TILTED_CROSS, 40, 5) # Slunce poloha
            cv2.arrowedLine(frame, center_point, sun_point, (0,255,0), 8, cv2.LINE_AA, 0, 0.1) # Zelená šipka (směr Slunce)

            # north direction
            cv2.arrowedLine(frame, center_point, north_point, (0, 0, 255), 15, cv2.LINE_AA, 0, 0.2) # Červená šipka (Sever)
            cv2.putText(frame, "NORTH", (n_pixel_x + 20, n_pixel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            # text
            text_twr = f"T_wr: R[0,0]={T_wr[0,0]:.2f}, R[1,1]={T_wr[1,1]:.2f}, R[2,2]={T_wr[2,2]:.2f}"
            text_north = f"N_r: ({n_r[0]:.2f}, {n_r[1]:.2f}, {n_r[2]:.2f})"
            
            cv2.putText(frame, text_twr, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, text_north, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # vis
            comb_vis = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
            cv2.imshow("Sun_Compass", comb_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()






def robot_north_direction(T_wr):
    n_w = np.array([0,1,0])
    n_r = T_wr.T @ n_w
    return n_r

if __name__ == "__main__":
    main()




    


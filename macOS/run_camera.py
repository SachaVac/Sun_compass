import cv2
import numpy as np
import sys
from pathlib import Path
import time

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

def infere_w_calib():
    return

def capture_from_cam(calib_path, index):
    K, D, (w,h)  = open_calibrationIMG(calib_path) #get calibration numbers
    print (f"camres {w}x{h}")
    cam = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cam.set(cv2.CAP_PROP_FPS, 30)



    if not cam.isOpened():
        print("ERROR: cam not opened")

    check_resolution(w,h,int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        ret, frame = cam.read()

        xy, mask = find_sun_pose(frame)
        x,y = xy
        mask = cv2.resize(mask, (w, h))
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # K should be [[fx 0 c_x][0 fy c_y][0 0 fz=1]] 
        c_x = K[0, 2] # centers of picture
        c_y = K[1, 2]

        x_v = x-c_x # vectors in picture
        y_v = y-c_y

        azimuth_deg = np.degrees(np.arctan2(x_v, y_v))
        azimuth_deg = (azimuth_deg + 360) % 360

        print(f"vector X{x_v}, vector Y{y_v}")
        print(f"AZIMUTH DEG {azimuth_deg}")

        sun_pixel_pos = np.array([[[x, y]]], dtype=np.float32)
        sun_vector = cv2.fisheye.undistortPoints(sun_pixel_pos, K, D)
        #breakpoint()
        print(f"SUN VECTOR {sun_vector}")
        x_n, y_n = sun_vector[0, 0]
        r_n = np.sqrt(x_n**2 + y_n**2)
        elevation_rad = np.arctan2(r_n, 1.0) 
        elevation_deg = 90 - np.degrees(elevation_rad)

        
        

        if xy is not None:
            sun_point = (int(round(x)), int(round(y))) 
            center_point = (int(round(c_x)), int(round(c_y)))
            
            cv2.drawMarker(frame, (int(round(x)), int(round(y))), (0,200,0), cv2.MARKER_TILTED_CROSS, 40, 10)
            cv2.drawMarker(frame, (int(round(c_x)), int(round(c_y))), (0,200,0), cv2.MARKER_TILTED_CROSS, 40, 10)

            cv2.arrowedLine(frame, center_point, sun_point, (0,0,200), 10, cv2.LINE_AA, 0, 0.1)

            # cv2.putText(image, azimuth_text, text_org, font, font_scale, color, text_thickness, cv2.LINE_AA)

            text_x_pos = sun_point[0] + 15
            text_y_pos = sun_point[1] - 15
            text_org = (text_x_pos, text_y_pos)

            # draw azimuth
            azimuth_text = f"Azimut: {azimuth_deg:.2f}deg"
            cv2.putText(frame, azimuth_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,200,0), 2, cv2.LINE_AA)

            elevation_text = f"Elevation: {elevation_deg:.2f}deg"
            cv2.putText(frame, elevation_text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,200,0), 2, cv2.LINE_AA)



            #draw vector
            if x_n < -1:
                sun_vector_txt = "Vector_normalized: ERROR"
            else:
                sun_vector_txt = f"Vector_normalized: X {float(x_n):.2f} Y {y_n:.2f} Z 1.00"
            cv2.putText(frame, sun_vector_txt, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,200,0), 2, cv2.LINE_AA)

            comb_vis = np.hstack((frame, mask_color))
            cv2.imshow("Sun detector", comb_vis)
            key = cv2.waitKey(1) & 0xFF
            print(f"vector X{x}, vector Y{y}")

            """q = input("quit(q)/save Vis(s)/new image(n)\n")
            if q == "q":
                cv2.destroyAllWindows()
                return
            elif q == "s":
                
                timestamp = int(time.time())
                save_filename = f"sun_vis_{timestamp}.png"
                success = cv2.imwrite(save_filename, comb_vis)
                
                if success:
                    print(f"Viz saved as{save_filename}")"""

                
    

            


def open_data(calib_path, data_path):
    K, D, (w,h)  = open_calibrationIMG(calib_path) #get calibration numbers
    
    image_files = list(Path(data_path.split('*')[0]).parent.glob(Path(data_path).name))

    if not image_files:
        print(f"ERROR: No images found")
        return
    
    for i, file_path in enumerate(image_files):
        frame = cv2.imread(str(file_path))

        frame = cv2.resize(frame.copy(), (w, h)) #resize to camera
        xy, mask = find_sun_pose(frame)
        x,y = xy
        mask = cv2.resize(mask, (w, h))
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # K should be [[fx 0 c_x][0 fy c_y][0 0 fz=1]] 
        c_x = K[0, 2] # centers of picture
        c_y = K[1, 2]

        x_v = x-c_x # vectors in picture
        y_v = y-c_y

        azimuth_deg = np.degrees(np.arctan2(x_v, y_v))
        azimuth_deg = (azimuth_deg + 360) % 360

        print(f"vector X{x_v}, vector Y{y_v}")
        print(f"AZIMUTH DEG {azimuth_deg}")


        sun_pixel_pos = np.array([[[x, y]]], dtype=np.float32)
        sun_vector = cv2.fisheye.undistortPoints(sun_pixel_pos, K, D)
        print(f"SUN VECTOR {sun_vector}")

        x_n, y_n = sun_vector[0, 0]
        r_n = np.sqrt(x_n**2 + y_n**2)
        elevation_rad = np.arctan2(r_n, 1.0) 
        elevation_deg = 90 - np.degrees(elevation_rad)


        
        

        
        
        
        
        if xy is not None:
            x,y = xy

            sun_point = (int(round(x)), int(round(y))) 
            center_point = (int(round(c_x)), int(round(c_y)))
            cv2.drawMarker(frame, (int(round(x)), int(round(y))), (0,200,0), cv2.MARKER_TILTED_CROSS, 40, 10)
            cv2.drawMarker(frame, (int(round(c_x)), int(round(c_y))), (0,200,0), cv2.MARKER_TILTED_CROSS, 40, 10)

            cv2.arrowedLine(frame, center_point, sun_point, (0,0,200), 10, cv2.LINE_AA, 0, 0.1)

            # cv2.putText(image, azimuth_text, text_org, font, font_scale, color, text_thickness, cv2.LINE_AA)

            text_x_pos = sun_point[0] + 15
            text_y_pos = sun_point[1] - 15
            text_org = (text_x_pos, text_y_pos)

            # draw azimuth
            azimuth_text = f"Azimut: {azimuth_deg:.2f}deg"
            cv2.putText(frame, azimuth_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,200,0), 2, cv2.LINE_AA)

            elevation_text = f"Elevation: {elevation_deg:.2f}deg"
            cv2.putText(frame, elevation_text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,200,0), 2, cv2.LINE_AA)



            #draw vector
            if x_n < -1:
                sun_vector_txt = "Vector_normalized: ERROR"
            else:
                sun_vector_txt = f"Vector_normalized: X {float(x_n):.2f} Y {y_n:.2f} Z 1.00"
            cv2.putText(frame, sun_vector_txt, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,200,0), 2, cv2.LINE_AA)

            comb_vis = np.hstack((frame, mask_color))
            cv2.imshow("Sun detector", comb_vis)
            key = cv2.waitKey(1) & 0xFF


            
            q = input("quit(q)/save Vis(s)/new image(n)\n")
            if q == "q":
                cv2.destroyAllWindows()
                return
            elif q == "s":
                
                timestamp = int(time.time())
                save_filename = f"sun_vis_{timestamp}.png"
                success = cv2.imwrite(save_filename, comb_vis)
                
                if success:
                    print(f"Viz saved as{save_filename}")


            

    return

if __name__ == "__main__":
    calib_path = "fisheye_calib.npz"
    data_path = "data/IMG*.jpeg" 
    cam_index = 0
    

    while True:
        state = input("cam(c)/data(d)/quit(q)\n")
        if state == "c":
            capture_from_cam(calib_path, cam_index)
        elif state == "d":
            open_data(calib_path, data_path)
        elif state == "q":
            sys.exit()
        else:
            print("ERROR: wrong input")
    


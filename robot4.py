import numpy as np
import sys
import time
import cv2
from SDK import ELITE
from CameraCalibration import CameraCalibrationHelper
import threading
from pyModbusTCP.client import ModbusClient
#version prueba5

"""
Important parameters to twitch:

- gray_thresh: (0 to 255) the higher the value the less pixels it will detect
      twitch it till the camera detects perfectly the wood piece.

- area_thresh: (0 to 480x640) the perfectly detect wood piece should be around
        50000 pixels, the higher the value the more precise the center and
        angle of the piece will be, but with the risk of not detecting it.
        twitch it till the camera detects perfectly the center and angle
        of the wood piece.

- cam_min_lim/cam_max_lim: ((0, 0) to (640, 480)) the limits of them mask within
        the camera, it should be align with the conveyor belt so that in the mask
        there is only th conveyor belt, otherwise the camera will detect noise
        from the background. These limits will be displayed in the camera as a
        black rectangle.
        twitch it till the black rectangle is completely engulfed by the conveyor
        belt.
"""

class PalletizingRobot:
        
    def __init__(self, robot_ip, gray_thresh = 100, area_thresh = 45000, 
                 cam_min_lim = (0, 0), cam_max_lim = (640, 480)):

        self.robot = ELITE(robot_ip)
        self.frame = None
        self.camera = None
        self.piece_num = 0 # number of the piece that is being picked
        self.object_detected = False
        self.gray_thresh = gray_thresh
        self.area_thresh = area_thresh
        self.cam_min_lim = cam_min_lim
        self.cam_max_lim = cam_max_lim
        self.plc = ModbusClient(host="169.168.0.241", port=502, unit_id=1, auto_open=True)
        self.last_center = None
        self.last_angle = None
        self.last_detection_ok = False
        self.wait_pose = [-143.44, 430.239,-30, 0, 0, 0]

        
    def initialize_camera(self):
        self.helper = CameraCalibrationHelper()
        self.camera = self.helper.initialize_raspicam(headless = True, sensor_index = -1)
        self.helper.calibrate_raspberry()
        time.sleep(1)
        self.camera_available = True
    

    def camera_thread(self):
         while True:
            frame = self.camera.capture_array()[:, :, 0:3]
            frame = self.helper.correct_image(frame)
            frame, mask, center, angle, success = self.detect_box(frame, self.gray_thresh, self.area_thresh, iter_=1)
            if success and (abs(angle) < 10 or abs(angle - 90) < 10):
                self.last_center = center
                self.last_angle = angle
                self.last_detection_ok = True
            else:
                self.last_detection_ok = False
    
    def detect_box(self, frame, gray_thresh, area_thresh, iter_ = 1):
      
        aux = frame[self.cam_min_lim[1]:self.cam_max_lim[1],
                    self.cam_min_lim[0]:self.cam_max_lim[0]]
        
        # Grayscale detection
        gray_image = cv2.cvtColor(aux, cv2.COLOR_BGR2GRAY)
        
        # mask thresh
        _, mask = cv2.threshold(gray_image, gray_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, None, iterations=iter_)
        mask = cv2.dilate(mask, None, iterations=iter_)
        
        # find contour with largest area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if there is contour:
        if not contours:
            return frame, mask, None, None, 0
        
        # check if area is over the min threshold
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < area_thresh:
            return frame, mask, None, None, 0
        
        # detecta posicion y orientacion
        rect = cv2.minAreaRect(largest_contour)
        center, (width, height), angle = rect
        
        if width < height:
            angle += 90
            
        center = (int(center[0]) + self.cam_min_lim[0], int(center[1]) + self.cam_min_lim[1])
    
        # draw over frame
        box = cv2.boxPoints(rect).astype(int)
        box[:, 0] =  box[:, 0] + self.cam_min_lim[0]
        box[:, 1] =  box[:, 1] + self.cam_min_lim[1]
        frame = cv2.drawContours(frame, [box], 0, (0, 255, 0), 2) 
        frame = cv2.circle(frame, center, 5, (255, 0, 0), 10)
        return frame, mask, center, angle, 1
        
    
    def map_camara2robot(self, center_x, angle):

        # Calculation of the robot delta x position
        self.piece_angle = 90 - angle
        width = 90.0 # mean width of the wood piece
        height = 140.0 # mean height of the wood piece
        beta = np.arctan(height/width) 
        L = np.sqrt((width/2)**2 + (height/2)**2)
        aux = -self.piece_angle
        if self.piece_angle < 0:
            aux = self.piece_angle
        else:
            aux = -self.piece_angle
        self.robot_x = L * np.sin(np.pi + (aux * (np.pi/180)) - beta)
        
        cam_min_x = self.cam_min_lim[0]
        cam_max_x = self.cam_max_lim[0]
        robot_y_min = -150
        robot_y_max = 150
        self.robot_y = ((center_x - cam_min_x) / (cam_max_x - cam_min_x)) * (robot_y_max - robot_y_min) + robot_y_min
        self.robot_angle = angle



    def mozaic_generator(self):
         

        columnas = 2
        col = self.piece_num % columnas
        row = self.piece_num // columnas
        if abs(self.robot_angle) < 10:

        # 0°
            base_x = -663.823
            base_y = -170.596
            base_z = -167.794
            base_rx = 176.509
            base_ry = -3.206
            base_rz = 2.355
            pitch_x = 0
            pitch_y = -60
            pitch_z = 100   # ¡AUMENTA z!

        elif abs(self.robot_angle - 90) < 10:
        # 90°
            base_x = -663.817
            base_y = 342.746
            base_z = -167.798
            base_rx = 178.510
            base_ry = -3.206
            base_rz = 2.355
            pitch_x = 0
            pitch_y = -60
            pitch_z = 100
        else:
            base_x = -643.317
            base_y = 59.351
            base_z = -155.339
            base_rx = 176.808
            base_ry = 1.524
            base_rz = 92.951
            pitch_x = 0
            pitch_y = -60
            pitch_z = 100

        x = base_x + col * pitch_x
        y = base_y + col * pitch_y
        z = base_z + row * pitch_z   #
    
        return [x, y, z, base_rx, base_ry, base_rz]

        
    def pick_and_place(self, center, angle):

    # 1. Calcular coordenadas del robot a partir del centro detectado por cámara
        self.map_camara2robot(center[0], angle)
        x = self.robot_x
        y = self.robot_y
        rz = self.robot_angle
        z_pick =  -92.045  # Ajusta esta altura según tu banda

        pick_pose_down = [x, y, z_pick, 0, 0, rz]
        pick_pose_up = [x, y, z_pick + 100, 0, 0, rz]  # Subida segura

    # 2. Ir a posición de pre-pick (arriba)
        self.robot.move_l_pose(pick_pose_up, speed=20)
        self.robot.wait_until_motion_complete()

    # 3. Bajar para agarrar la pieza
        self.robot.move_l_pose(pick_pose_down, speed=20)
        self.robot.wait_until_motion_complete()

    # 4. Cerrar garra
        self.helper.cerrar_garra()
        time.sleep(1)

    # 5. Subir pieza
        self.robot.move_l_pose(pick_pose_up, speed=20)
        self.robot.wait_until_motion_complete()

    # 6. Generar posición de depósito
        place_pose = self.mozaic_generator()
        place_pose_up = place_pose.copy()
        place_pose_up[2] += 100  # Altura segura sobre la caja

    # 7. Ir sobre la caja/palet
        self.robot.move_l_pose(place_pose_up, speed=20)
        self.robot.wait_until_motion_complete()

    # 8. Bajar a depositar
        self.robot.move_l_pose(place_pose, speed=20)
        self.robot.wait_until_motion_complete()

    # 9. Abrir garra
        self.helper.abrir_garra()
        time.sleep(1)

    # 10. Subir garra tras dejar pieza
        self.robot.move_l_pose(place_pose_up, speed=20)
        self.robot.wait_until_motion_complete()

    # 11. Volver a la posición de espera
        self.robot.move_l_pose(self.wait_pose, speed=20)
        self.robot.wait_until_motion_complete()

    # 12. Actualizar número de pieza
        self.piece_num += 1
        return None

   
    def run(self):
        thread = threading.Thread(target=self.camera_thread, daemon=True)
        thread.start()
        
        if self.robot.connect():
            print("Successfully connected to robot")
        
            # [Incomplete]: basic robot functions...
            # self.robot.something()
            self.robot.set_user_number(4)
            self.robot.set_current_coord(3)
            self.robot.set_tool_number(2)
            self.robot.move_l_pose(self.wait_pose)
            self.robot.wait_until_motion_complete()

            while True:
                plc_coil = self.plc.read_coils(0, 1)
                if plc_coil and plc_coil[0]:
                    if self.last_detection_ok:

                    #Usando datos almacenados — ejecutando pick and place#
                        self.pick_and_place(self.last_center, self.last_angle)

                        while self.plc.read_coils(0, 1)[0]:
                            time.sleep(1)
                        self.last_detection_ok = False  # Resetea bandera para esperar la próxima pieza
                else:
                    print(" No hay datos recientes de detección — no se ejecuta pick and place")
                time.sleep(1)
            else:
                time.sleep(1)
        self.robot.disconnect()

if __name__ == "__main__":
    # Example usage
    robot_ip = "169.168.0.200"
    robot = PalletizingRobot(robot_ip)
    robot.initialize_camera()
    robot.run()

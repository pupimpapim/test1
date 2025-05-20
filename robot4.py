import numpy as np
import sys
import time
import cv2
from SDK import ELITE
from CameraCalibration import CameraCalibrationHelper
import threading
from pyModbusTCP.client import ModbusClient

class PalletizingRobot:

    def __init__(self, robot_ip, gray_thresh=100, area_thresh=45000, 
                 cam_min_lim=(0, 0), cam_max_lim=(640, 480)):

        self.robot = ELITE(robot_ip)
        self.frame = None
        self.camera = None
        self.piece_num = 0
        self.object_detected = False
        self.gray_thresh = gray_thresh
        self.area_thresh = area_thresh
        self.cam_min_lim = cam_min_lim
        self.cam_max_lim = cam_max_lim
        self.plc = ModbusClient(host="169.168.0.241", port=502, unit_id=1, auto_open=True)
        self.last_center = None
        self.last_angle = None
        self.last_detection_ok = False
        self.wait_pose = [-143.44, 430.239, -30, 0, 0, 0]

    def initialize_camera(self):
        print("[INFO] Inicializando cámara...")
        self.helper = CameraCalibrationHelper()
        try:
            self.camera = self.helper.initialize_raspicam(headless=True, sensor_index=-1)
            print("[INFO] Cámara inicializada:", self.camera)
            self.helper.calibrate_raspberry()
            print("[INFO] Calibración completada.")
            self.camera_available = True
        except Exception as e:
            print("[ERROR] No se pudo inicializar la cámara:", e)
            self.camera_available = False
        time.sleep(1)

    def camera_thread(self):
        while True:
            try:
                frame = self.camera.capture_array()[:, :, 0:3]
                frame = self.helper.correct_image(frame)
                frame, mask, center, angle, success = self.detect_box(
                    frame, self.gray_thresh, self.area_thresh, iter_=1
                )
                # Debug:
                if success:
                    print(f"[INFO] Pieza detectada. Centro: {center}, Ángulo: {angle:.2f}")
                if success and (abs(angle) < 10 or abs(angle - 90) < 10):
                    self.last_center = center
                    self.last_angle = angle
                    self.last_detection_ok = True
                else:
                    self.last_detection_ok = False
                # Si quieres ver el video mientras, descomenta:
                # frame = cv2.rectangle(frame, self.cam_min_lim, self.cam_max_lim, (0, 0, 0), 10)
                # cv2.imshow("Robot Camera", frame)
                # cv2.imshow("Robot Camera mask", mask)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            except Exception as e:
                print("[ERROR] En hilo de cámara:", e)
                time.sleep(1)

    def detect_box(self, frame, gray_thresh, area_thresh, iter_=1):
        aux = frame[self.cam_min_lim[1]:self.cam_max_lim[1],
                    self.cam_min_lim[0]:self.cam_max_lim[0]]
        gray_image = cv2.cvtColor(aux, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, gray_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, None, iterations=iter_)
        mask = cv2.dilate(mask, None, iterations=iter_)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame, mask, None, None, 0
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < area_thresh:
            return frame, mask, None, None, 0
        rect = cv2.minAreaRect(largest_contour)
        center, (width, height), angle = rect
        if width < height:
            angle += 90
        center = (int(center[0]) + self.cam_min_lim[0], int(center[1]) + self.cam_min_lim[1])
        box = cv2.boxPoints(rect).astype(int)
        box[:, 0] = box[:, 0] + self.cam_min_lim[0]
        box[:, 1] = box[:, 1] + self.cam_min_lim[1]
        frame = cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        frame = cv2.circle(frame, center, 5, (255, 0, 0), 10)
        return frame, mask, center, angle, 1

    def map_camara2robot(self, center_x, angle):
        self.piece_angle = 90 - angle
        width = 90.0
        height = 140.0
        beta = np.arctan(height / width)
        L = np.sqrt((width / 2)**2 + (height / 2)**2)
        aux = -self.piece_angle
        if self.piece_angle < 0:
            aux = self.piece_angle
        else:
            aux = -self.piece_angle
        self.robot_x = L * np.sin(np.pi + (aux * (np.pi / 180)) - beta)
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
            base_x = -663.823
            base_y = -170.596
            base_z = -167.794
            base_rx = 176.509
            base_ry = -3.206
            base_rz = 2.355
            pitch_x = 0
            pitch_y = -60
            pitch_z = 100
        elif abs(self.robot_angle - 90) < 10:
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
        z = base_z + row * pitch_z
        return [x, y, z, base_rx, base_ry, base_rz]

    def pick_and_place(self, center, angle):
        self.map_camara2robot(center[0], angle)
        x = self.robot_x
        y = self.robot_y
        rz = self.robot_angle
        z_pick = -92.045  # Ajusta esta altura según tu banda
        pick_pose_down = [x, y, z_pick, 0, 0, rz]
        pick_pose_up = [x, y, z_pick + 100, 0, 0, rz]  # Subida segura

        self.robot.move_l_pose(np.array(pick_pose_up), speed=20,acc=20)
        self.robot.wait_until_motion_complete()

        self.robot.move_l_pose(np.array(pick_pose_down), speed=20,acc=20)
        self.robot.wait_until_motion_complete()

        self.helper.cerrar_garra()
        time.sleep(1)

        self.robot.move_l_pose(np.array(pick_pose_up), speed=20,acc=20)
        self.robot.wait_until_motion_complete()

        place_pose = self.mozaic_generator()
        place_pose_up = place_pose.copy()
        place_pose_up[2] += 100  # Altura segura sobre la caja

        self.robot.move_l_pose(np.array(place_pose_up), speed=20,acc=20)
        self.robot.wait_until_motion_complete()

        self.robot.move_l_pose(np.array(place_pose), speed=20, acc=20)
        self.robot.wait_until_motion_complete()

        self.helper.abrir_garra()
        time.sleep(1)

        self.robot.move_l_pose(np.array(place_pose_up), speed=20, acc=20)
        self.robot.wait_until_motion_complete()

        self.robot.move_l_pose(np.array(self.wait_pose), speed=20, acc=20)
        self.robot.wait_until_motion_complete()

        self.piece_num += 1
        return None

    def run(self):
        thread = threading.Thread(target=self.camera_thread, daemon=True)
        thread.start()

        if self.robot.connect():
            print("Successfully connected to robot")
            self.robot.set_user_number(4)
            self.robot.set_current_coord(3)
            self.robot.set_tool_number(2)
            self.robot.move_l_pose(np.array(self.wait_pose))
            self.robot.wait_until_motion_complete()

            while True:
                plc_coil = self.plc.read_coils(0, 1)
                if plc_coil and plc_coil[0]:
                    if self.last_detection_ok:
                        self.pick_and_place(self.last_center, self.last_angle)
                        while self.plc.read_coils(0, 1)[0]:
                            time.sleep(1)
                        self.last_detection_ok = False
                else:
                    print(" No hay datos recientes de detección — no se ejecuta pick and place")
                time.sleep(1)
            else:
                time.sleep(1)
        self.robot.disconnect()

if __name__ == "__main__":
    robot_ip = "169.168.0.200"
    robot = PalletizingRobot(robot_ip)
    robot.initialize_camera()
    robot.run()

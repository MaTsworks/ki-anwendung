import threading
import queue
import cv2
import keyboard
from robomaster import robot
from object_detector import ObjectDetector
import time

class RoboMasterVisionControl:
    def __init__(self):
        self.robot = robot.Robot()
        self.detector = ObjectDetector()
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.camera = None
        self.chassis = None
        self.gimbal = None

    def initialize(self):
        try:
            self.robot.initialize(conn_type="ap", proto_type="tcp")
            self.camera = self.robot.camera
            self.chassis = self.robot.chassis
            self.gimbal = self.robot.gimbal
            self.camera.start_video_stream(display=False)
            print("Robot initialized successfully")
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
        return True

    def camera_capture_thread(self):
        while not self.stop_event.is_set():
            try:
                frame = self.camera.read_cv2_image(strategy="newest", timeout=0.1)
                if frame is not None:
                    if not self.frame_queue.empty():
                        self.frame_queue.get()
                    self.frame_queue.put(frame)
            except Exception as e:
                print(f"Camera capture error: {e}")
                print("Attempting to reconnect camera...")
                try:
                    # Stop and restart video stream
                    self.camera.stop_video_stream()
                    self.camera.start_video_stream(display=False)
                    print("Camera reconnected successfully")
                except Exception as reconnect_error:
                    print(f"Reconnection failed: {reconnect_error}")
                    print("Waiting 2 seconds before retry...")
                    time.sleep(2)

    def run(self):
        if not self.initialize():
            return

        camera_thread = threading.Thread(target=self.camera_capture_thread)
        camera_thread.start()

        try:
            while not self.stop_event.is_set():
                # Robot Movement
                x_speed, y_speed, z_speed = self.get_chassis_input()
                self.chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed)

                # Gimbal/Cannon Control
                pitch, yaw = self.get_gimbal_input()
                self.gimbal.drive_speed(pitch_speed=pitch, yaw_speed=yaw)

                # Camera Processing
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    detected_frame, _ = self.detector.detect(frame)
                    cv2.imshow("RoboMaster S1", detected_frame)
                except queue.Empty:
                    continue

                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop_event.set()

        except Exception as e:
            print(f"Main loop error: {e}")
        finally:
            self.cleanup()

    def get_chassis_input(self):
        x_val, y_val, z_val = 0, 0, 0
        speed = 0.5

        if keyboard.is_pressed('w'):
            x_val = speed
        if keyboard.is_pressed('s'):
            x_val = -speed
        if keyboard.is_pressed('a'):
            y_val = -speed
        if keyboard.is_pressed('d'):
            y_val = speed
        if keyboard.is_pressed('q'):
            z_val = -45
        if keyboard.is_pressed('e'):
            z_val = 45

        return x_val, y_val, z_val

    def get_gimbal_input(self):
        pitch_speed, yaw_speed = 0, 0
        speed = 45

        if keyboard.is_pressed('i'):
            pitch_speed = -speed
        if keyboard.is_pressed('k'):
            pitch_speed = speed
        if keyboard.is_pressed('j'):
            yaw_speed = -speed
        if keyboard.is_pressed('l'):
            yaw_speed = speed

        return pitch_speed, yaw_speed

    def cleanup(self):
        self.stop_event.set()
        self.chassis.drive_speed(x=0, y=0, z=0)
        self.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        self.robot.close()
        cv2.destroyAllWindows()
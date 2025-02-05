import threading
import queue
import cv2
import keyboard
from robomaster import robot
from object_detector import ObjectDetector
import time

from py.control.model_type import ModelType
from py.control.performance_evaluator import PerformanceEvaluator


class RoboMasterVisionControl:
    def __init__(self,model_type: ModelType):
        self.model_type = model_type
        self.robot = robot.Robot()
        self.detector = ObjectDetector(model_type)
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.camera = None
        self.chassis = None
        self.gimbal = None
        self.target_threshold = 0.05
        self.gimbal_speed = 20
        self.target_lock_time = None
        self.lock_duration = 0.5
        self.last_shot_time = 0
        self.shot_cooldown = 0.1
        self.evaluator = PerformanceEvaluator()
        self.evaluator.start_evaluation()

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

    def run(self):
        if not self.initialize():
            return

        camera_thread = threading.Thread(target=self.camera_capture_thread)
        camera_thread.start()

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    start_time = time.time()
                    detected_frame, target_info = self.detector.detect(frame)
                    detection_time = time.time() - start_time

                    self.evaluator.log_detection(detection_time, target_info)

                    if self.model_type == ModelType.PISTOL and target_info:
                        yaw_speed = self.gimbal_speed * target_info['x_offset']
                        pitch_speed = -self.gimbal_speed * target_info['y_offset']

                        self.evaluator.log_tracking(target_info)

                        self.track_and_shoot(target_info)
                    else:
                        x_speed, y_speed, z_speed = self.get_chassis_input()
                        self.chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed)

                        pitch, yaw = self.get_gimbal_input()
                        self.gimbal.drive_speed(pitch_speed=pitch, yaw_speed=yaw)

                        if target_info:
                            self.evaluator.log_tracking(target_info)

                    cv2.imshow("RoboMaster S1", detected_frame)
                except queue.Empty:
                    continue

                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop_event.set()

        except Exception as e:
            print(f"Main loop error: {e}")
        finally:
            self.cleanup()

    def track_and_shoot(self, target_info):
        # Calculate gimbal movements
        yaw_speed = self.gimbal_speed * target_info['x_offset']
        pitch_speed = -self.gimbal_speed * target_info['y_offset']

        current_time = time.time()

        # Check if we're on target
        if abs(target_info['x_offset']) < self.target_threshold and \
                abs(target_info['y_offset']) < self.target_threshold:

            # Stop gimbal movement when on target
            self.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

            # Initialize target lock time if not set
            if self.target_lock_time is None:
                self.target_lock_time = current_time

            # Check if we've been locked long enough and cooldown has elapsed
            elif current_time - self.target_lock_time >= self.lock_duration and \
                    current_time - self.last_shot_time >= self.shot_cooldown:
                self.robot.blaster.fire()
                self.last_shot_time = current_time

        else:
            # Reset lock time if we're not on target
            self.target_lock_time = None
            # Move gimbal to track target
            self.gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=yaw_speed)

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
        self.evaluator.save_report()
        cv2.destroyAllWindows()
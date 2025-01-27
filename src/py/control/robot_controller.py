import keyboard

class RobotController:
    def __init__(self):
        self.speed = 0.5
        self.max_speed = 2.0
        self.min_speed = 0.1
        self.speed_increment = 0.1
        self.rotation_speed = 45

    def get_input(self):
        x_val, y_val, z_val = 0, 0, 0

        if keyboard.is_pressed('w'):
            x_val = self.speed
        if keyboard.is_pressed('s'):
            x_val = -self.speed
        if keyboard.is_pressed('a'):
            y_val = -self.speed
        if keyboard.is_pressed('d'):
            y_val = self.speed
        if keyboard.is_pressed('q'):
            z_val = -self.rotation_speed
        if keyboard.is_pressed('e'):
            z_val = self.rotation_speed

        return x_val, y_val, z_val

    def adjust_speed(self):
        if keyboard.is_pressed('up'):
            self.speed = min(self.speed + self.speed_increment, self.max_speed)
        if keyboard.is_pressed('down'):
            self.speed = max(self.speed - self.speed_increment, self.min_speed)
        return self.speed
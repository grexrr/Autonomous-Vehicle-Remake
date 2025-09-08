#Front-wheel Steering Model (Kinematic Bicycle Model)
import numpy as np
import utilities.WrapAngle as wa

class Car:
    def __init__(self, x:float, y:float, yaw:float) -> None:
        self.x = x             # [m]
        self.y = y             # [m]
        self.yaw = yaw         # [rad], [-pi, pi]
        
        self.velocity = 0.0    # [m/s], [MIN_SPEED, MAX_SPEED]
        self.MIN_SPEED = -30 / 3.6
        self.MAX_SPEED = 55.0 / 3.6
        
        self.steer = 0.0       # δ
        self.MAX_STEER = np.deg2rad(40.0)  # [rad]

        self.WHEEL_BASE = 2.5  # [m]

        self.MAX_ACCEL = 15.0 
        

    def update(self, dt, do_warp_angle: bool = True) -> None:
        self.x += self.velocity * np.cos(self.yaw) * dt
        self.y += self.velocity * np.sin(self.yaw) * dt
        self.yaw += self.velocity / self.WHEEL_BASE * np.tan(self.steer) * dt
        if do_warp_angle:
            self.yaw = wa.wrap_to_pi(self.yaw)

    def update_with_control(self, target_velocity, target_steer, dt, do_warp_angle: bool = True):
        # self.update(dt, do_warp_angle=do_warp_angle)
        
        target_velocity = np.clip(target_velocity, self.MIN_SPEED, self.MAX_SPEED)
        target_steer = np.clip(target_steer, -self.MAX_STEER, self.MAX_STEER)
        
        dv_max = self.MAX_ACCEL * dt
        self.velocity += np.clip(target_velocity - self.velocity, -dv_max, dv_max)


        dsteer_max = self.MAX_STEER_SPEED * dt
        self.steer += np.clip(target_steer - self.steer, -dsteer_max, dsteer_max)

        self.update(dt, do_wrap_angle=do_warp_angle)
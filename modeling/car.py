#Front-wheel Steering Model (Kinematic Bicycle Model)
import numpy as np
from utils.wrap_angle import wrap_angle

class Car:

    LENGTH, WIDTH = 4.5, 2.0
    BACK_TO_WHEEL = 1.0
    BACK_TO_CENTER = LENGTH/2 - BACK_TO_WHEEL

    WHEEL_BASE = 2.5              # [m]
    MAX_STEER = np.deg2rad(40.0)  # [rad]
    MAX_STEER_SPEED = np.deg2rad(360.0)  # [rad/s]
    MIN_SPEED = -30 / 3.6
    MAX_SPEED = 55.0 / 3.6
    MAX_ACCEL = 15.0 

    COLLISION_LENGTH = LENGTH + 2.0
    COLLISION_WIDTH = WIDTH + 2.0
    COLLISION_RADIUS = float(np.hypot(COLLISION_LENGTH/2, COLLISION_WIDTH/2))

    AX_STEER = np.deg2rad(40.0)  # [rad]
    TARGET_MAX_STEER = np.deg2rad(35.0)  # [rad], for global planner
    
    MAX_CENTRIPETAL_ACCEL = 16.0  # [m/ss]

    TARGET_SPEED = 40.0 / 3.6  # [m/s]

    TARGET_MIN_TURNING_RADIUS = WHEEL_BASE / np.tan(TARGET_MAX_STEER)  # [m], for global planner

    SCAN_RADIUS = 15.0  # [m]

    
    def __init__(self, x:float, y:float, yaw:float=0.0,velocity: float=0.0, steer:float=0.0) -> None:
        self.x = x                # [m]
        self.y = y                # [m]
        self.yaw = yaw            # [rad], [-pi, pi]
        self.velocity = velocity  # [m/s], [MIN_SPEED, MAX_SPEED]
        self.steer = steer          # δ

    def update(self, dt, do_wrap_angle: bool = True) -> None:
        self.x += self.velocity * np.cos(self.yaw) * dt
        self.y += self.velocity * np.sin(self.yaw) * dt
        self.yaw += self.velocity / self.WHEEL_BASE * np.tan(self.steer) * dt
        if do_wrap_angle:
            self.yaw = wrap_angle(self.yaw)

    def update_with_control(self, target_velocity, target_steer, dt, do_wrap_angle: bool = True):
        # self.update(dt, do_warp_angle=do_warp_angle)
        self.update(dt, do_wrap_angle=do_wrap_angle)
        
        target_velocity = np.clip(target_velocity, self.MIN_SPEED, self.MAX_SPEED)
        target_steer = np.clip(target_steer, -self.MAX_STEER, self.MAX_STEER)
        
        dv_max = self.MAX_ACCEL * dt
        dsteer_max = self.MAX_STEER_SPEED * dt
        
        self.velocity += np.clip(target_velocity - self.velocity, -dv_max, dv_max)
        self.steer += np.clip(target_steer - self.steer, -dsteer_max, dsteer_max)

    def align_yaw(self, target_yaw: float) -> None:
        self.yaw = target_yaw + wrap_angle(self.yaw - target_yaw) 

    def check_collision(self, obstacles) -> bool:
        pts = obstacles.coordinates
        if pts.size == 0:
            return False

        cos, sin = np.cos(self.yaw), np.sin(self.yaw)
        cx = self.x + self.BACK_TO_CENTER * cos
        cy = self.y + self.BACK_TO_CENTER * sin
        ids = obstacles.kd_tree.query_ball_point([cx, cy], self.COLLISION_RADIUS)
        
        candidates = obstacles.coordinates[ids]
        candidates = (candidates - [cx, cy]) @ np.array([[cos, sin], [-sin, cos]])

        return np.any(
            np.logical_and(
                np.abs(candidates[:, 0]) < self.COLLISION_LENGTH / 2,
                np.abs(candidates[:, 1]) < self.COLLISION_WIDTH / 2,
            )
        )
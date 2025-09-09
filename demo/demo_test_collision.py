# demo_world_car.py
import numpy as np
import matplotlib.pyplot as plt

from modeling.obstacles import Obstacles
from modeling.car import Car

def compute_inflation_radius(resolution: float) -> float:
    """
    用“半宽 + 半个格子余量”的安全半径(工程上更好调)。
    你也可以换成: min(COLLISION_LENGTH, COLLISION_WIDTH)/2(如果你在 Car 里有这两个)。
    """
    return Car.WIDTH / 2 + 0.5 * resolution

def make_test_obstacles() -> Obstacles:
    pts = []
    # 四周 60x60 围墙(红点是采样用来做KDTree/可视化)
    for x in np.linspace(0, 60, 121):
        pts += [(x, 0), (x, 60)]
    for y in np.linspace(0, 60, 121):
        pts += [(0, y), (60, y)]
    # 两条竖直“栏杆”
    for y in np.linspace(10, 50, 81):
        pts.append((20, y))
    for y in np.linspace(10, 50, 81):
        pts.append((40, y))
    return Obstacles(np.array(pts, dtype=float))

def car_outline(car: Car) -> np.ndarray:
    """返回车身矩形轮廓在世界坐标系下的五个点(闭合)。"""
    L, W = car.LENGTH, car.WIDTH
    cx = car.x + car.BACK_TO_CENTER * np.cos(car.yaw)
    cy = car.y + car.BACK_TO_CENTER * np.sin(car.yaw)
    box = np.array([[-L/2, -W/2], [-L/2, W/2], [L/2, W/2], [L/2, -W/2], [-L/2, -W/2]])
    c, s = np.cos(car.yaw), np.sin(car.yaw)
    R = np.array([[c, -s], [s, c]])
    return (box @ R.T) + np.array([cx, cy])

def main():
    # 1) 障碍 & 栅格
    obs = make_test_obstacles()
    RES = 2.0
    r = compute_inflation_radius(RES)
    grid = obs.downsampling_to_grid(RES, r)

    # 2) 画占据栅格(黑=障碍) + 障碍点(红)
    fig, ax = plt.subplots()
    ax.imshow(
        grid.grid, origin="lower", cmap="gray_r",
        extent=[grid.minx, grid.maxx, grid.miny, grid.maxy], alpha=0.6, zorder=0
    )
    ax.plot(*obs.coordinates.T, ".r", ms=2, zorder=1)

    # 3) 车初始化 & 动画元素
    car = Car(5.0, 5.0, 0.0)
    car.velocity = 3.0  # m/s
    car.steer = np.deg2rad(15)  # 固定15°看看圆弧
    xs, ys = [], []
    (trace_line,) = ax.plot([], [], "-b", lw=2, label="trace")
    (car_poly,)  = ax.plot([], [], "-g", lw=2, label="car")
    ax.set_aspect("equal", "box")
    ax.set_xlim(grid.minx, grid.maxx); ax.set_ylim(grid.miny, grid.maxy)
    ax.legend(); plt.ion()

    # 4) 动画循环：更新状态→检查碰撞→更新可视化
    dt = 0.1
    for _ in range(800):
        collided = car.check_collision(obs)  # 注意：Car.check_collision里inside要用 & 连接两个条件
        color = "r" if collided else "g"

        xs.append(car.x); ys.append(car.y)
        trace_line.set_data(xs, ys)

        poly = car_outline(car)
        car_poly.set_data(poly[:, 0], poly[:, 1])
        car_poly.set_color(color)

        if collided:
            ax.set_title("Collision!", color="r")
            plt.pause(0.5)
            break

        car.update(dt)
        plt.pause(0.02)

    plt.ioff(); plt.show()

if __name__ == "__main__":
    main()

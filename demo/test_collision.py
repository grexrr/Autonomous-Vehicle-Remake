# demo_world_car.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime

from modeling.obstacles import Obstacles
from modeling.car import Car

def compute_inflation_radius(resolution: float) -> float:
    """
    用"半宽 + 半个格子余量"的安全半径(工程上更好调)。
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
    # 两条竖直"栏杆"
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

def main(save_gif=True):
    # 1) 障碍 & 栅格
    obs = make_test_obstacles()
    RES = 2.0
    r = compute_inflation_radius(RES)
    grid = obs.downsampling_to_grid(RES, r)

    # 2) 画占据栅格(黑=障碍) + 障碍点(红)
    fig, ax = plt.subplots(figsize=(10, 8))
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
    ax.legend()
    ax.set_title("Autonomous Vehicle Collision Detection Demo")
    
    # 存储动画帧数据
    frames_data = []
    dt = 0.1
    max_frames = 800
    
    # 预计算所有帧数据
    print("正在预计算动画帧...")
    for frame in range(max_frames):
        collided = car.check_collision(obs)
        color = "r" if collided else "g"
        
        xs.append(car.x)
        ys.append(car.y)
        poly = car_outline(car)
        
        frames_data.append({
            'xs': xs.copy(),
            'ys': ys.copy(), 
            'poly': poly,
            'color': color,
            'collided': collided
        })
        
        if collided:
            print(f"碰撞发生在第 {frame} 帧")
            break
            
        car.update(dt)
    
    # 动画更新函数
    def animate(frame):
        if frame >= len(frames_data):
            return trace_line, car_poly
            
        data = frames_data[frame]
        
        # 更新轨迹
        trace_line.set_data(data['xs'], data['ys'])
        
        # 更新车辆
        car_poly.set_data(data['poly'][:, 0], data['poly'][:, 1])
        car_poly.set_color(data['color'])
        
        # 更新标题
        if data['collided']:
            ax.set_title("Collision Detected!", color="r", fontsize=14)
        else:
            ax.set_title(f"Autonomous Vehicle Demo - Frame {frame}", fontsize=14)
        
        return trace_line, car_poly
    
    if save_gif:
        # 创建images目录(如果不存在)
        os.makedirs("images", exist_ok=True)
        
        # 生成GIF文件名(带时间戳)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_filename = f"images/collision_demo.gif"
        
        print(f"正在生成GIF动画: {gif_filename}")
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames_data),
            interval=50, blit=False, repeat=True
        )
        
        # 保存为GIF
        anim.save(gif_filename, writer='pillow', fps=20)
        print(f"GIF已保存到: {gif_filename}")
        
        # 显示动画
        plt.show()
    else:
        # 原始实时动画模式
        plt.ion()
        for frame in range(len(frames_data)):
            data = frames_data[frame]
            
            trace_line.set_data(data['xs'], data['ys'])
            car_poly.set_data(data['poly'][:, 0], data['poly'][:, 1])
            car_poly.set_color(data['color'])
            
            if data['collided']:
                ax.set_title("Collision!", color="r")
                plt.pause(0.5)
                break
            
            plt.pause(0.02)
        
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    # 设置为True来保存GIF，False来显示实时动画
    main(save_gif=True)

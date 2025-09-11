# scripts/test_hybrid_astar.py
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime

# === 按你的项目结构调整这三行 ===
from modeling.obstacles import Obstacles
from modeling.car import Car
import global_planner.hybrid_a_star as hb

# 可选: 统一随机数种子，便于复现
np.random.seed(0)

# ---------- 1) 造障碍 ----------
def make_test_obstacles():
    """60x60 围墙 + 两根栏杆(在 x=20 和 x=40, y∈[10,50])"""
    pts = []
    W, H = 60.0, 60.0

    # 四周围墙(稠一些可视化更清晰)
    for x in np.linspace(0, W, int(W*2)+1):
        pts += [(x, 0.0), (x, H)]
    for y in np.linspace(0, H, int(H*2)+1):
        pts += [(0.0, y), (W, y)]

    # 两根竖直栏杆
    for y in np.linspace(10, 50, 81):
        pts.append((20.0, y))
        pts.append((40.0, y))

    # 可选: 添加少量随机散点障碍(注释掉即可关闭)
    # for _ in range(40):
    #     pts.append((np.random.uniform(5, 55), np.random.uniform(5, 55)))

    return Obstacles(np.array(pts, dtype=float))

# ---------- 2) 画车身矩形 ----------
def car_corners(x, y, yaw, L=Car.LENGTH, W=Car.WIDTH, back_to_center=Car.BACK_TO_CENTER):
    """返回车身矩形四角(闭合)，世界坐标"""
    cx = x + back_to_center * math.cos(yaw)
    cy = y + back_to_center * math.sin(yaw)
    box = np.array([[-L/2, -W/2],
                    [-L/2,  W/2],
                    [ L/2,  W/2],
                    [ L/2, -W/2],
                    [-L/2, -W/2]], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],[s, c]])
    return (box @ R.T) + np.array([cx, cy])

# ---------- 3) 碰撞复核 ----------
def any_collision_along(path_xyyaw, obstacles):
    """用 Car.check_collision 对结果做一次逐点复核"""
    for x, y, yaw in path_xyyaw[:, :3]:
        if Car(x, y, yaw).check_collision(obstacles):
            return True
    return False

# ---------- 4) 动画版本的主测试 ----------
def run_case_animated(start, goal, obstacles, title="Hybrid A* demo", save_gif=True, case_name="demo"):
    """
    start, goal: [x, y, yaw](yaw单位: 弧度)
    """
    print(f"[CASE] start={start}, goal={goal}")

    path = hb.hybrid_a_star(np.array(start, float), np.array(goal, float), obstacles)

    if path is None:
        print("✗ 未找到路径(可能参数需要微调: 转角离散/段长/步长/安全半径)")
        return

    # 统计信息
    N = len(path)
    fwd_len = np.sum(np.hypot(np.diff(path[:,0]), np.diff(path[:,1]))[path[1:,3] > 0])
    bwd_len = np.sum(np.hypot(np.diff(path[:,0]), np.diff(path[:,1]))[path[1:,3] < 0])
    print(f"✓ 路径点数={N}, 前进长度≈{fwd_len:.2f} m, 倒车长度≈{bwd_len:.2f} m")

    # 碰撞复核
    collided = any_collision_along(path, obstacles)
    print("碰撞复核: ", "✗ 有碰撞" if collided else "✓ 无碰撞")

    # ---------- 动画设置 ----------
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', 'box')
    ax.set_title(title, fontsize=14)

    # 障碍(灰色)
    ax.scatter(obstacles.coordinates[:,0], obstacles.coordinates[:,1], 
               s=8, c='#888888', alpha=0.7, label='obstacles')

    # 起终点
    ax.plot(start[0], start[1], 'go', ms=12, label='start', markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(goal[0], goal[1], 'rx', ms=12, mew=3, label='goal')

    # 完整路径(淡色)
    path_x, path_y = path[:, 0], path[:, 1]
    ax.plot(path_x, path_y, '-', lw=1, c='lightblue', alpha=0.5, label='planned path')

    # 动画元素
    (trace_line,) = ax.plot([], [], '-b', lw=3, label='vehicle trace')
    (car_poly,) = ax.plot([], [], '-g', lw=2, label='vehicle')
    (direction_arrow,) = ax.plot([], [], '->', c='red', lw=2, ms=8, label='direction')

    ax.set_xlim(-2, 62)
    ax.set_ylim(-2, 62)
    ax.grid(True, ls='--', alpha=0.3)
    ax.legend(loc='upper left')

    # 存储动画数据
    frames_data = []
    dt = 0.1
    total_time = 0.0
    
    # 预计算动画帧
    print("正在预计算动画帧...")
    for i in range(N):
        # 计算当前位置
        current_pos = path[i]
        x, y, yaw, direction = current_pos
        
        # 计算已走过的路径
        trace_x = path_x[:i+1]
        trace_y = path_y[:i+1]
        
        # 计算车身轮廓
        poly = car_corners(x, y, yaw)
        
        # 计算方向箭头(从车中心指向前方)
        arrow_length = 2.0
        arrow_x = [x, x + arrow_length * math.cos(yaw)]
        arrow_y = [y, y + arrow_length * math.sin(yaw)]
        
        # 计算进度信息
        progress = (i + 1) / N * 100
        
        frames_data.append({
            'trace_x': trace_x,
            'trace_y': trace_y,
            'poly': poly,
            'arrow_x': arrow_x,
            'arrow_y': arrow_y,
            'progress': progress,
            'direction': direction,
            'total_time': total_time
        })
        
        # 更新时间(基于路径段长度)
        if i < N - 1:
            segment_length = math.hypot(path[i+1, 0] - path[i, 0], path[i+1, 1] - path[i, 1])
            total_time += segment_length / 3.0  # 假设平均速度3m/s

    # 动画更新函数
    def animate(frame):
        if frame >= len(frames_data):
            return trace_line, car_poly, direction_arrow
            
        data = frames_data[frame]
        
        # 更新轨迹
        trace_line.set_data(data['trace_x'], data['trace_y'])
        
        # 更新车辆
        car_poly.set_data(data['poly'][:, 0], data['poly'][:, 1])
        
        # 更新方向箭头
        direction_arrow.set_data(data['arrow_x'], data['arrow_y'])
        
        # 更新标题
        direction_text = "Forward" if data['direction'] >= 0 else "Reverse"
        ax.set_title(f"{title}\nProgress: {data['progress']:.1f}% | {direction_text} | Time: {data['total_time']:.1f}s", 
                    fontsize=12)
        
        return trace_line, car_poly, direction_arrow

    if save_gif:
        # 创建images目录
        os.makedirs("images", exist_ok=True)
        
        # 生成GIF文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_filename = f"images/hybrid_astar_{case_name}_{timestamp}.gif"
        
        print(f"正在生成GIF动画: {gif_filename}")
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames_data),
            interval=100, blit=False, repeat=True
        )
        
        # 保存为GIF
        anim.save(gif_filename, writer='pillow', fps=10)
        print(f"GIF已保存到: {gif_filename}")
        
        # 显示动画
        plt.show()
    else:
        # 实时动画模式
        plt.ion()
        for frame in range(len(frames_data)):
            data = frames_data[frame]
            
            trace_line.set_data(data['trace_x'], data['trace_y'])
            car_poly.set_data(data['poly'][:, 0], data['poly'][:, 1])
            direction_arrow.set_data(data['arrow_x'], data['arrow_y'])
            
            direction_text = "Forward" if data['direction'] >= 0 else "Reverse"
            ax.set_title(f"{title}\nProgress: {data['progress']:.1f}% | {direction_text}", fontsize=12)
            
            plt.pause(0.1)
        
        plt.ioff()
        plt.show()

# ---------- 5) 原始静态版本(保留) ----------
def run_case_static(start, goal, obstacles, title="Hybrid A* demo"):
    """
    原始的静态绘图版本
    """
    print(f"[CASE] start={start}, goal={goal}")

    path = hb.hybrid_a_star(np.array(start, float), np.array(goal, float), obstacles)

    if path is None:
        print("✗ 未找到路径(可能参数需要微调: 转角离散/段长/步长/安全半径)")
        return

    # 统计信息
    N = len(path)
    fwd_len = np.sum(np.hypot(np.diff(path[:,0]), np.diff(path[:,1]))[path[1:,3] > 0])
    bwd_len = np.sum(np.hypot(np.diff(path[:,0]), np.diff(path[:,1]))[path[1:,3] < 0])
    print(f"✓ 路径点数={N}, 前进长度≈{fwd_len:.2f} m, 倒车长度≈{bwd_len:.2f} m")

    # 碰撞复核
    collided = any_collision_along(path, obstacles)
    print("碰撞复核: ", "✗ 有碰撞" if collided else "✓ 无碰撞")

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal', 'box')
    ax.set_title(title)

    # 障碍(灰色)
    ax.scatter(obstacles.coordinates[:,0], obstacles.coordinates[:,1], s=8, c='#888888', alpha=0.7, label='obstacles')

    # 起终点
    ax.plot(start[0], start[1], 'go', ms=8, label='start')
    ax.plot(goal[0], goal[1], 'rx', ms=10, mew=2, label='goal')

    # 路径: 按方向上色
    # 前进蓝色，倒车橙色
    for i in range(1, N):
        x0,y0,_, d0 = path[i-1]
        x1,y1,_, d1 = path[i]
        color = '#1f77b4' if d1 >= 0 else '#ff7f0e'
        ax.plot([x0,x1],[y0,y1], '-', lw=2, c=color)

    # 每隔若干步画一个车身盒子
    step = max(1, N // 20)
    for i in range(0, N, step):
        poly = car_corners(*path[i,:3])
        ax.plot(poly[:,0], poly[:,1], '-k', lw=1)

    ax.set_xlim(-2, 62); ax.set_ylim(-2, 62)
    ax.grid(True, ls='--', alpha=0.3)
    ax.legend(loc='upper left')
    plt.show()

def main(animated=True, save_gif=True):
    obstacles = make_test_obstacles()

    if animated:
        print("=== 生成动画版本 ===")
        
        # 场景1: 左下 -> 右上，朝向均为0°
        start = [5.0, 5.0, 0.0]
        goal  = [55.0, 55.0, 0.0]
        run_case_animated(start, goal, obstacles, 
                         title="Hybrid A* — Diagonal Path", 
                         save_gif=save_gif, case_name="diagonal")

        # 场景2: 左下 -> 右上，目标朝向90°(考验终端对齐)
        start2 = [5.0, 5.0, 0.0]
        goal2  = [55.0, 55.0, math.radians(90)]
        run_case_animated(start2, goal2, obstacles, 
                         title="Hybrid A* — Diagonal Path, Goal Yaw=90°", 
                         save_gif=save_gif, case_name="diagonal_90")

        # 场景3: 通道穿越(起点在两栏杆之间)
        start3 = [30.0, 8.0, math.radians(90)]
        goal3  = [30.0, 52.0, math.radians(90)]
        run_case_animated(start3, goal3, obstacles, 
                         title="Hybrid A* — Corridor Navigation", 
                         save_gif=save_gif, case_name="corridor")
    else:
        print("=== 生成静态版本 ===")
        
        # 场景1: 左下 -> 右上，朝向均为0°
        start = [5.0, 5.0, 0.0]
        goal  = [55.0, 55.0, 0.0]
        run_case_static(start, goal, obstacles, title="Hybrid A* — diagonal")

        # 场景2: 左下 -> 右上，目标朝向90°(考验终端对齐)
        start2 = [5.0, 5.0, 0.0]
        goal2  = [55.0, 55.0, math.radians(90)]
        run_case_static(start2, goal2, obstacles, title="Hybrid A* — diagonal, goal yaw=90°")

        # 场景3: 通道穿越(起点在两栏杆之间)
        start3 = [30.0, 8.0, math.radians(90)]
        goal3  = [30.0, 52.0, math.radians(90)]
        run_case_static(start3, goal3, obstacles, title="Hybrid A* — corridor")

if __name__ == "__main__":
    # 设置为True生成动画GIF，False显示静态图
    main(animated=True, save_gif=True)

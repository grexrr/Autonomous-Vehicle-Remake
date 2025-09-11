# demo/run_mpc.py
import math
import numpy as np
import matplotlib.pyplot as plt

from modeling.car import Car
from modeling.obstacles import Obstacles
from global_planner.hybrid_a_star import hybrid_a_star
from local_planner.model_predictive_control import ModelPredictiveControl as MPC
from constants import LOCAL_PLANNER_DELTA_TIME, MAP_WIDTH, MAP_HEIGHT

# --- 兼容：若 Car 没 copy()，打个猴子补丁 ---
if not hasattr(Car, "copy"):
    def _car_copy(self):
        return Car(self.x, self.y, self.yaw, self.velocity, self.steer)
    Car.copy = _car_copy  # type: ignore

# ---------- 可视化：画车矩形 ----------
def draw_car(ax, car: Car):
    L, W = car.LENGTH, car.WIDTH
    c, s = np.cos(car.yaw), np.sin(car.yaw)
    cx = car.x + car.BACK_TO_CENTER * c
    cy = car.y + car.BACK_TO_CENTER * s
    box = np.array([[-L/2,-W/2],[-L/2,W/2],[L/2,W/2],[L/2,-W/2],[-L/2,-W/2]])
    R = np.array([[c,-s],[s,c]])
    pts = (box @ R.T) + np.array([cx, cy])
    (line,) = ax.plot(pts[:,0], pts[:,1], '-k', lw=2)
    (head,) = ax.plot([car.x], [car.y], 'or', ms=4)
    return [line, head]

# ---------- 场景：四周围墙 + 两根栏杆 ----------
def make_test_obstacles():
    pts = []
    for x in np.linspace(0, 60, 121):
        pts += [(x, 0), (x, 60)]
    for y in np.linspace(0, 60, 121):
        pts += [(0, y), (60, y)]
    for y in np.linspace(10, 50, 81):
        pts.append((20, y))
    for y in np.linspace(10, 50, 81):
        pts.append((40, y))
    return Obstacles(np.array(pts, float))

# ---------- 最近点误差 ----------
def nearest_error(ref_xyyaw: np.ndarray, x: float, y: float, yaw: float):
    dx = ref_xyyaw[:,0] - x
    dy = ref_xyyaw[:,1] - y
    idx = int(np.argmin(dx*dx + dy*dy))
    ex = ref_xyyaw[idx,0] - x
    ey = ref_xyyaw[idx,1] - y
    ref_yaw = ref_xyyaw[idx,2]
    cross_track = -math.sin(ref_yaw)*ex + math.cos(ref_yaw)*ey
    head_err = (yaw - ref_yaw + np.pi) % (2*np.pi) - np.pi
    return cross_track, head_err, idx

# ---------- 兜底：把 dir 列强制修成 ±1 ----------
def _ensure_dir_sign_local(path_xyyawdir: np.ndarray) -> np.ndarray:
    assert path_xyyawdir.ndim == 2 and path_xyyawdir.shape[1] in (3,4)
    if path_xyyawdir.shape[1] == 3:
        path_xyyawdir = np.hstack([path_xyyawdir, np.ones((path_xyyawdir.shape[0],1), float)])
    traj = path_xyyawdir.copy()
    dirc = traj[:,3].astype(float)
    N = traj.shape[0]
    for k in range(1, N):
        if dirc[k] == 0:
            dx = traj[k,0] - traj[k-1,0]
            dy = traj[k,1] - traj[k-1,1]
            yaw = traj[k,2]
            proj = dx*np.cos(yaw) + dy*np.sin(yaw)
            if proj > 1e-6: dirc[k] = 1.0
            elif proj < -1e-6: dirc[k] = -1.0
            else: dirc[k] = dirc[k-1] if dirc[k-1] != 0 else 1.0
    if dirc[0] == 0:
        nz = np.flatnonzero(dirc)
        dirc[0] = dirc[nz[0]] if nz.size else 1.0
    traj[:,3] = np.sign(dirc)
    return traj

# ---------- 跑一个用例 ----------
def run_case(start, goal, obstacles):
    print(f"[CASE] start={start.tolist()}, goal={goal.tolist()}")

    # 1) 全局路径：Hybrid A*
    global_path = hybrid_a_star(np.array(start, float), np.array(goal, float), obstacles)
    if global_path is None or len(global_path) < 2:
        print("✗ 全局路径失败"); return

    # dir 列兜底 → ±1
    global_path = _ensure_dir_sign_local(global_path)
    # print("dir unique:", np.unique(global_path[:,3]))

    # 2) 构造 MPC（你的复制版构造器吃完整路径）
    mpc = MPC(global_path)

    # 3) 仿真闭环
    car = Car(start[0], start[1], start[2], velocity=0.0, steer=0.0)

    xs, ys = [car.x], [car.y]
    vs, deltas = [car.velocity], [car.steer]
    cte_list, head_list = [], []

    T_max = 60.0
    steps = int(T_max / LOCAL_PLANNER_DELTA_TIME)

    plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_aspect('equal','box')
    ax.set_xlim(-2, MAP_WIDTH+2); ax.set_ylim(-2, MAP_HEIGHT+2)
    ax.plot(global_path[:,0], global_path[:,1], '--', lw=1, label='global path')
    ax.plot(obstacles.coordinates[:,0], obstacles.coordinates[:,1], 'r.', ms=2, alpha=0.4, label='obstacles')
    (traj_line,) = ax.plot([], [], 'b-', lw=2, label='mpc trace')
    car_art = None
    ax.legend()

    reached = False
    for k in range(steps):
        # 取控制（与你复制来的 MPC 接口对齐）
        res = mpc.update(car, LOCAL_PLANNER_DELTA_TIME)  # -> MPCResult
        a, delta = float(res.controls[0, 0]), float(res.controls[0, 1])

        # 应用到车模型
        target_v = car.velocity + a * LOCAL_PLANNER_DELTA_TIME
        # 限幅保护
        target_v = float(np.clip(target_v, Car.MIN_SPEED, Car.MAX_SPEED))
        delta = float(np.clip(delta, -Car.MAX_STEER, Car.MAX_STEER))
        car.update_with_control(target_v, delta, LOCAL_PLANNER_DELTA_TIME, do_wrap_angle=True)

        # 记录 & 误差
        xs.append(car.x); ys.append(car.y)
        vs.append(car.velocity); deltas.append(car.steer)
        cte, head_err, _ = nearest_error(global_path[:,:3], car.x, car.y, car.yaw)
        cte_list.append(cte); head_list.append(head_err)

        if car.check_collision(obstacles):
            print("✗ 仿真中发生碰撞，提前结束"); break

        if np.hypot(car.x-goal[0], car.y-goal[1]) < 0.8 and abs(head_err) < np.deg2rad(12) and abs(car.velocity) < 0.5:
            reached = True; break

        if k % 2 == 0:
            traj_line.set_data(xs, ys)
            if car_art:
                for aobj in car_art: aobj.remove()
            car_art = draw_car(ax, car)
            plt.pause(0.001)

    traj_line.set_data(xs, ys)
    if car_art:
        for aobj in car_art: aobj.remove()
    draw_car(ax, car)
    plt.title("MPC Tracking")
    plt.tight_layout()
    plt.show()

    # 指标
    fwd_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    cte_rmse = float(np.sqrt(np.mean(np.square(cte_list)))) if cte_list else float("nan")
    head_rmse = float(np.sqrt(np.mean(np.square(head_list)))) if head_list else float("nan")
    print(f"✓ 轨迹点数={len(xs)}, 轨迹总长≈{fwd_len:.2f} m")
    print(f"跟踪误差：横向 RMSE={cte_rmse:.3f} m, 航向 RMSE={np.rad2deg(head_rmse):.2f} deg")
    print("到达目标：" + ("✓" if reached else "✗"))

def main():
    # 依赖提醒（只提示，不强制）
    try:
        import cvxpy  # noqa: F401
    except Exception:
        print("⚠️ 需要安装 cvxpy 以及一个 QP 求解器（OSQP 或 Clarabel）：")
        print("   pip install cvxpy osqp   或   pip install cvxpy clarabel")

    obstacles = make_test_obstacles()
    cases = [
        (np.array([5.0, 5.0, 0.0]),           np.array([55.0, 55.0, 0.0])),
        (np.array([5.0, 5.0, 0.0]),           np.array([55.0, 55.0, np.pi/2])),
        (np.array([30.0, 8.0, np.pi/2]),      np.array([30.0, 52.0, np.pi/2])),
    ]
    for s, g in cases:
        run_case(s, g, obstacles)

if __name__ == "__main__":
    main()

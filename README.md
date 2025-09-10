# Autonomous Vehicle 

This project is a learning project for an autonomous driving path planning system based on the **Hybrid A* Algorithm**. The original project was inspired by FredBill1's open-source implementation: https://github.com/FredBill1/AutonomousDrivingDemo.git

**Learning Background:**
- Initially participated as a group project, mainly responsible for the global planner part
- To deeply understand the algorithm principles and implementation details, I am now independently redoing the entire project
- Focus on learning the application of the Hybrid A* Algorithm in vehicle motion planning with non-holonomic constraints

## Core Algorithm

In autonomous driving systems, path planning is typically divided into two levels: **Global Planner** and **Local Planner**. These two levels work together, each with its own responsibilities, to achieve safe and efficient vehicle autonomous navigation.

- Global Planner (Hybrid A*) provides a "big road to follow"—a series of discrete $[x, y, \psi]$ points that can avoid obstacles but do not consider constraints like motor/tires/steering speed/acceleration or timing (such as how much to steer or accelerate every 70ms in a real car).

- Local Planner (MPC) checks the current position, orientation, and speed of the car at regular intervals (your LOCAL_PLANNER_DELTA_TIME); it then extracts a "reference segment" (prediction domain) from the global reference for the next few seconds, "tries" various future sequences of throttle and steering actions on paper, and selects the best plan (with the smallest error, smoothest actions, and within physical limits)—only executing the first action and recalculating in the next step. This is called receding horizon optimization.

### Global Planner
The path planning system is based on the **Hybrid A* Algorithm**. The core idea of this algorithm is to use the evaluation function **f = g + h** to guide the search process:

- **g(n)**: The actual cost from the start to the current node n
  - Includes path length, steering angle changes, direction switches, and other motion costs
  - Ensures that the found path is kinematically feasible and has the minimum cost

- **h(n)**: The heuristic estimated cost from the current node n to the goal
  - Uses the Dijkstra distance field to provide acceptable heuristic information
  - Guides the search towards the goal direction, improving algorithm efficiency

- **f(n) = g(n) + h(n)**: Total Evaluation Function
  - Balances path quality and search efficiency
  - Prioritizes expanding nodes that are most promising to reach the goal

The algorithm generates candidate path segments through **Motion Primitives**, combined with **collision detection** and **cost evaluation**, to find the optimal path in continuous state space. This method is particularly suitable for handling vehicle motion planning problems with non-holonomic constraints.

The pseudo code is as follows：

```psudo code
# Preprocessing: Obtain the occupancy grid and Dijkstra distance field D
grid = obstacles.downsampling_to_grid(...)
D = _distance_field(grid, goal_xy)

# Initialize the start node (empty segment/zero segment)
start_sp = SimplePath(ijk=discretize(start), trajectory=np.array([[x0,y0,yaw0]]),
                      direction=0, steer=0.0)
start = Node(path=start_sp, cost=0.0, h_cost = H_COST * D[i0,j0], parent=None)
open = min_heap [start]
best_g = dictionary: key=(i,j,k) -> known minimum g

while open is not empty:
    cur = heappop(open)  # node with the smallest f

    if near the goal or can solve direct connection:
        # If direct connection is successful, wrap RSPath as Node for the last child, h=0
        backtrack parent chain to form the final path
        break

    # Expansion: try a batch of motion primitives (dir in {+1,-1} × steer in STEER_SET)
    for dir, steer in action set:
        # Use Car.update to roll continuously for a short distance (e.g., 3 meters), check for collisions every 0.5 meters within the segment
        traj, collided = rollout(cur.path.end_state, dir, steer, obstacles)
        if collided: continue

        child_sp = SimplePath(ijk=discretize(segment end), trajectory=traj, direction=dir, steer=steer)
        g2 = cur.cost + edge_cost(cur, child_sp)  # segment length + switch direction/steering angle/steering angle change penalty
        key = child_sp.ijk
        if key in best_g and g2 >= best_g[key]:  # prune inferior solutions
            continue

        h2 = H_COST * D[i,j]  # use the grid at the end of the segment for heuristic
        child = Node(path=child_sp, cost=g2, h_cost=h2, parent=cur)
        best_g[key] = g2
        heappush(open, child)
```

# Local Planner (MPC)

**Model Predictive Control（MPC）** 用来把全局路径（如 Hybrid A* 的 \([x,y,\psi]\) 点列）转换成每一帧可执行的**油门/刹车 + 转向**指令；它在给定预测域内，考虑车辆动力学与约束，优化控制序列，使车辆**平滑且受限**地跟踪参考路线。每个控制周期只执行求得序列的**第一步**，然后滚动重复。

---

## 预测域（Prediction Horizon）

选择预测步长 \(N\)（比如 10）与离散步距 \(\Delta t\)（你的 `LOCAL_PLANNER_DELTA_TIME`）。  
MPC 每次只优化**未来 \(N\) 步**。

---

## 车辆模型（Kinematic Bicycle，离散）

控制量 \(u_k=[a_k,\delta_k]\)（纵向加速度与前轮转角），轴距 \(L\)（`Car.WHEEL_BASE`）。离散动力学为：

$$
\begin{aligned}
X_{k+1} &= X_k + v_k \cos\psi_k\,\Delta t \\
Y_{k+1} &= Y_k + v_k \sin\psi_k\,\Delta t \\
v_{k+1} &= v_k + a_k\,\Delta t \\
\psi_{k+1} &= \psi_k + \frac{v_k}{L}\tan\delta_k\,\Delta t
\end{aligned}
$$

状态 \(x_k=[X_k, Y_k, v_k, \psi_k]^\top\)。

---

## 目标函数（Tracking + Smoothness）

在预测域内最小化**跟踪误差**与**控制平滑度**：

$$
J
= \sum_{k=0}^{N}\lVert x_k - x_k^{\mathrm{ref}}\rVert_{Q}^{2}
+ \sum_{k=0}^{N-1}\lVert u_k - u_k^{\mathrm{ref}}\rVert_{R}^{2}
+ \sum_{k=0}^{N-2}\lVert \Delta u_k\rVert_{R_\Delta}^{2}\,,
$$

其中 \(\Delta u_k = u_{k+1}-u_k\)。常见做法是对横向位置 \(Y\) 与航向 \(\psi\) 赋较大权重，对控制增量赋平滑权重。

---

## 硬约束（Physical/Regulatory Constraints）

转角限制（与机械极限一致）：
$$
|\delta_k| \le \delta_{\max}\,.
$$

加速度限制：
$$
|a_k| \le a_{\max}\,.
$$

速度范围：
$$
0 \le v_k \le v_{\max}\,.
$$

**横向加速度限制（关键）**：
$$
\Bigl|a_{y,k}\Bigr|
= \left|\frac{v_k^{2}\tan\delta_k}{L}\right|
\le a_{y,\max}\,.
$$

实践中常用**速度相关的转角上限**来近似实现横向加速度约束：
$$
|\delta_k|
\le
\min\!\left(
\delta_{\max},\;
\arctan\!\frac{a_{y,\max}\,L}{\max(v_k^{2},\,\varepsilon)}
\right),
$$

> 直觉：车速越高，安全可用的瞬时转角就越小，避免侧滑或失稳。

---

## 求解方法（QP + 迭代线性化）

将非线性模型在名义轨迹 \((\bar{x}_k,\bar{u}_k)\) 处线性化为：
$$
x_{k+1} \approx A_k\,x_k + B_k\,u_k + C_k\,,
$$

把目标函数写成二次型、约束写成线性/盒约束，即得到**带约束二次规划（QP）**。每个控制周期执行：

1. **构造参考**：从全局路径上找当前最近点，沿弧长/时间取 \(N{+}1\) 个参考状态 \(x_k^{\mathrm{ref}}\) 与（可选）\(u_k^{\mathrm{ref}}\)。
2. **线性化**：在名义轨迹上计算 \((A_k,B_k,C_k)\)。
3. **求解 QP**：最小化 \(J\)，满足动力学等式与上述硬约束。
4. **执行**：只下发第一帧控制 \(u_0\)，其余作废。
5. **滚动**：用解出的控制更新名义轨迹，下一拍重复 1–4。

---

## 实用提示

- **航向展开**：参考与实车的 \(\psi\) 建议用“角度展开”（unwrap）避免 \(-\pi/\pi\) 跳变导致误差暴涨。  
- **速度相关转角上限**：高效且稳定，强烈建议启用。  
- **参考窗口**：始终从“最近点”向前取 \(N\) 步参考，避免盯着历史点“回头看”。  
- **权重初值**：可从 \(Q=\mathrm{diag}(3,6,0.5,2)\)、\(R=\mathrm{diag}(0.2,0.3)\)、\(R_\Delta=\mathrm{diag}(0.1,1.0)\) 起步，再按误差与抖动调优。

---


## Test-Demo

### Collision Detection Test

```bash
python -m demo.demo_test_collision
```
Tests vehicle collision detection system with animated car movement. Shows real-time collision checking as the car moves through an obstacle environment with visual feedback.

![Autonomous Vehicle Collision Detection Demo](./images/collision_demo_20250910_171221.gif)


### Hybrid A* Path Planning

```bash
python -m demo.demo_hybridAstar_test
```
This is a complete demonstration of Hybrid A* path planning. It tests various scenarios, including diagonal navigation, goal orientation alignment, and corridor traversal, and visualizes the planned path results.

#### Scenario 1: Diagonal Path Planning
![Diagonal Path Planning](./images/hybrid_astar_diagonal_20250910_171714.gif)
**Description:** The vehicle plans a diagonal path from the bottom left (5,5) to the top right (55,55). This demonstrates the Hybrid A* algorithm's pathfinding capability in a complex obstacle environment, where the vehicle needs to navigate around two vertical poles to reach the target position.

#### Scenario 2: Goal Orientation Alignment
![Goal Orientation Alignment](./images/hybrid_astar_diagonal_90_20250910_171808.gif)
**Description:** Tests the algorithm's ability to handle terminal constraints. The vehicle starts from (5,5,0°) and the target position is (55,55,90°), requiring both position and orientation alignment. The algorithm achieves precise goal orientation alignment through a combination of forward and reverse maneuvers.

#### Scenario 3: Corridor Navigation
![Corridor Navigation](./images/hybrid_astar_corridor_20250910_171854.gif)

**Description:** The vehicle navigates through a narrow corridor between two poles, from (30,8,90°) to (30,52,90°). This scenario tests the algorithm's path planning capability in constrained spaces, requiring precise vehicle control to avoid collisions.
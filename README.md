# Autonomous Vehicle 

This project is a learning project for an autonomous driving path planning system based on the **Hybrid A* Algorithm** and **MPC** local planning. Huge credit to FredBill1. The original implementation: https://github.com/FredBill1/AutonomousDrivingDemo.git

**Learning Background:**
- Initially participated as a group project, mainly responsible for the global planner part
- To deeply understand the algorithm principles and implementation details, I am now independently redoing the entire project
- Focus on learning the application of the Hybrid A* Algorithm in vehicle motion planning with non-holonomic constraints

![Route Planning Demo](./test_output/screen_recording_720p.gif)

**Project Overview:** Developed a comprehensive autonomous driving path planning system based on Hybrid A* Algorithm and Model Predictive Control (MPC), implementing a two-tier architecture with Global Planner and Local Planner

- Core Technologies:
   - Global Planner: Implemented Hybrid A* algorithm for vehicle motion planning with non-holonomic constraints
   - Local Planner: Real-time trajectory tracking control using MPC with kinematic bicycle model
   - Collision Detection: Real-time obstacle detection and avoidance algorithms
   - Visualization Interface: Real-time simulation interface built with PySide6 and PyQtGraph

**Tech Stack:** Python 3.12+, NumPy, SciPy, CVXPY, PySide6, OpenCV, Matplotlib, PyQtGraph

**Key Features:**
   - Implemented complete autonomous driving path planning algorithms supporting complex obstacle environments
   - Adopted hierarchical architecture design with global planner providing coarse paths and local planner for fine control
   - Supported multiple test scenarios: diagonal navigation, goal orientation alignment, narrow corridor traversal
   - Provided real-time visualization interface with vehicle state monitoring, trajectory display, and performance metrics

## Dependencies

Requires Python 3.12 or later.

```bash
pip install -r requirements.txt
```

## Launching

### 1. **Docker Backend** version: 

```bash
docker-compose up --build
```
validate with:

```bash
curl http://localhost:5000/api/vehicle/health
```

### 2. **Desktop-QT** version:

```bash
python -m AutonomousVehicle
```

launches the main app. 

If you encounter the following error on macOS:

```qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in "" This application failed to start because no Qt platform plugin could be initialized.```

This is a common issue with PySide6 on macOS in virtual environments. Follow these steps to fix it:

#### Method1 (Manual Fix) 

1. **Install a compatible PySide6 version:**
   ```bash
   pip install PySide6==6.8.0
   ```

2. **Fix the Qt plugin rpath:**
   ```bash
   install_name_tool -add_rpath "$(pwd)/.venv/lib/python$(python -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=".")')/site-packages/PySide6/Qt/lib" .venv/lib/python$(python -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=".")')/site-packages/PySide6/Qt/plugins/platforms/libqcocoa.dylib
   ```

3. **Test the application:**
   ```bash
   python -m your_module_name
   ```

**Why this happens:** PySide6 6.9.x has known issues with macOS virtual environments. Using version 6.8.0 with proper rpath configuration resolves the plugin loading issue.
   

**Troubleshooting:**
- If you still get errors, try recreating your virtual environment
- Make sure you're running the commands from the project root directory
- For system-wide Python installations, you may need to use `sudo` with `install_name_tool`

#### Method2 (Recommended)

Or you can launch the `start_app` script instead should automatically fix the issue: 

```bash
python start_app.py
```

## Event Management and Architecture

[TODO]

## Test-Demo

### Collision Detection Test

```bash
python -m demo.test_collision
```
Tests vehicle collision detection system with animated car movement. Shows real-time collision checking as the car moves through an obstacle environment with visual feedback.

![Autonomous Vehicle Collision Detection Demo](./test_output/collision_demo_20250910_171221.gif)


### Hybrid A* Path Planning

```bash
python -m demo.test_hybridAstar
```
This is a complete demonstration of Hybrid A* path planning. It tests various scenarios, including diagonal navigation, goal orientation alignment, and corridor traversal, and visualizes the planned path results.

#### Scenario 1: Diagonal Path Planning
![Diagonal Path Planning](./test_output/hybrid_astar_diagonal.gif)
**Description:** The vehicle plans a diagonal path from the bottom left (5,5) to the top right (55,55). This demonstrates the Hybrid A* algorithm's pathfinding capability in a complex obstacle environment, where the vehicle needs to navigate around two vertical poles to reach the target position.

#### Scenario 2: Goal Orientation Alignment
![Goal Orientation Alignment](./test_output/hybrid_astar_diagonal_90.gif)
**Description:** Tests the algorithm's ability to handle terminal constraints. The vehicle starts from (5,5,0°) and the target position is (55,55,90°), requiring both position and orientation alignment. The algorithm achieves precise goal orientation alignment through a combination of forward and reverse maneuvers.

#### Scenario 3: Corridor Navigation
![Corridor Navigation](./test_output/hybrid_astar_corridor.gif)

**Description:** The vehicle navigates through a narrow corridor between two poles, from (30,8,90°) to (30,52,90°). This scenario tests the algorithm's path planning capability in constrained spaces, requiring precise vehicle control to avoid collisions.

## [Dev-log] Core Algorithm

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

---

### **Global Planner Implementation**

#### 0. **Constants and Action Space**

```python
XY_GRID_RESOLUTION = 1.0
YAW_GRID_RESOLUTION = deg2rad(15)
MOTION_DISTANCE = XY_GRID_RESOLUTION * 1.5 
MOTION_RESOLUTION    # simulation step size used in generate_neighbor()

NUM_STEER_COMMANDS = 10
STEER_COMMANDS = linspace(-Car.TARGET_MAX_STEER, +Car.TARGET_MAX_STEER, NUM_STEER_COMMANDS) + [0]
STEER_COMMANDS = unique(STEER_COMMANDS)

# 8-neighborhood moves for Dijkstra
MOVEMENTS = [(di,dj,cost) for di,dj in {-1,0,1}^2 except (0,0), cost=sqrt(di^2+dj^2)]

# Cost Weight
SWITCH_DIRECTION_COST
BACKWARDS_COST
STEER_CHANGE_COST
STEER_COST
H_DIST_COST
H_YAW_COST
H_COLLISION_COST
```

- SWITCH_DIRECTION_COST
    - g-cost; added when switching between forward and reverse (not applied to the initial node)
    - discourages frequent gear changes; reduces F-R-F "jitter" paths

- BACKWARDS_COST
    - g-cost; multiplies distance cost when moving backward
    - makes reverse more expensive, so forward is preferred if otherwise similar

- STEER_CHANGE_COST
    - g-cost; proportional to abs(steer - curr.path.steer)
    - penalizes sharp steering changes between segments; encourages a smooth steering profile

- STEER_COST
    - g-cost; proportional to abs(steer) * MOTION_DISTANCE
    - penalizes large steering angles; prefers straighter motion unless turning is necessary

- H_DIST_COST
    - h-cost; scales heuristic_grid.grid[i, j] (Dijkstra distance-to-go)
    - guides the search around obstacles and toward the goal efficiently

- H_YAW_COST
    - h-cost; multiplies heading error abs(wrap_angle(goal_yaw - yaw))
    - encourages arriving with correct final orientation (important for parking-like maneuvers)
- H_COLLISION_COST
    - heuristic "large number" used as initial/unreachable penalty in the Dijkstra grid
    - cells that are blocked/unreachable appear extremely distant, making A* avoid them

**For the purpose of Hybrid A*,**, we also need two lightweight data containers to (1) represent a locally simulated motion segment (the “edge” produced by one control action) and (2) represent a search node in the open list with enough information for DP pruning and traceback.

```python
class SimplePath(NamedTuple):
    ijk: tuple[int, int, int]
    trajectory: npt.NDArray[np.floating[Any]]
    direction: Literal[1, 0, -1]
    steer: float

class Node(NamedTuple):
    path: SimplePath | RSPath
    cost: float
    h_cost: float
    parent: Optional["Node"]

    def __lt__(self, other: "Node") -> bool:
        return (self.h_cost + self.cost, self.cost) < (other.h_cost + other.cost, other.cost)
```

- **SimplePath** (local segment / edge)
  - `ijk`: end-state discrete index for DP lookup (dp[i,j,k])
  - `trajectory`: the simulated continuous segment points
  - `direction`: +1 forward / -1 reverse / 0 initial
  - `steer`: steering angle used for this segment (used for steer change penalty)

- **Node** (search node / state in open list)
  - `path`: either a SimplePath segment or an RSPath final-connection segment
  - `cost`: g-cost accumulated from start to this node
  - `h_cost`: heuristic estimate to goal from this node
  - `parent`: backpointer for traceback_path
  - `__lt__`: heap ordering by f = g+h, tie-breaker by smaller g


#### 1. **heuristic: Dijkstra prediction of all grid to goal**

We need a fast-to-query heuristic `h_dist` that approximates **distance-to-go while respecting obstacles**.

We precompute a 2D grid `heuristic_grid.grid[i, j]` using Dijkstra on the downsampled obstacle grid, so later each Node can get `h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]` in O(1).


``` python
def _distance_heuristic(grid: ObstacleGrid, goal_xy):
    H, W = grid.grid.shape
    dist = full(H,W, H_COLLISION_COST)      # 初始都很大
    ij = grid.calc_index(goal_xy)           # goal 在 grid 上的 (i,j)
    dist[ij] = 0
    pq = [(0, ij)]                          # heapq: (distance, (i,j))

    while pq not empty:
        d, (i,j) = heappop(pq)
        if d > dist[i,j]: continue          # 过期条目剪枝

        for (di,dj,cost) in MOVEMENTS:
            ni,nj = i+di, j+dj
            n_cost = d + cost
            if inside(ni,nj) AND not grid.grid[ni,nj] (非障碍) AND n_cost < dist[ni,nj]:
                dist[ni,nj] = n_cost
                heappush(pq, (n_cost, (ni,nj)))

    return ObstacleGrid(..., dist)  # dist 当作新的 grid.grid
```

-  **ObstacleGrid (for heuristic)**
   - `grid.grid[i, j]`: boolean occupancy (True = blocked, False = free)
   - `grid.calc_index([x, y])`: maps continuous (x, y) → discrete (i, j)
- **dist / heuristic_grid.grid**
   - `dist[i, j]`: best-known Dijkstra distance from (i, j) to goal_ij
   - Initialized with `H_COLLISION_COST` (inf)
- **MOVEMENTS (8-neighborhood)**
   - Each move is (di, dj, step_cost)
   - Includes diagonals, with `step_cost = sqrt(di^2 + dj^2)` 
- **Blocked cell handling**
   - Condition not `grid.grid[ni, nj]` prevents entering obstacles
   - Blocked/unreachable remain near `H_COLLISION_COST`
   - **Effect on A***:  
      - `h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]` becomes huge ⇒ those states are strongly deprioritized
- **How Hybrid A* consumes it**
   - After simulating a neighbor and discretizing to (i, j, k):
      - `h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]`
      - Plus yaw term:  
         - `h_yaw_cost = H_YAW_COST * abs(wrap_angle(goal_yaw - yaw))`
      - `h_cost = h_dist_cost + h_yaw_cost`

#### 2. **state discretization + DP table (calc_ijk + dp pruning)**

We need a consistent way to **deduplicate states** during A* search.
Because the car state is continuous `(x, y, yaw)`, we discretize it into `(i, j, k)` so we can index a 3D DP table `dp[i, j, k]` and store the best-known `Node` for that discretized state.

We use:

```python
K = int(2*pi / YAW_GRID_RESOLUTION)
dp = full((N,M,K), None)     

def calc_ijk(x,y,yaw):
    i,j = heuristic_grid.calc_index([x,y])
    yaw0 = wrap_angle(yaw, zero_to_2pi=True)
    k = int(yaw0 // YAW_GRID_RESOLUTION)
    return (i,j,k)

```

- **(i, j) from heuristic_grid**
  - `heuristic_grid.calc_index([x, y])` maps continuous `(x, y)` → discrete grid indices.
  - Using the same grid as Dijkstra makes `h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]` directly usable.
- **k (yaw bin)**
  - `wrap_angle(yaw, zero_to_2pi=True)` normalizes yaw into `[0, 2π)`.
  - `k = int(yaw // YAW_GRID_RESOLUTION)` bins yaw into discrete buckets.
  - **Effect:** DP treats “same cell + same yaw bucket” as the same state.
- **dp[i, j, k] stores Node (best-known arrival)**
  - Each dp entry stores the `Node` with the smallest `cost` (g-cost) found so far for that `(i, j, k)`.
  - If a newly generated node is not better, it is discarded (pruned).
- **DP pruning rule (how it is used in the main loop)**
  - When popping from heap:
    - If `curr.cost > dp[curr.path.ijk].cost`, then `curr` is an outdated heap entry ⇒ skip.
  - When pushing neighbors:
    - If `dp[ijk] is None` or `neighbor.cost < dp[ijk].cost`, update dp and push into heap.
  - **Effect on A***: dramatically reduces redundant expansions and keeps search tractable.
- **Boundary note**
  - `calc_ijk()` itself does not guarantee `(i, j)` is inside `[0..N), [0..M)`.
  - Usually the boundary check happens after simulating motion (e.g., in `generate_neighbor`) before accepting a neighbor.

#### 3. **local expansion: generate_neighbor (simulate motion + collision + g/h cost)**

We need a way to expand a `Node` into dynamically-feasible neighbors.  
Instead of “grid-adjacent” moves, Hybrid A* generates each neighbor by **simulating the car** forward/backward for `MOTION_DISTANCE` with a chosen steering command.

Each successful simulation becomes a `SimplePath` segment and a new `Node` carrying:
- `cost` (g-cost): accumulated from start
- `h_cost` (h-cost): heuristic distance + heading error

```python
def generate_neighbor(curr: Node, direction: int, steer: float) -> Optional[Node]:
    # start pose = end pose of current segment
    x0, y0, yaw0 = curr.path.trajectory[-1, :3]

    # simulate car using the chosen control (direction, steer)
    car = Car(x0, y0, yaw0, velocity=direction, steer=steer)
    trajectory = []

    # integrate motion for a fixed arc length
    for step in range(int(MOTION_DISTANCE / MOTION_RESOLUTION)):
        car.update(MOTION_RESOLUTION)

        # collision check along the segment
        if (not start_collided) and car.check_collision(obstacles):
            return None

        trajectory.append([car.x, car.y, car.yaw])

    # discretize end pose into dp index
    i, j, k = calc_ijk(car.x, car.y, car.yaw)
    if not inside_grid(i, j, N, M):
        return None

    # g-cost (true accumulated cost)
    distance_cost = MOTION_DISTANCE if direction == +1 else MOTION_DISTANCE * BACKWARDS_COST
    switch_direction_cost = SWITCH_DIRECTION_COST if (curr.path.direction != 0 and direction != curr.path.direction) else 0
    steer_change_cost = STEER_CHANGE_COST * abs(steer - curr.path.steer)
    steer_cost = STEER_COST * abs(steer) * MOTION_DISTANCE
    new_cost = curr.cost + distance_cost + switch_direction_cost + steer_change_cost + steer_cost

    # h-cost (heuristic)
    h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]
    h_yaw_cost  = H_YAW_COST  * abs(wrap_angle(goal_yaw - car.yaw))
    new_h_cost  = h_dist_cost + h_yaw_cost

    # package neighbor node
    new_path = SimplePath(ijk=(i, j, k), trajectory=array(trajectory), direction=direction, steer=steer)
    return Node(path=new_path, cost=new_cost, h_cost=new_h_cost, parent=curr)
```

- **Why we simulate (instead of grid moves)**
  - The car has nonholonomic constraints; many “adjacent cells” are not reachable with feasible curvature.
  - Simulation ensures each neighbor segment is physically plausible under `(direction, steer)`.

- **Start pose source**
  - `curr.path.trajectory[-1, :3]` is the current end pose.
  - **Effect:** the search state is continuous; discretization is only for DP indexing.

- **Segment length and resolution**
  - `MOTION_DISTANCE`: total arc length for one expansion step.
  - `MOTION_RESOLUTION`: integration step size inside that segment.
  - **Effect:** smaller `MOTION_RESOLUTION` improves collision accuracy but costs more compute.

- **Collision checking along the segment**
  - Check collision at each integration step (not just at the end).
  - **Effect:** prevents “tunneling” through obstacles between sparse samples.

- **Discretization of the end pose**
  - `calc_ijk(car.x, car.y, car.yaw)` produces `(i, j, k)` for `dp[i, j, k]`.
  - Boundary checks are done here; out-of-grid neighbors are rejected.

- **g-cost components (accumulate into `new_cost`)**
  - `distance_cost`: forward cost vs backward cost (`BACKWARDS_COST`).
  - `switch_direction_cost`: penalize gear changes (`SWITCH_DIRECTION_COST`).
  - `steer_change_cost`: penalize sudden steer changes (`STEER_CHANGE_COST`).
  - `steer_cost`: penalize large steering magnitude (`STEER_COST`).

- **h-cost components (store into `new_h_cost`)**
  - `h_dist_cost`: obstacle-aware distance-to-go from Dijkstra (`H_DIST_COST * heuristic_grid.grid[i, j]`).
  - `h_yaw_cost`: heading alignment toward `goal_yaw` (`H_YAW_COST * abs(wrap_angle(goal_yaw - yaw))`).
  - `new_h_cost = h_dist_cost + h_yaw_cost`.


#### 4. **local expansion: generate_neighbors (enumerate action space)**

We need to generate **all candidate neighbors** of the current `Node` by enumerating the discrete action space:  
`direction ∈ {+1, -1}` (forward/reverse) × `steer ∈ STEER_COMMANDS`.

`generate_neighbors(curr)` is a thin wrapper that repeatedly calls `generate_neighbor(curr, direction, steer)` and yields only feasible results (non-colliding, in-bounds).

```python
def generate_neighbors(curr: Node) -> Generator[Node, None, None]:
    nonlocal start_collided

    for direction in [+1, -1]:
        for steer in STEER_COMMANDS:
            node = generate_neighbor(curr, direction, steer)
            if node is not None:
                yield node

    # start-collision special-case is only allowed once;
    # after the first expansion from the start, turn it off
    start_collided = False
```

- **Action space definition**
  - `direction`: `+1` forward, `-1` reverse.
  - `STEER_COMMANDS`: discretized steering angles in `[-Car.TARGET_MAX_STEER, +Car.TARGET_MAX_STEER]` plus `0`.

- **Filtering invalid actions**
  - `generate_neighbor` returns `None` if the simulated segment collides or goes out of bounds.
  - `generate_neighbors` yields only valid `Node` objects.

- **Why yield (Generator)**
  - Neighbors are produced lazily; the caller can process them one-by-one without allocating a full list.
  - This reduces memory pressure when many expansions happen.

- **start_collided handling**
  - `start_collided` is a special flag allowing a start pose that is initially in collision.
  - After the first expansion attempt, `start_collided = False` so later expansions enforce normal collision rules.


#### 5. **main A* loop (dp pruning + priority queue expansion)**

We need a global search loop that repeatedly expands the most promising `Node` (lowest `f = g + h`) until a solution is found or the open list is exhausted.

We use:
- `q` (a `heapq`) as the open list, ordered by `Node.__lt__` (primary: `g+h`, tie-break: smaller `g`)
- `dp[i, j, k]` to store the best-known `Node` for each discretized state and prune outdated/expensive states

```python
# initialization (start node already constructed)
dp[start_ijk] = start_node
q = [start_node]  # heapq open list

while q:
    curr = heapq.heappop(q)  # pop min f = g+h

    # outdated entry pruning:
    # heap may contain older versions of the same (i,j,k) state
    if curr.cost > dp[curr.path.ijk].cost:
        continue

    # optional early-cancel hook (UI / websocket / user stop)
    if cancel_callback is not None and cancel_callback(curr):
        return None

    # expand neighbors using motion primitives
    for neighbor in generate_neighbors(curr):
        ijk = neighbor.path.ijk

        # DP relaxation: keep only cheaper g-cost for each discretized state
        if dp[ijk] is None or neighbor.cost < dp[ijk].cost:
            dp[ijk] = neighbor
            heapq.heappush(q, neighbor)

# if open list exhausted, no feasible path found
return None
```

- **Why we still need dp even with A***
  - The same discretized state `(i, j, k)` can be reached via many different motion sequences.
  - `dp[i, j, k]` stores the best-known (lowest g-cost) arrival and prunes worse duplicates.

- **Outdated entry pruning**
  - Even after `dp[ijk]` is improved, older nodes for the same `ijk` may still remain in the heap.
  - Condition `curr.cost > dp[curr.path.ijk].cost` skips those stale heap entries.

- **DP relaxation rule**
  - Only accept `neighbor` if it improves the best-known g-cost for its `ijk`.
  - If accepted: update `dp[ijk]` and push into `q`.

- **Connection to your earlier components**
  - `generate_neighbors(curr)` provides feasible motion segments (already collision-checked).
  - `neighbor.path.ijk` is the discretized key used for dp pruning.
  - `neighbor.cost` is accumulated g-cost; `neighbor.h_cost` is computed inside `generate_neighbor`.

#### 6. **goal connection: Reeds–Shepp shortcut (generate_rspath near the goal)**

We need a reliable way to finish the plan once the search gets close to the goal.  
Instead of continuing many small motion-primitive expansions, Hybrid A* tries an analytic connection using **Reeds–Shepp (RS)** from the current pose to the goal pose.

When `||curr_xy - goal_xy|| <= REEDS_SHEPP_MAX_DISTANCE`, we attempt:
- generate candidate `RSPath` solutions
- collision-check them
- compute the RS segment cost
- wrap the best RS segment into a `Node(path=RSPath, parent=curr)`
- either return immediately (fast) or push it into the heap

```python
# constants (typical pattern)
REEDS_SHEPP_MAX_DISTANCE = ...         # try RS when close enough to goal
RETURN_RS_PATH_IMMEDIATELY = ...       # if True: return once RS succeeds

def generate_rspath(curr: Node) -> Optional[Node]:
    start_pose = tuple(curr.path.trajectory[-1, :3])   # (x, y, yaw)
    goal_pose  = tuple(goal)                           # (x, y, yaw)

    # 1) generate RS candidates (analytic paths that allow forward+reverse)
    paths = solve_rspath(
        start_pose,
        goal_pose,
        Car.TARGET_MIN_TURNING_RADIUS,
        MOTION_RESOLUTION,
    )

    # 2) filter by collision + feasibility (library-dependent)
    paths = filter(check_rspath_collision, paths)

    # 3) compute RS segment cost and pick the best one
    best_path, best_cost = argmin_over_paths(paths, key=calc_rspath_cost)
    if best_path is None:
        return None

    # 4) wrap into a Node; heuristic can be 0 because RS reaches the goal
    rs_node = Node(path=best_path, cost=curr.cost + best_cost, h_cost=0.0, parent=curr)
    return rs_node

# integration into main loop (inside while q:)
if distance(curr_xy, goal_xy) <= REEDS_SHEPP_MAX_DISTANCE:
    rsnode = generate_rspath(curr)
    if rsnode is not None:
        if RETURN_RS_PATH_IMMEDIATELY:
            return traceback_path(rsnode)
        else:
            heapq.heappush(q, rsnode)
```

- **Why we need RS near the goal**
  - Small motion primitives may take many expansions to precisely match the goal pose (especially yaw).
  - RS provides an analytic “direct connection” that can finish efficiently.

- **When we attempt RS**
  - Condition: `np.linalg.norm(curr_xy - goal_xy) <= REEDS_SHEPP_MAX_DISTANCE`.
  - **Effect:** only try RS in a local region to keep computation reasonable.

- **What RSPath represents**
  - An analytic path that respects a minimum turning radius and allows forward/reverse.
  - Typically already includes direction for each waypoint (`driving_direction`).

- **Collision checking is mandatory**
  - RS solutions are geometric; without `check_rspath_collision`, they may cut through obstacles.
  - Only collision-free RS candidates are allowed.

- **RS segment cost**
  - Usually includes distance (forward/reverse), direction switches, and steering-related penalties (implementation-dependent).
  - The chosen RS candidate is the one with smallest added cost from `curr`.

- **Why `h_cost = 0.0` for RS node**
  - The RS segment is intended to reach the goal pose directly.
  - After adding the RS node, the remaining heuristic is effectively zero.

- **Two policies after RS success**
  - `RETURN_RS_PATH_IMMEDIATELY = True`: fastest; stop search and reconstruct.
  - Otherwise: push RS node into heap and let A* decide (more optimal but slower).


#### 7. **traceback: reconstruct final trajectory (traceback_path + cleanup)**

We need to convert the linked `Node` chain (via `parent`) into a single final trajectory.  
Each `Node` stores only a local segment (`SimplePath` or `RSPath`), so we traceback from the goal node to the start node, collect all segments, reverse them, and `vstack` into one continuous array.

We also normalize the output format to `[[x, y, yaw, direction], ...]` and perform a small cleanup to remove “jitter” points around consecutive direction switches.

```python
def traceback_path(node: Node) -> NDArray:
    segments = []

    # collect segments from goal -> start
    while node is not None:
        path = node.path

        if isinstance(path, SimplePath):
            # SimplePath trajectory can be Nx3 or Nx4
            if traj_has_4_cols(path.trajectory):
                segments.append(path.trajectory)
            else:
                # append direction column using path.direction
                segments.append(hstack_xyz_and_direction(path.trajectory, path.direction))

        else:  # RSPath
            # RSPath includes the start point, skip it to avoid duplicate joints
            seg = [[p.x, p.y, p.yaw, p.driving_direction] for p in islice(path.waypoints(), 1, None)]
            if seg is not empty:
                segments.append(seg)

        node = node.parent

    # reverse to start -> goal, then stitch
    segments.reverse()
    trajectory = vstack(segments)  # shape (T, 4)

    # fix initial direction (avoid 0 / undefined at index 0)
    d = trajectory[:, 3]
    d[0] = d[1] if len(d) > 1 else 1

    # cleanup: remove consecutive direction-changing points (F-R-F or R-F-R)
    switch = (d[t] != d[t+1]) for t in [0..T-2]
    remove_mid = switch[t] AND switch[t+1]          # consecutive switches
    keep_mask = [True] + [not remove_mid] + [True]
    trajectory = trajectory[keep_mask]

    return trajectory
```

- **Why we need traceback_path**
  - A* stores a `parent` pointer instead of storing the full path for every node.
  - `traceback_path(goal_node)` reconstructs the chosen route by following `parent` links.

- **What is being stitched**
  - Each `Node.path` is a local segment:
    - `SimplePath`: simulated motion primitive segment
    - `RSPath`: final analytic connection segment near the goal
  - The final trajectory is the concatenation of all segments in order.

- **Why we reverse segments**
  - The `parent` chain goes from goal → start.
  - Reversing `segments` restores chronological order: start → goal.

- **Why RSPath skips its first waypoint**
  - RSPath often includes its start pose as the first waypoint.
  - If we also include the last pose of the previous segment, the joint point duplicates.
  - `islice(..., 1, None)` avoids repeating that shared point.

- **Output normalization (Nx4)**
  - `SimplePath.trajectory` may be `Nx3`; direction is stored separately in `SimplePath.direction`.
  - Traceback ensures output is always `[[x, y, yaw, direction], ...]`.

- **Cleanup: remove consecutive direction-switch points**
  - If direction changes twice in a row (`+1 -> -1 -> +1` or `-1 -> +1 -> -1`), the middle point causes jitter.
  - The mask removes the middle point of these “double-switch” patterns to smooth the final trajectory.


---

### Local Planner (MPC)

#### 1. Intuition

**Model Predictive Control (MPC)** is an advanced control strategy used in the local planning of autonomous vehicles. It involves predicting the future behavior of the vehicle over a defined prediction horizon and optimizing the control inputs to achieve desired objectives. MPC takes into account the vehicle's dynamics, constraints, and a reference trajectory to minimize tracking errors and ensure smooth control actions. By solving an optimization problem at each time step, MPC provides a sequence of control actions that guide the vehicle along the optimal path while respecting physical and regulatory constraints.

Think of **MPC** as:

At every frame (each dt), I “look ahead into the future” for a few steps (the prediction horizon), simulating what would happen if I apply a sequence of throttle and steering inputs. I then pick the “least-bad” control sequence, but actually only execute the first step — the process repeats at the next frame.

So, the core MPC loop is:

- Give the current vehicle state (x, y, yaw, v)
- Provide a reference trajectory you want the vehicle to follow
- Search for a control sequence (accelerations, steering angles) over the next HORIZON_LENGTH steps
- Make the vehicle “stick to” the trajectory as closely as possible, while keeping the controls smooth and not too aggressive
- Output: the next few control actions + the predicted path

In the demo, the global planner (Hybrid A*) provides a coarse route; MPC is responsible for making the car smoothly follow this route. In `main.py` you can also see how MPC is called: `mpc.update(state, DT)`, then you use `result.controls[1]` to update the car.


**Movement Modeling of `Car`**

What `Car.update_with_control()` does:

- The car first moves a small step forward based on its current speed and heading
- Then, its velocity gradually approaches the target_velocity (limited by the maximum acceleration)
- Then, its steering angle gradually approaches the target_steer (limited by the maximum steering rate)

In other words: the control input does not "instantly teleport" the car to a new state, but instead, the state changes gradually under physical constraints.

MPC works by planning controls that make the car follow the trajectory while respecting these constraints.

**Scoring Rule**

For the matrices:
- `R`: Penalizes "excessively large control inputs" (throttle/steering too aggressive)
- `R_D`: Penalizes "rapid changes in control" (jittery actions)
- `Q`: Penalizes "deviation of state from reference trajectory"
- `Q_F`: Heavier penalty at the final state (final step matters most)

You can think of it as a "scoring function":
- If the vehicle deviates from the path → subtract points (`Q`)
- If throttle/steering is too aggressive → subtract points (`R`)
- If actions are jittery, e.g., rapidly switching left/right → subtract points (`R_D`)
- Being close to the final target at the last step matters even more → subtract more points (`Q_F`)

What MPC does: under constraints, it finds the set of control actions that minimize the total penalty score.

#### 2. **The Structuring of the Computation**

There are mainly **4 parts** in the MPC module:

- Linearized model: _get_linear_model_matrix()
- Prediction (rollout): _predict_motion()
- Solving the optimization problem with cvxpy: _linear_mpc_control()
- Preparing the "reference trajectory" into an MPC-usable form + per-frame update: ModelPredictiveControl class

---

**Part 1. `_get_linear_model_matrix()`**

The real vehicle model is **nonlinear** (it involves $\cos\psi$, $\sin\psi$, and $\tan\delta$ terms). Solving nonlinear optimal control problems directly is computationally expensive and often intractable for real-time applications. 

MPC uses a clever trick: instead of solving the nonlinear problem globally, we **linearize** the model around a reference trajectory. This transforms the nonlinear optimization into a **quadratic programming (QP)** problem, which can be solved efficiently using standard solvers (like OSQP or CVXPY).

> **Kinematic Bicycle Model**
>
> Given:
> - $u_k = [a_k, \delta_k]$: throttle/brake (longitudinal acceleration $a_k$) and front wheel steering angle $\delta_k$
> - $L$: wheelbase (`Car.WHEEL_BASE`)
>The discretized kinematic model is:
>$$
>\begin{aligned}
>X_{k+1} &= X_k + v_k \cos\psi_k\,\Delta t \\
>Y_{k+1} &= Y_k + v_k \sin\psi_k\,\Delta t \\
>v_{k+1} &= v_k + a_k\,\Delta t \\
>\psi_{k+1} &= \psi_k + \frac{v_k}{L}\tan\delta_k\,\Delta t
>\end{aligned}
>$$

**1.1 The Mathematical Principle**

The linearization is based on the **first-order Taylor expansion**. For a nonlinear function $f(x, u)$, we approximate it around a reference point $(\bar{x}, \bar{u})$:

$$f(x, u) \approx f(\bar{x}, \bar{u}) + \frac{\partial f}{\partial x}\Big|_{\bar{x},\bar{u}}(x - \bar{x}) + \frac{\partial f}{\partial u}\Big|_{\bar{x},\bar{u}}(u - \bar{u})$$

For the bicycle model, this gives us a linear discrete-time model:

$$X[t+dt] = A \cdot X[t] + B \cdot u[t] + C$$

where:
- **$A$** (state matrix): describes how the current state $X[t]$ affects the next state
- **$B$** (input matrix): describes how the control input $u[t]$ affects the next state  
- **$C$** (affine term): constant offset that accounts for the linearization around the reference point

**1.2 The "Small dt" Assumption**

The key assumption is that within a very small time step `dt` (typically 0.1-0.2 seconds), the vehicle's velocity, yaw angle, and steering angle remain **almost constant**. This is reasonable because:
- Vehicles have physical limits on how fast they can change speed, direction, and steering
- Over a short time interval, these changes are small compared to their current values
- The linearization error is proportional to the size of `dt` — smaller `dt` means better approximation

**1.3 Physical Interpretation of the Matrices**

The matrices have intuitive meanings:

- **$A[0,2] = dt \cdot \cos(\psi)$**: How velocity affects X-position (depends on heading)
- **$A[1,2] = dt \cdot \sin(\psi)$**: How velocity affects Y-position (depends on heading)
- **$A[0,3] = -dt \cdot v \cdot \sin(\psi)$**: How yaw angle affects X-position (cross-coupling term)
- **$A[1,3] = dt \cdot v \cdot \cos(\psi)$**: How yaw angle affects Y-position (cross-coupling term)
- **$A[3,2] = dt \cdot \tan(\delta) / L$**: How velocity affects yaw rate (steering effect)
- **$B[2,0] = dt$**: How acceleration directly affects velocity (simple integration)
- **$B[3,1] = dt \cdot v / (L \cdot \cos^2(\delta))$**: How steering angle affects yaw rate (depends on speed and wheelbase)

**1.4 The Affine Term $C$**

The $C$ vector captures the "offset" from linearization. It accounts for terms that don't fit into the $AX + Bu$ form, ensuring the linearized model matches the nonlinear model at the reference point.

**1.5 Why This Matters**

- **Computational efficiency**: QP problems can be solved in milliseconds, enabling real-time control
- **Stability**: Linear models are easier to analyze and stabilize
- **Iterative refinement**: MPC solves this linearized problem iteratively, refining the reference trajectory each time (see Part 2)

**1.6 Trade-offs**

- **Accuracy vs. Speed**: Smaller `dt` improves accuracy but requires more computation
- **Linearization error**: The approximation is only valid near the reference trajectory
- **Iterative convergence**: If the initial guess is poor, it may take several iterations to converge

Here, `X` is `[x, y, v, yaw]`, and `u` is `[accel, steer]`. The `Car.WHEEL_BASE` parameter appears in the yaw dynamics, reflecting how the distance between front and rear axles affects turning behavior.

---

**Part 2. `_predict_motion()`: Use the "previous round of controls" to first guess a future trajectory**

MPC solves the optimization problem iteratively. Like any iterative solver, it needs a starting point (initial guess) to begin the optimization process:

- The vehicle dynamics model (bicycle model) is **nonlinear** (notice the $\cos\psi_k$, $\sin\psi_k$, and $\tan\delta_k$ terms)
- To make the optimization tractable, we linearize the model around a reference trajectory
- This reference trajectory comes from the initial guess: we predict where the vehicle would go if we applied a certain control sequence

**2.1 The Question: What Control Sequence for Initial Guess?**

The question is: what control sequence should we assume for the next $N$ steps as our initial guess?

**2.2 The Initial Guess Strategy**

- Initialize the `controls` as all zeros: `[accel=0, steer=0]` for all $N$ steps
  - This means: "assume we maintain current speed and go straight"
  - It's a safe, conservative starting point that avoids aggressive initial guesses
- Use `Car.update_with_control()` to simulate forward: starting from the **current state** (position, velocity, yaw), apply these zero controls step by step for $N$ time steps
   - This produces a predicted trajectory: `states = [[x, y, v, yaw], ...]` (length $N+1$)
      - The first point is the current state: `[x_0, y_0, v_0, \psi_0]`
      - The next $N$ points are predicted future states: `[x_1, y_1, v_1, \psi_1]`, ..., `[x_N, y_N, v_N, \psi_N]`
      - **In simple terms**: Based on the current heading and velocity, we simulate forward $N$ steps with zero control inputs, generating $N$ additional predicted state points

**2.3 How This Is Used**

This predicted trajectory (often denoted as $\bar{x}$) becomes the **linearization point**. In the next step (`_get_linear_model_matrix()`), we linearize the nonlinear bicycle model around each point in this trajectory. This transforms the nonlinear optimization problem into a **quadratic programming (QP)** problem, which can be solved efficiently.

**2.4 Why This Works**

- If the initial guess is close to the optimal solution, the linearization is accurate and the solver converges quickly
- Even if the guess is poor, the iterative nature of MPC means we'll refine it in subsequent iterations
- Using zero controls is reasonable because: (1) it's safe (no sudden movements), (2) it's simple, and (3) the optimizer will quickly adjust it toward the optimal solution

---

**Part 3. `_linear_mpc_control()`: How CVXPY solves the problem**

The key point here: it formulates the problem as a convex optimization (quadratic program/QP), and then lets CVXPY solve it.

> **Quadratic Programming (QP)**
>
> **QP** is a special type of optimization problem where:
> - **Objective function** is quadratic (contains terms like $x^T Q x$)
> - **Constraints** are linear (equalities $Ax = b$ and inequalities $Cx \leq d$)
>
> **General QP form:**
> $$
> \begin{aligned}
> \min_x \quad & \frac{1}{2}x^T Q x + q^T x \\
> \text{s.t.} \quad & Ax = b \\
> & Cx \leq d
> \end{aligned}
> $$
>
> **Why QP is special:**
> - **Convex**: If $Q$ is positive semi-definite, the problem is convex, meaning any local minimum is also the global minimum
> - **Efficient solvers**: QP problems can be solved very quickly (milliseconds) using specialized algorithms (interior-point methods, active-set methods)
> - **Well-understood**: Decades of research have produced robust, reliable solvers

**3.1 QP in MPC Context**

- The **cost function** (tracking error, control effort) is quadratic: $\|x - x_{ref}\|_Q^2 + \|u\|_R^2$
- The **dynamics** (after linearization) become linear constraints: $x_{k+1} = A x_k + B u_k + C$
- The **physical limits** (speed, acceleration, steering bounds) are linear inequalities

This is why MPC can run in real-time: the nonlinear problem is transformed into a QP, which modern solvers (like OSQP, used by CVXPY) can solve extremely fast.

**3.2 Decision Variables**

- **$x$**: The next `HORIZON_LENGTH + 1` state vectors (shape $N_X \times (H+1)$)
- **$u$**: The next `HORIZON_LENGTH` control inputs (shape $N_U \times H$)

In other words: we need to decide all control actions for the next $H$ steps at once.

**3.3 Cost Function (What We're Trying to Minimize)**

The cost function measures how "good" a solution is. The solver tries to find the control sequence that minimizes this total cost. The cost is computed by looping over each time step $t$ and summing up penalty terms:

- **Control effort penalty** (`quad_form(u, R)`): Smaller control inputs are better
  - Why: Saves energy, reduces wear, and avoids aggressive maneuvers
  - Example: Prefer gentle acceleration over hard braking when possible

- **Tracking error penalty** (`quad_form(x_{ref} - x, Q)`): Closer tracking to the reference trajectory is better
  - Why: We want the vehicle to follow the planned path accurately
  - The weight matrix $Q$ determines which states matter more (typically $Y$ position and yaw angle have higher weights than $X$ position)

- **Control smoothness penalty** (`quad_form(u[t] - u[t-1], R_D)`): Smaller changes between consecutive control actions are better
  - Why: Prevents jerky motion (sudden steering changes, abrupt acceleration/braking)
  - This is crucial for passenger comfort and vehicle stability

- **Final state penalty** (`quad_form(x_{ref}[H] - x[H], Q_F)`): The final predicted state should be close to the reference
  - Why: Ensures the vehicle is on track at the end of the prediction horizon
  - Typically $Q_F$ has larger weights than $Q$ to emphasize terminal accuracy

**In simple terms**: The cost function balances three competing goals:
1. **Follow the path** (tracking)
2. **Be gentle** (small, smooth controls)
3. **End up in the right place** (terminal accuracy)

The weight matrices ($Q$, $R$, $R_D$, $Q_F$) determine the relative importance of these goals. Tuning these weights is a key part of MPC design.

**3.4 Constraints**

Constraints are the "rules" that the solution must follow. These are the most engineering-heavy but also most intuitive part of MPC. They ensure the solution is physically feasible and safe.

**3.4.1 Physical/Regulatory Constraints (Hard Limits)**

These constraints represent real-world physical limits that the vehicle cannot exceed:

> 1. **Steering limit:**  
>    $$
>    |\delta_k| \le \delta_{\max}
>    $$
>    The steering angle cannot exceed the maximum steering angle of the vehicle (e.g., ±30°). This is a mechanical limit of the steering system.
>
> 2. **Acceleration limit:**  
>    $$
>    |a_k| \le a_{\max}
>    $$
>    The longitudinal acceleration (throttle/brake) is limited by the engine power and braking capacity. You can't accelerate or brake infinitely fast.
>
> 3. **Speed range:**  
>    $$
>    0 \le v_k \le v_{\max}
>    $$
>    The vehicle speed must be non-negative (can't go backwards in this model) and cannot exceed the maximum speed (engine/road limits).
>
> 4. **Lateral acceleration limit (critical!):**  
>    $$
>    \Bigl|a_{y,k}\Bigr| = \left|\frac{v_k^{2}\tan\delta_k}{L}\right| \le a_{y,\max}
>    $$
>    This is the **most important safety constraint**. When you turn at high speed, the vehicle experiences lateral (sideways) acceleration. If this exceeds the tire friction limit, the vehicle will skid. The formula shows that lateral acceleration increases with the square of speed and the steering angle.
>
>    **Commonly used speed-dependent steering angle limit:**
>    $$
>    |\delta_k| \le \min\left( \delta_{\max},\, \arctan\frac{a_{y,\max} L}{\max(v_k^{2}, \varepsilon)} \right)
>    $$
>    This combines the mechanical steering limit with the lateral acceleration limit. **The faster you go, the smaller the allowable steering angle to prevent skidding.** This is why you can't make sharp turns at high speeds in real life.

**3.4.2 Implementation Constraints (How the Solver Works)**

These constraints ensure the mathematical formulation is correct and the solution respects the vehicle dynamics:

1. **Dynamics constraint**: The next state must equal what the linearized model predicts:
   $$x_{k+1} = A \cdot x_k + B \cdot u_k + C$$
   This ensures the predicted trajectory actually follows the vehicle's physics (as approximated by the linearized model).

2. **Initial steering constraint**: By default, the first control step cannot immediately change the steering angle:
   $$u[1, 0] = \text{last\_steer}$$
   This prevents sudden steering changes that would be physically impossible (steering wheels don't turn instantly).

3. **Steering rate limit**: The change in steering angle between consecutive steps is limited:
   $$|\Delta \delta_k| = |\delta_k - \delta_{k-1}| \le \text{MAX\_STEER\_SPEED} \times dt$$
   This models the physical limit on how fast the steering wheel can turn (e.g., maximum steering rate of 30°/s).

4. **Speed bounds**: 
   $$\text{MIN\_SPEED} \le v_k \le \text{MAX\_SPEED}$$
   Enforces the speed range constraint at every time step.

5. **Acceleration bounds**: 
   $$|a_k| \le \text{MAX\_ACCEL}$$
   Limits the throttle/brake input at every step.

6. **Steering bounds**: 
   $$|\delta_k| \le \text{MAX\_STEER}$$
   Limits the steering angle at every step (may be further restricted by speed-dependent limit above).

7. **Initial state constraint**: The first state must match the current vehicle state:
   $$x[:, 0] = \bar{x}[:, 0]$$
   This "anchors" the optimization to start from where the vehicle actually is right now.

**3.5 Solving the QP Problem**

Finally, `cvxpy.Problem(...).solve(...)` uses the **CLARABEL** solver (or OSQP) to find the optimal solution. 

- If a solution is found: returns the optimal control sequence and predicted states
- If no solution exists (infeasible): prints an error and returns `None`
  - This can happen if constraints are too tight (e.g., trying to make an impossible turn)
  - In practice, the MPC should be designed so this rarely happens

**3.6 Why Constraints Matter**

- **Safety**: Prevents the optimizer from suggesting physically impossible or dangerous maneuvers
- **Realism**: Ensures the solution can actually be executed by a real vehicle
- **Stability**: Helps the MPC produce smooth, predictable behavior 

---

**Part 4. Trajectory Preprocessing**

**The Problem: Why Preprocessing is Critical**

MPC is very sensitive to two common issues in reference trajectories:

1. **Yaw angle discontinuities**: When the yaw angle jumps suddenly (e.g., from $+\pi$ to $-\pi$)
   - This happens because angles wrap around at $\pm\pi$
   - Direct interpolation between $+\pi$ and $-\pi$ would make the vehicle "spin" 360°, which is physically impossible
   - MPC's linearization breaks down when the reference trajectory has such discontinuities

2. **Forward/reverse direction switches without stopping**: When the trajectory switches from forward to reverse (or vice versa) but doesn't require the vehicle to stop first
   - You cannot instantly switch from high-speed forward to reverse
   - The vehicle must come to a complete stop before changing direction
   - Without this, MPC will try to find an impossible solution

Therefore, the initialization process performs extensive preprocessing to make the trajectory "trackable" by MPC.

**Step 4.1: Input Format Requirements**

The `ref_trajectory` must have the format: each row is `[x, y, yaw, direction]`, where:
- `direction` can only be `+1` (forward) or `-1` (reverse), **never 0**
- This matches the output format from the global planner (Hybrid A* traceback also outputs a `direction` column)

**Step 4.2: Remove Consecutive Duplicate Points**

Duplicate points (same x, y coordinates) are removed to prevent issues with:
- **Interpolation**: Can't interpolate between identical points
- **Distance calculations**: Zero distance between points causes division by zero
- **Spline fitting**: Duplicates make the spline ill-conditioned

**Step 4.3: Smooth Yaw Angles (`smooth_yaw`)**

**The problem**: Angles have a "wrap-around" at $\pm\pi$. Direct interpolation between $+\pi$ and $-\pi$ would cause a sudden jump of $2\pi$ radians (360°), which is physically impossible.

**The solution**: `smooth_yaw()` processes the yaw sequence:
1. Computes the difference between consecutive yaw angles
2. Wraps each difference to the range $[-\pi, \pi]$ using `wrap_angle()`
3. Cumulatively sums these wrapped differences to create a continuous sequence

**Example**:
- Input: `[0, π/2, π, -π, -π/2]` (has a jump at the 4th element)
- After wrapping differences: `[0, π/2, π/2, 0, π/2]` (no jump)
- After cumulative sum: `[0, π/2, π, π, 3π/2]` (continuous, no sudden jumps)

This is called in initialization: `ref_trajectory[:, 2] = smooth_yaw(...)`

**Step 4.4: Handle Forward/Reverse Direction Switches**

**This is critical logic**: When the `direction` changes (from +1 to -1 or vice versa):

1. **Insert additional points** near the switch point to create a smooth transition
2. **Set the speed to 0** at the switch point
3. **Set the goal point speed to 0** as well

**Why this matters**: You cannot instantly switch from high-speed forward to reverse. The vehicle must come to a complete stop first. This constraint is enforced by setting $v = 0$ at direction change points, ensuring MPC plans a stop before reversing.

**Physical intuition**: Think of parallel parking — you must stop completely before shifting from forward to reverse gear.

**Step 4.5: Convert Direction to Target Velocity**

The `direction` column (values +1 or -1) is converted to target velocity:
- Multiply by `Car.TARGET_SPEED` to get the desired speed magnitude
- Apply direction: positive for forward (+1), negative for reverse (-1)
- Clip to the vehicle's speed range: `[MIN_SPEED, MAX_SPEED]` or `[-MAX_SPEED, -MIN_SPEED]` for reverse

**Step 4.6: Reorder Columns to Match MPC State Format**

The trajectory is reordered from `[x, y, yaw, direction]` to `[x, y, v, yaw]` to match MPC's state definition where `NX = 4` and the state vector is `[x, y, v, yaw]`.

**Step 4.7: Create Cumulative Distance Parameter `u`**

The trajectory is parameterized by cumulative distance along the path:
1. Compute distances between consecutive points: `dists = [d_0, d_1, ..., d_{n-1}]`
2. Cumulative sum: `u = [0, d_0, d_0+d_1, ..., total_distance]`

**Think of `u` as a "progress bar" along the trajectory**: 
- `u = 0` at the start
- `u = total_distance` at the end
- Any value in between represents a point along the path

**Why this is useful**: 
- All "find nearest point" and "interpolate future horizon" operations work on this `u` parameter
- It's easier to interpolate along a 1D parameter (distance) than in 2D space (x, y)
- Makes it simple to query "what should the state be at distance `u` from the start?"

**Step 4.8: Speed Limit Based on Braking Distance**

When there's a point with $v = 0$ ahead (stop point or direction switch), the code enforces a speed limit based on braking distance:

**The physics**: To stop safely, you need enough distance to decelerate. The maximum speed at distance $d$ from a stop point is:
$$v_{\max} = \sqrt{2 \cdot a_{\max} \cdot d}$$

where $a_{\max}$ is the maximum deceleration.

**In practice**: The closer you are to a stop point, the lower the allowed speed (otherwise you can't brake in time). This prevents MPC from planning trajectories that would require impossible braking.

**You don't need to memorize the formula** — because I don't... Just understand: "the closer to a stop, the slower you must go."

**Step 4.9: Speed Limit Based on Curvature (Centripetal Acceleration)**

The code computes the trajectory's curvature using spline interpolation (`_get_curvature()`).

**The physics**: When turning, the vehicle experiences centripetal (lateral) acceleration:
$$a_{\text{centripetal}} = \frac{v^2}{R} = v^2 \cdot \kappa$$

where $R$ is the turning radius and $\kappa$ (kappa) is the curvature.

**The constraint**: To prevent skidding, lateral acceleration must not exceed the tire friction limit:
$$v^2 \cdot \kappa \le a_{y,\max}$$

**The speed limit**: 
$$v_{\max} = \sqrt{\frac{a_{y,\max}}{\kappa}}$$

**In simple terms**: 
- **Sharper turns** (higher curvature) → **lower maximum speed**
- This is why you slow down before tight corners in real driving
- The formula ensures MPC never plans speeds that would cause the vehicle to skid

**Step 4.10: Create Interpolatable Spline**

Finally, the entire reference trajectory is converted into a **B-spline** using `splprep(...)` and stored as `self._tck`.

**What this enables**:
- Given any distance parameter `u` (from Step 4.7), the spline can interpolate the corresponding state `[x, y, v, yaw]`
- Smooth, continuous trajectory representation
- Efficient querying: "What should the state be at progress `u` along the path?"

**Why splines?**
- **Smoothness**: Splines provide smooth interpolation between discrete waypoints
- **Efficiency**: Once constructed, querying any point is very fast
- **Flexibility**: Can easily sample the trajectory at any resolution needed for MPC

**Summary of Preprocessing Pipeline:**

1. Format validation and duplicate removal
2. Yaw smoothing (handle angle wrap-around)
3. Direction switch handling (enforce stops)
4. Direction → velocity conversion
5. Column reordering
6. Distance parameterization
7. Braking distance speed limits
8. Curvature-based speed limits
9. Spline construction

The result is a **smooth, physically feasible, MPC-friendly reference trajectory** that respects all vehicle constraints and can be efficiently queried during MPC execution.

---

#### 3. **Execution: What Happens During Runtime**

`update(state, dt)` is the entry point called every frame during execution. This is where MPC actually runs in real-time, processing the current vehicle state and producing control commands.

**Step 1. Finding the Reference Segment (`_find_xref()`)**

Before solving the optimization problem, MPC needs to extract a "reference segment" from the global trajectory that represents what the vehicle should follow over the next prediction horizon.

**Step 2. Finding the Nearest Point on Trajectory**

`_find_nearest_point()` locates the closest point on the trajectory to the current vehicle position:
- Starts from the current progress `self._cur_u` (where we were last frame)
- Uses **hill climbing** to search forward slightly, finding the "locally nearest" point
- **Why "local" search?** Prevents the algorithm from "jumping" to a distant point on the trajectory (e.g., if the vehicle is slightly off-path, we want to find the nearest point ahead, not jump to a point far behind)

**Step 3. Determining Horizon Length**

The horizon length (how far ahead to look) is computed as:
$$\text{length} = \max(\text{MIN\_HORIZON\_DISTANCE}, v \times dt \times \text{HORIZON\_LENGTH})$$

**Intuition**: 
- **Faster vehicles** → look further ahead (speed × time × steps)
- **Slower vehicles** → at least look a minimum distance ahead
- This adaptive horizon ensures MPC has enough "look-ahead" time to plan smooth maneuvers

**Step 4. Interpolating Reference States**

Using the spline created during preprocessing (Step 4.10), the code interpolates reference states at evenly spaced `u` values over the horizon:
- Input: array of `u` values (distance parameters)
- Output: `xref` with shape $(H+1) \times 4$ (e.g., $6 \times 4$ if $H=5$)
- Each row is `[x, y, v, yaw]` at that point along the trajectory

**Step 5. Handling Direction Switches in Horizon**

If the horizon range contains a direction switch point (forward ↔ reverse):

**Option 1**: Advance the starting point to the switch point (if close enough, skip past it)
- Prevents MPC from trying to plan through an impossible instant direction change

**Option 2**: Truncate the horizon and pad it, forcing the vehicle to "stop at the switch point first"
- Ensures the vehicle comes to a complete stop before changing direction
- The padded segment has zero velocity, giving MPC time to plan the stop

--- 

**Braking Logic**

If braking is needed (`self._brake` flag is set, or the vehicle has reached the goal):
- Sets reference velocity to 0 for all points in the horizon
- Optionally sets the final point to a **negative velocity** to help the vehicle brake more aggressively
- This ensures MPC plans a smooth deceleration to a complete stop

**The Goal**

The entire logic in `_find_xref()` has one goal: **ensure that every frame, the reference trajectory fed to the optimizer is reachable, reasonable, and doesn't require impossible maneuvers** (like teleportation or instant direction changes).

**Yaw Alignment (`state.align_yaw(xref[0, 3])`)**

Before computing tracking errors, the current vehicle yaw is aligned with the reference yaw:

**The problem**: Yaw angles wrap around at $\pm\pi$. If the current yaw is $+\pi$ and the reference is $-\pi$, the difference appears to be $2\pi$ (360°), which is physically the same orientation but mathematically a huge error.

**The solution**: `align_yaw()` adjusts the vehicle's yaw representation to be within $\pi$ radians of the reference yaw:
- If the difference is more than $\pi$, add or subtract $2\pi$ to bring it closer
- This ensures the tracking error calculation is meaningful

**Why this matters**: Without alignment, MPC would see a huge yaw error and try to "correct" it by spinning 360°, which is wasteful and can cause instability.

---

**Iterative Solving (Up to `MAX_ITER` iterations)**

MPC solves the optimization problem iteratively because the linearization needs to be refined around the actual solution trajectory.

**The Iterative Loop**

The process repeats up to `MAX_ITER` times:

1. **Initial guess**: Start with a control sequence `controls` (initially all zeros)
2. **Predict motion**: Use `_predict_motion(state, controls, dt)` to get `xbar` (predicted future states)
   - This provides the linearization point for the next step
3. **Solve QP**: Call `_linear_mpc_control(xref, xbar, last_steer, dt)` to solve the linearized QP
   - Returns improved `controls` and `states`
4. **Check convergence**: Compute the change in controls:
   $$du = \|\text{new\_controls} - \text{old\_controls}\|$$
   - If $du < \text{DU\_TH}$ (threshold), the solution has converged → exit loop
   - Otherwise, continue iterating

- **Linearization accuracy**: The linearized model is only accurate near the reference trajectory
- **First iteration**: Linearizes around the zero-control prediction (may be far from optimal)
- **Subsequent iterations**: Linearizes around the improved trajectory from the previous iteration
- **Convergence**: Usually converges in 1-3 iterations because each iteration refines the linearization point

---

**Output: MPCResult**

After convergence (or reaching `MAX_ITER`), the function returns an `MPCResult` containing:

- **`controls`**: Future $H$ steps of `[accel, steer]` pairs (shape $(H \times 2)$)
  - Only the **first** control (`controls[0]`) is actually executed
  - The rest are planned but discarded (receding horizon principle)

- **`states`**: Predicted future states (shape $(H+1) \times 4$)
  - Reordered to a format suitable for visualization: `[x, y, yaw, v]`
  - Shows where the vehicle is predicted to go if these controls are applied

- **`ref_states`**: Corresponding reference points from the trajectory
  - Same shape as `states`
  - Used for comparison and visualization

- **`brake_trajectory`**: If braking was triggered, this contains the braking reference trajectory
  - Used for visualization and debugging

---

**The Complete Execution Flow**

**Every frame (every `dt` seconds):**
1. **Extract reference segment** (`_find_xref()`)
   - Find where we are on the trajectory
   - Extract the next $H$ steps worth of reference states
   - Handle edge cases (direction switches, braking, goal arrival)

2. **Align yaw** (`state.align_yaw()`)
   - Ensure yaw error calculation is meaningful

3. **Iterative optimization** (up to `MAX_ITER` times)
   - Predict motion with current guess
   - Linearize around prediction
   - Solve QP
   - Check convergence
   - Repeat if needed

4. **Return result**
   - Extract first control action
   - Execute it on the vehicle
   - Discard the rest (will be recomputed next frame)

**Why this works**: By recomputing the entire plan every frame, MPC can:
- **React to disturbances**: If the vehicle drifts off-path, the next frame's optimization corrects it
- **Handle uncertainty**: Model errors and external disturbances are naturally compensated
- **Maintain safety**: Constraints are re-evaluated every frame, ensuring feasibility

### **MPC Implementation**


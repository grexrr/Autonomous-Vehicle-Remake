# Autonomous Vehicle (Individual Rework)



```
# 预处理：得到占据栅格和 Dijkstra 距离场 D
grid = obstacles.downsampling_to_grid(...)
D = _distance_field(grid, goal_xy)

# 初始化起点节点（空段/零段）
start_sp = SimplePath(ijk=离散(start), trajectory=np.array([[x0,y0,yaw0]]),
                      direction=0, steer=0.0)
start = Node(path=start_sp, cost=0.0, h_cost = H_COST * D[i0,j0], parent=None)
open = 小根堆 [start]
best_g = 字典：key=(i,j,k) -> 已知最小 g

while open 非空:
    cur = heappop(open)  # f 最小的节点

    if 到达目标附近 or 可解析直连:
        # 若解析直连成功，可把 RSPath 包成 Node 做最后一个孩子，h=0
        回溯 parent 链条，拼成最终路径
        break

    # 扩展：试一批 motion primitives（dir in {+1,-1} × steer in STEER_SET）
    for dir, steer in 动作集合:
        # 用 Car.update 连续滚动一小段（例如 3 米），段内每 0.5 米检查碰撞
        traj, collided = rollout(cur.path.end_state, dir, steer, obstacles)
        if collided: continue

        child_sp = SimplePath(ijk=离散(段末端), trajectory=traj, direction=dir, steer=steer)
        g2 = cur.cost + edge_cost(cur, child_sp)  # 段长 + 换向/转角/转角变化惩罚
        key = child_sp.ijk
        if key in best_g and g2 >= best_g[key]:  # 劣解剪枝
            continue

        h2 = H_COST * D[i,j]  # 用段末端的格子取启发式
        child = Node(path=child_sp, cost=g2, h_cost=h2, parent=cur)
        best_g[key] = g2
        heappush(open, child)

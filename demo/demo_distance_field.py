# examples/demo_distance_field.py
import numpy as np, matplotlib.pyplot as plt
from modeling.obstacles import Obstacles
from modeling.car import Car
from global_planner.hybrid_a_star import _heuristic_distance_field

def make_test_obstacles():
    pts=[]
    for x in np.linspace(0,60,121): pts+= [(x,0),(x,60)]
    for y in np.linspace(0,60,121): pts+= [(0,y),(60,y)]
    for y in np.linspace(10,50,81): pts.append((20,y))
    for y in np.linspace(10,50,81): pts.append((40,y))
    return Obstacles(np.array(pts,float))

obs = make_test_obstacles()
RES = 2.0
r = min(Car.COLLISION_LENGTH, Car.COLLISION_WIDTH)/2  
grid = obs.downsampling_to_grid(RES, r)

goal = (55.0, 55.0)
heur = _heuristic_distance_field(grid, goal)

# —— 可视化距离场（更清晰的版本）——
fig, ax = plt.subplots()

# 1) 先算网格中心坐标轴（和 distance 的 (i,j) 对应）
xs = np.linspace(grid.minx + RES/2, grid.maxx - RES/2, grid.grid.shape[1])
ys = np.linspace(grid.miny + RES/2, grid.maxy - RES/2, grid.grid.shape[0])

# 2) 自检：目标是否在可行格？距离场是否有有限值？
gi, gj = grid.calc_index(goal)
print("goal_cell:", (gi, gj), "inside_obstacle?", bool(grid.grid[gi, gj]))
finite_mask = np.isfinite(heur)
print("finite cells:", int(finite_mask.sum()), "of", heur.size)

# 3) 距离场热力图（无穷大用掩码隐藏）
H = np.ma.masked_where(~finite_mask, heur)
im = ax.imshow(H, origin='lower',
               extent=[xs[0], xs[-1], ys[0], ys[-1]],
               cmap='viridis', alpha=0.95)
cb = plt.colorbar(im, ax=ax)
cb.set_label('grid-distance to goal (in grid steps)')

# 4) 加几条白色等高线，线宽加粗一点，级数少一点更醒目
finite_vals = np.where(finite_mask, heur, np.nan)
vmin = np.nanmin(finite_vals); vmax = np.nanmax(finite_vals)
levels = np.linspace(vmin, vmax, 10)  # 10 条线就够
cs = ax.contour(xs, ys, finite_vals, levels=levels, colors='w', linewidths=0.8, alpha=0.8)

# 5) 最后再把占据栅格（障碍）盖到上面（半透明黑）
ax.imshow(grid.grid, origin='lower',
          extent=[grid.minx, grid.maxx, grid.miny, grid.maxy],
          cmap='gray', alpha=0.35)

# 6) 画目标点
ax.scatter([goal[0]], [goal[1]], c='r', s=30, zorder=5)

ax.set_title("Dijkstra Distance Field (heatmap + contours)")
ax.set_aspect('equal', 'box')
plt.show()


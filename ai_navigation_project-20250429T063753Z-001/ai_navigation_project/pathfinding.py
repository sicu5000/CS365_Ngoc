import numpy as np

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(pos, width, height):
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    return [(pos[0]+dx, pos[1]+dy) for dx,dy in directions if 0 <= pos[0]+dx < width and 0 <= pos[1]+dy < height]

def check_collision(pos, fixed, moving):
    for obs in fixed:
        if np.linalg.norm(pos - obs) < 1: return True
    for obs in moving:
        if np.linalg.norm(pos - obs[:2]) < 1: return True
    return False

def reconstruct_path(came_from, current):
    path = [current]
    while tuple(current) in came_from:
        current = came_from[tuple(current)]
        path.append(current)
    return path[::-1]

def hybrid_astar(start, goal, fixed, moving, width, height):
    start, goal = tuple(start), tuple(goal)
    open_set = [(heuristic(start, goal), 0, start)]
    came_from, g_score = {}, {start: 0}
    while open_set:
        _, g, current = min(open_set); open_set.remove((_, g, current))
        if heuristic(current, goal) < 1: return np.array(reconstruct_path(came_from, current))
        for neighbor in get_neighbors(current, width, height):
            if check_collision(np.array(neighbor), fixed, moving): continue
            temp_g = g + heuristic(current, neighbor)
            if neighbor not in g_score or temp_g < g_score[neighbor]:
                g_score[neighbor] = temp_g
                f = temp_g + heuristic(neighbor, goal)
                came_from[neighbor] = current
                open_set.append((f, temp_g, neighbor))
    return np.array([])

def move_obstacles(moving, width, height):
    for obs in moving:
        obs[0] += obs[2]; obs[1] += obs[3]
        if obs[0] < 0 or obs[0] >= width: obs[2] *= -1
        if obs[1] < 0 or obs[1] >= height: obs[3] *= -1
    return moving

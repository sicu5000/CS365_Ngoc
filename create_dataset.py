# create_dataset.py
import numpy as np
import pandas as pd
from pathfinding import hybrid_astar, move_obstacles

def create_environment(width, height, num_obstacles):
    fixed = np.random.rand(num_obstacles, 2) * [width, height]
    moving = np.hstack((np.random.rand(num_obstacles, 2) * [width, height],
                        (np.random.rand(num_obstacles, 2) - 0.5) * 2))
    return fixed, moving

def generate_dataset(fixed_obstacles, moving_obstacles, start, goal, num_steps, run_index, width, height):
    data = []
    current_position = np.array(start)
    for _ in range(num_steps):
        path = hybrid_astar(current_position, goal, fixed_obstacles, moving_obstacles, width, height)
        if len(path) == 0:
            break
        for point in path:
            if not np.allclose(point, current_position):
                row = [run_index, *start, *current_position, *point]
                data.append(row)
                current_position = point
                break
        moving_obstacles = move_obstacles(moving_obstacles, width, height)
        if np.linalg.norm(current_position - goal) < 1:
            break
    return np.array(data)

def main():
    width, height = 10, 10
    num_obstacles = 5
    num_steps = 100
    num_runs = 1000
    all_data = []

    for run in range(1, num_runs + 1):
        start = np.random.rand(2) * [width, height]
        goal = np.random.rand(2) * [width, height]
        while np.allclose(start, goal):
            goal = np.random.rand(2) * [width, height]

        fixed, moving = create_environment(width, height, num_obstacles)
        run_data = generate_dataset(fixed, moving, start, goal, num_steps, run, width, height)
        if run_data.size > 0:
            all_data.append(run_data)

    all_data = np.vstack(all_data)
    columns = ['Run_Index', 'Start_X', 'Start_Y', 'Current_X', 'Current_Y', 'Next_X', 'Next_Y']
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv("dataset.csv", index=False)
    print("Complete data generation and save to dataset.csv.")

if __name__ == '__main__':
    main()

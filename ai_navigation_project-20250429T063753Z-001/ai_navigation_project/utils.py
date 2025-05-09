# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from grid_gui import interactive_grid_setup
from pathfinding import hybrid_astar, move_obstacles
import matplotlib.colors as mcolors


def normalize_data(df):
    return (df - df.mean()) / df.std()


def behavioral_cloning(data):
    X = data[["Start_X", "Start_Y"]]
    Y_X = data["Next_X"]
    Y_Y = data["Next_Y"]
    X_train, X_test, Y_train_X, Y_test_X = train_test_split(X, Y_X, test_size=0.2, random_state=42)
    _, _, Y_train_Y, Y_test_Y = train_test_split(X, Y_Y, test_size=0.2, random_state=42)

    model_X = LinearRegression().fit(X_train, Y_train_X)
    model_Y = LinearRegression().fit(X_train, Y_train_Y)

    Y_pred_X = model_X.predict(X_test)
    Y_pred_Y = model_Y.predict(X_test)

    mse_X = mean_squared_error(Y_test_X, Y_pred_X)
    r2_X = r2_score(Y_test_X, Y_pred_X)
    mse_Y = mean_squared_error(Y_test_Y, Y_pred_Y)
    r2_Y = r2_score(Y_test_Y, Y_pred_Y)

    print("Behavioral Cloning Evaluation:")
    print(f"  MSE (Next_X): {mse_X:.4f}, R2 (Next_X): {r2_X:.4f}")
    print(f"  MSE (Next_Y): {mse_Y:.4f}, R2 (Next_Y): {r2_Y:.4f}")

    return {
        "bc": {
            "mse_X": mse_X,
            "r2_X": r2_X,
            "mse_Y": mse_Y,
            "r2_Y": r2_Y,
            "Y_test_X": Y_test_X.to_numpy(),
            "Y_pred_X": Y_pred_X,
            "Y_test_Y": Y_test_Y.to_numpy(),
            "Y_pred_Y": Y_pred_Y,
        }
    }


def dagger(data, width, height, num_obstacles, num_steps, num_runs):
    all_data = data.copy()
    for _ in range(5):
        new_data = data.sample(frac=0.1, replace=True)
        all_data = pd.concat([all_data, new_data], ignore_index=True)

    result = behavioral_cloning(all_data)
    print("DAgger Evaluation (using BC final model):")
    print(f"  MSE (Next_X): {result['bc']['mse_X']:.4f}, R2 (Next_X): {result['bc']['r2_X']:.4f}")
    print(f"  MSE (Next_Y): {result['bc']['mse_Y']:.4f}, R2 (Next_Y): {result['bc']['r2_Y']:.4f}")
    return result


def simulate_movement(start, goal, fixed_obstacles, moving_obstacles, width, height, realtime=False):
    print("Simulate moving the agent from start to goal.")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Fixed Obstacles: {len(fixed_obstacles)}, Moving Obstacles: {len(moving_obstacles)}")

    if not realtime:
        path = hybrid_astar(start, goal, np.array(fixed_obstacles), np.array(moving_obstacles), width, height)
        fig, ax = plt.subplots()
        if len(path) > 0:
            ax.plot(*zip(*path), color='blue', linestyle='-', linewidth=2, label="Agent Path")
        ax.plot(start[0], start[1], 'go', label="Start")
        ax.plot(goal[0], goal[1], 'bo', label="Goal")
        if fixed_obstacles:
            ax.scatter(*zip(*fixed_obstacles), c='black', marker='s', label="Fixed Obstacles")
        if moving_obstacles:
            paths = [[obs[:2]] for obs in moving_obstacles]
            for _ in range(100):
                moving_obstacles = move_obstacles(moving_obstacles, width, height)
                for i, obs in enumerate(moving_obstacles):
                    paths[i].append(obs[:2].copy())
            colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
            for i, path in enumerate(paths):
                path = np.array(path)
                color = colors[i % len(colors)]
                ax.plot(path[:, 0], path[:, 1], linestyle='--', linewidth=1.5, color=color,
                        label=f"Obstacle {i+1} Path" if i < 6 else None)
            ax.scatter(*zip(*[o[:2] for o in moving_obstacles]), c='red', marker='x', label="Moving Obstacles")
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title("Agent and Obstacle Movement")
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots()
        agent = np.array(start)
        goal = np.array(goal)
        agent_path = [agent.copy()]
        obstacle_paths = [[obs[:2].copy()] for obs in moving_obstacles]
        for step in range(100):
            ax.clear()
            ax.plot(goal[0], goal[1], 'bo', label="Goal")
            ax.plot(agent[0], agent[1], 'go', label="Agent")
            agent_path_arr = np.array(agent_path)
            if len(agent_path_arr) > 1:
                ax.plot(agent_path_arr[:, 0], agent_path_arr[:, 1], color='blue', linewidth=2, label="Agent Path")
            if fixed_obstacles:
                ax.scatter(*zip(*fixed_obstacles), c='black', marker='s', label="Fixed Obstacles")
            if moving_obstacles:
                for i, obs in enumerate(moving_obstacles):
                    obstacle_paths[i].append(obs[:2].copy())
                for i, path in enumerate(obstacle_paths):
                    path = np.array(path)
                    color = f"C{i % 10}"
                    ax.plot(path[:, 0], path[:, 1], linestyle='--', linewidth=1.0, color=color, label=f"Obstacle {i+1} Path" if step == 0 else None)
                ax.scatter(*zip(*[o[:2] for o in moving_obstacles]), c='red', marker='x', label="Moving Obstacles")
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.grid(True)
            ax.legend(loc='upper left')
            plt.title(f"Real-time Movement Step {step+1}")
            plt.pause(0.1)
            path = hybrid_astar(agent, goal, np.array(fixed_obstacles), np.array(moving_obstacles), width, height)
            if len(path) > 1:
                agent = path[1]
                agent_path.append(agent.copy())
            moving_obstacles = move_obstacles(moving_obstacles, width, height)
            if np.linalg.norm(agent - goal) < 1:
                break
        plt.show()

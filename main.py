# main.py
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    normalize_data,
    behavioral_cloning,
    dagger,
    interactive_grid_setup,
    simulate_movement,
)

def main():
    data = pd.read_csv("dataset.csv")
    print("First 10 rows of the dataset:")
    print(data.head(10))

    data.iloc[:, 1:] = normalize_data(data.iloc[:, 1:])

    algorithm = int(input("Select algorithm (1: Behavioral Cloning, 2: DAgger): "))
    realtime = input("Want to simulate real-time motion with path drawing? (y/n): ").strip().lower() == 'y'

    width, height = 16, 16
    start, goal, fixed_obstacles, moving_obstacles = interactive_grid_setup(width, height)

    if algorithm == 1:
        results = behavioral_cloning(data)
    elif algorithm == 2:
        results = dagger(data, width, height, len(fixed_obstacles), num_steps=100, num_runs=100)
    else:
        print("Lựa chọn không hợp lệ.")
        return

    for key in results:
        if results[key]["Y_test_X"] is not None:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.plot(results[key]["Y_test_X"], label="Actual")
            plt.plot(results[key]["Y_pred_X"], label="Predicted")
            plt.title(f"{key}: Next_X")
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.scatter(results[key]["Y_test_X"], results[key]["Y_pred_X"])
            plt.title(f"{key}: Next_X Scatter")

            plt.subplot(2, 2, 3)
            plt.plot(results[key]["Y_test_Y"], label="Actual")
            plt.plot(results[key]["Y_pred_Y"], label="Predicted")
            plt.title(f"{key}: Next_Y")
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.scatter(results[key]["Y_test_Y"], results[key]["Y_pred_Y"])
            plt.title(f"{key}: Next_Y Scatter")
            plt.tight_layout()
            plt.show()

    simulate_movement(start, goal, fixed_obstacles, moving_obstacles, width, height, realtime=realtime)

if __name__ == "__main__":
    main()

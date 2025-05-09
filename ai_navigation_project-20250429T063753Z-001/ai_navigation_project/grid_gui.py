import tkinter as tk
from tkinter import messagebox
import numpy as np

class GridApp:
    def __init__(self, width=16, height=16, cell_size=30):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.start = None
        self.goal = None
        self.fixed_obstacles = []
        self.moving_obstacles = []
        self.mode = 'start'

        self.root = tk.Tk()
        self.root.title("Interactive Grid Setup")
        self.canvas = tk.Canvas(self.root, width=self.width * self.cell_size,
                                height=self.height * self.cell_size, bg="white")
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_grid()

        button_frame = tk.Frame(self.root)
        button_frame.pack()
        tk.Button(button_frame, text="Set Goal", command=self.set_goal_mode).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Fixed Obstacle", command=self.set_fixed_mode).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Moving Obstacle", command=self.set_moving_mode).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Done", command=self.done).pack(side=tk.LEFT)

    def draw_grid(self):
        for i in range(self.width):
            for j in range(self.height):
                x0 = i * self.cell_size
                y0 = j * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="gray")

    def on_click(self, event):
        i = event.x // self.cell_size
        j = event.y // self.cell_size
        if self.mode == 'start':
            self.start = [i, j]
            self.canvas.create_oval(i * self.cell_size, j * self.cell_size,
                                    (i + 1) * self.cell_size, (j + 1) * self.cell_size,
                                    fill='green')
        elif self.mode == 'goal':
            self.goal = [i, j]
            self.canvas.create_oval(i * self.cell_size, j * self.cell_size,
                                    (i + 1) * self.cell_size, (j + 1) * self.cell_size,
                                    fill='blue')
        elif self.mode == 'fixed':
            self.fixed_obstacles.append([i, j])
            self.canvas.create_rectangle(i * self.cell_size, j * self.cell_size,
                                         (i + 1) * self.cell_size, (j + 1) * self.cell_size,
                                         fill='black')
        elif self.mode == 'moving':
            vx, vy = (np.random.rand(2) - 0.5) * 2
            self.moving_obstacles.append([i, j, vx, vy])
            self.canvas.create_rectangle(i * self.cell_size, j * self.cell_size,
                                         (i + 1) * self.cell_size, (j + 1) * self.cell_size,
                                         fill='red')

    def set_goal_mode(self): self.mode = 'goal'
    def set_fixed_mode(self): self.mode = 'fixed'
    def set_moving_mode(self): self.mode = 'moving'

    def done(self):
        if self.start is None or self.goal is None:
            messagebox.showwarning("Warning", "Please set start and goal points.")
            return
        self.root.quit()
        self.root.destroy()

    def launch(self):
        self.root.mainloop()
        return self.start, self.goal, self.fixed_obstacles, self.moving_obstacles


def interactive_grid_setup(width, height):
    app = GridApp(width, height)
    return app.launch()

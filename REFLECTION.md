# Reflections ✨

## 🚀 **What Worked Well**

* The integration of `Hybrid A*` with Behavioral Cloning (BC) and DAgger for pathfinding worked effectively for generating reliable training data.
* The use of `grid_gui.py` for interactive map setup provided an intuitive way to configure start, goal, and obstacles, making testing and experimentation easier.
* Real-time simulation visualization using `matplotlib` allowed for clear inspection of the agent's decision-making process.

---

## 🧐 **Challenges Faced**

* Behavioral Cloning sometimes underperformed when the agent diverged from the A\* path, highlighting the issue of **covariate shift**.
* Generating large datasets took time, especially when many obstacles were present, slowing down `create_dataset.py`.
* The current implementation of `Hybrid A*` is effective but can be computationally expensive for very large grids.

---

## 🔍 **Lessons Learned**

* **Data Preprocessing Matters**: Normalization significantly improved model training consistency.
* **Expert Data is Key**: The quality of A\* paths directly impacted the model's ability to imitate.
* **DAgger Improves Robustness**: Retraining on the agent's own mistakes made it more reliable in unfamiliar states.

---

## 📌 **Future Directions**

* Implementing a **neural network-based planner** instead of linear regression for more complex path prediction.
* Replacing `Hybrid A*` with the trained model for live simulation to improve speed.
* Experimenting with larger maps and more dynamic obstacles.

---

## 🤝 **Acknowledgments**

Special thanks to the concepts from imitation learning and pathfinding research, which inspired the architecture of this project.

# KVGCE HACKWISE 2025
# Problem Statement - 3
# Traveling Salesman Problem (TSP) Solver

This project provides an efficient solution to the Traveling Salesman Problem (TSP) in **3D space**, optimizing paths using **Euclidean** and **Minkowski** distance metrics. The solver leverages the **Held-Karp dynamic programming algorithm** with **branch-and-bound pruning** to compute exact optimal tours for **5 to 15 waypoints**, while meeting strict performance constraints (e.g., <5s for N=15).

---

## ✨ Features

- **Input Processing**
  - Reads waypoints from `waypoints.txt` in the format:  
    `id x y z`  
  - Validates:
    - Unique integer IDs (1 to N)
    - Coordinates within `[-1000, 1000]`

- **Distance Metrics**
  - **Euclidean Distance**: `p=2.0` (default for hackathon scoring)
  - **Minkowski Distance**: `p = 1.0, 3.0, 4.0, 6.0, 8.0, 10.0, 20.0, 50.0, 100.0`

- **Optimization**
  - **Held-Karp Algorithm**: `O(N^2 * 2^N)` complexity
  - Enhanced with:
    - Branch-and-bound pruning
    - LRU caching for Minkowski calculations

- **Output**
  - Saves best Minkowski path to `path.txt` in format:  
    `id1 id2 ... idN id1 cost`
  - Summary of all paths and costs in `results.txt`
  - 3D path visualizations saved as PNG in `sample_output/process_results/`

- **Performance Monitoring**
  - Tracks execution time and memory (e.g., time <10s, memory <512MB)

- **Validation**
  - Ensures correctness of input, path, and cost

---

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Idea-2-Implement/kvgce-hackwise-problem3
cd kvgce-hackwise-problem3
```

## To test the solution
- go the
```bash
cd ./sample_input/waypoints.txt
```
change the values in the `waypoints.txt`

## 🧪 How to Test the Solution

1. Navigate to the `sample_input` directory:
   ```bash
   cd sample_input
    ```

2. Open the `waypoints.txt` file:

- Edit the file using your preferred text editor
- Modify the waypoint values as needed to test different inputs.
- Save the file and run the solution to observe the changes.
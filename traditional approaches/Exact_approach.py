# Standard library imports for mathematical operations, timing, and file handling
import math  # For sqrt in Euclidean distance calculations
import time  # For measuring total execution time of the TSP solver
import psutil  # For tracking CPU time and memory usage
import os  # For file path handling (e.g., reading waypoints.txt, writing path.txt)

# Itertools for generating permutations in brute-force TSP
from itertools import permutations  # For generating all possible paths in tsp_brute (N ≤ 10)

# Third-party imports for 3D visualization
import matplotlib.pyplot as plt  # For creating and displaying 3D path plots
from mpl_toolkits.mplot3d import Axes3D  # For 3D axis support in visualizations

# ------------------------------
# Start timing and resource tracking
# ------------------------------
# Record start time to measure total execution duration
start_time = time.time()
# Initialize process monitoring for CPU and memory usage
process = psutil.Process(os.getpid())

# ------------------------------
# Step 1: Load waypoints.txt
# ------------------------------
# Define input file path (modify as needed for different environments)
filepath = "sample input/waypoints.txt"  # ✅ Change this as needed
waypoints = []
# Read waypoints from file, expecting format: id x y z per line
with open(filepath, "r") as f:
    for line in f:
        line = line.strip()
        if line:  # Only process non-empty lines
            id_, x, y, z = map(float, line.split())
            waypoints.append((int(id_), x, y, z))

# Store number of waypoints for use in distance matrix and TSP
N = len(waypoints)

# ------------------------------
# Step 2: Build Distance Matrix
# ------------------------------
# Initialize NxN distance matrix with zeros
dist = [[0.0]*N for _ in range(N)]
# Compute Euclidean distances between all pairs of waypoints
for i in range(N):
    for j in range(N):
        # Extract coordinates of waypoints i and j
        _, x1, y1, z1 = waypoints[i]
        _, x2, y2, z2 = waypoints[j]
        # Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
        dist[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# ------------------------------
# Step 3: Brute Force TSP (for N ≤ 10)
# ------------------------------
def tsp_brute() -> tuple[list[int], float]:
    """Solve TSP using brute-force enumeration for N ≤ 10.

    Generates all permutations of waypoints (excluding start) and computes the cost
    of each closed tour starting and ending at waypoint 0.

    Returns:
        Tuple of (optimal path as list of indices, total cost of path).
    """
    # Initialize minimum cost and best path
    min_cost = float('inf')
    best_path = []
    # Iterate over all permutations of waypoints 1 to N-1
    for perm in permutations(range(1, N)):
        # Construct path: start at 0, visit permutation, return to 0
        path = [0] + list(perm) + [0]
        # Compute total cost by summing distances between consecutive waypoints
        cost = sum(dist[path[i]][path[i+1]] for i in range(len(path)-1))
        # Update best path if current cost is lower
        if cost < min_cost:
            min_cost = cost
            best_path = path
    return best_path, min_cost

# ------------------------------
# Step 4: Nearest Neighbor TSP (for N > 10)
# ------------------------------
def tsp_nearest_neighbor() -> tuple[list[int], float]:
    """Solve TSP using nearest-neighbor heuristic for N > 10.

    Starts at waypoint 0, greedily selects the closest unvisited waypoint,
    and returns to 0 to complete the tour.

    Returns:
        Tuple of (path as list of indices, total cost of path).
    """
    # Initialize set of unvisited waypoints (excluding start)
    unvisited = set(range(1, N))
    # Start path at waypoint 0
    path = [0]
    current = 0
    # Continue until all waypoints are visited
    while unvisited:
        # Find closest unvisited waypoint to current
        next_node = min(unvisited, key=lambda x: dist[current][x])
        # Add to path and remove from unvisited
        path.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    # Complete tour by returning to start
    path.append(0)
    # Compute total cost by summing distances between consecutive waypoints
    cost = sum(dist[path[i]][path[i+1]] for i in range(len(path)-1))
    return path, cost

# ------------------------------
# Step 5: Solve TSP with appropriate algorithm
# ------------------------------
# Choose algorithm based on number of waypoints
if N <= 10:
    # Use brute-force for small instances to ensure optimality
    path, cost = tsp_brute()
else:
    # Use nearest-neighbor heuristic for larger instances to ensure scalability
    path, cost = tsp_nearest_neighbor()

# ------------------------------
# Step 6: Output path to path.txt
# ------------------------------
# Convert path indices to 1-based waypoint IDs
path_ids = [waypoints[i][0] for i in path]
# Format output: space-separated IDs followed by cost (rounded to 2 decimals)
output_str = " ".join(map(str, path_ids)) + f" {cost:.2f}"
# Write output to path.txt
with open("traditional approaches/Exact_approach_results/path.txt", "w") as f:
    f.write(output_str + "\n")

# Print final path for verification
print("✅ Final Path Output:")
print(output_str)

# ------------------------------
# Step 7: 3D Visualization
# ------------------------------
# Initialize 3D plot with specified figure size
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Extract x, y, z coordinates for path waypoints
xs = [waypoints[i][1] for i in path]
ys = [waypoints[i][2] for i in path]
zs = [waypoints[i][3] for i in path]

# Plot path as a line with waypoint markers
ax.plot(xs, ys, zs, marker='o')
# Add ID labels to each waypoint
for i in path:
    id_, x, y, z = waypoints[i]
    ax.text(x, y, z, str(id_), size=10)

# Set axis labels and plot title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Space Path Optimization - TSP in 3D")
# Display the plot
plt.show()

# ------------------------------
# Step 8: Enhanced Segment Breakdown
# ------------------------------
# Print detailed table of path segments
print("\n📍 Optimized Visit Sequence & Segment Details:")
print("-" * 60)
print(f"{'Step':<5} {'From':<5} {'To':<5} {'Distance':<10} {'To Coordinates (x,y,z)'}")
print("-" * 60)

# Calculate and display segment details
total_cost = 0
for i in range(len(path) - 1):
    # Get indices of current and next waypoints
    a, b = path[i], path[i+1]
    # Get distance between waypoints
    dist_ = dist[a][b]
    # Accumulate total cost
    total_cost += dist_
    # Get coordinates of destination waypoint
    x, y, z = waypoints[b][1], waypoints[b][2], waypoints[b][3]
    # Print segment details
    print(f"{i+1:<5} {waypoints[a][0]:<5} {waypoints[b][0]:<5} {dist_:.2f}     ({x:.2f}, {y:.2f}, {z:.2f})")

print("-" * 60)
# Print total cost and path for verification
print(f"🛢  Total Fuel Cost: {total_cost:.2f}")
print(f"📄 Path Output Format: {' '.join(map(str, path_ids))} {total_cost:.2f}")
print("-" * 60)

# ------------------------------
# Step 9: Performance Summary
# ------------------------------
# Calculate final execution time
end_time = time.time()
# Calculate total CPU time (user + system)
cpu_time = process.cpu_times().user + process.cpu_times().system
# Calculate memory usage in MB
memory_used_mb = process.memory_info().rss / 1024**2  # in MB

# Print performance metrics
print("⚙  Performance Summary")
print("-" * 60)
print(f"🕒 Execution Time  : {end_time - start_time:.4f} seconds")
print(f"🧠 Memory Used     : {memory_used_mb:.2f} MB")
print(f"🧮 CPU Time Used   : {cpu_time:.4f} seconds")
print("-" * 60)
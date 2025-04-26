# Standard library imports for mathematical operations, timing, and system resources
import math  # For sqrt in Euclidean distance calculations
import time  # For measuring execution time of TSP and visualization steps
import psutil  # For monitoring memory usage to report performance metrics
import os  # For file path handling (e.g., reading waypoints.txt, writing path.txt)

# Third-party imports for numerical computations and visualization
import matplotlib.pyplot as plt  # For creating and displaying 3D path plots
from mpl_toolkits.mplot3d import Axes3D  # For 3D axis support in visualizations
import numpy as np  # For potential array operations (unused here but included for extensibility)


def read_waypoints(file_path: str) -> list[tuple[int, float, float, float]]:
    """Read and validate waypoints from a file.

    Args:
        file_path: Path to waypoints file (format: id x y z per line).

    Returns:
        List of tuples (id, x, y, z) representing waypoints.

    Raises:
        FileNotFoundError: If waypoints.txt is not found.
        ValueError: If file format, waypoint count, IDs, or coordinates are invalid.
    """
    waypoints = []
    try:
        # Read and parse each line from the file
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty lines or lines with just whitespace
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                # Validate line format: exactly 4 components (id, x, y, z)
                # Validate line format: exactly 4 components (id, x, y, z)
                if len(parts) != 4:
                    raise ValueError("Invalid waypoint format")
                # Convert to floats, ensuring ID is a positive integer
                id, x, y, z = map(float, parts)
                if not id.is_integer() or id < 1:
                    raise ValueError("Invalid waypoint ID")
                waypoints.append((int(id), x, y, z))
        
        # Validate number of waypoints (5 to 15 per problem constraints)
        N = len(waypoints)
        if N < 5 or N > 15:
            raise ValueError("Number of waypoints must be between 5 and 15")
        # Ensure unique, consecutive IDs from 1 to N
        ids = set(wp[0] for wp in waypoints)
        if len(ids) != N or max(ids) > N or min(ids) < 1:
            raise ValueError("Waypoint IDs must be unique and in range 1 to N")
        # Validate coordinate bounds: [-1000, 1000]
        for _, x, y, z in waypoints:
            if not (-1000 <= x <= 1000 and -1000 <= y <= 1000 and -1000 <= z <= 1000):
                raise ValueError("Coordinates out of bounds")
        return waypoints
    except FileNotFoundError:
        raise FileNotFoundError("waypoints.txt not found")
    except Exception as e:
        raise ValueError(f"Error reading waypoints: {str(e)}")

def compute_distance_matrix(waypoints: list[tuple[int, float, float, float]]) -> list[list[float]]:
    """Compute Euclidean distance matrix for waypoints.

    Args:
        waypoints: List of waypoints (id, x, y, z).

    Returns:
        NxN list of lists containing distances between waypoints (0 on diagonal).
    """
    N = len(waypoints)
    # Initialize NxN distance matrix with zeros
    dist = [[0.0] * N for _ in range(N)]
    # Compute Euclidean distances for each pair of waypoints
    for i in range(N):
        for j in range(N):
            if i != j:
                # Extract coordinates of waypoints i and j
                _, x1, y1, z1 = waypoints[i]
                _, x2, y2, z2 = waypoints[j]
                # Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
                dist[i][j] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist

def solve_tsp(dist: list[list[float]], N: int) -> tuple[list[int], float]:
    """Solve TSP using dynamic programming with bitmasking.

    Uses Held-Karp algorithm to find the exact optimal tour in O(N^2 * 2^N) time.
    Tracks visited waypoints with a bitmask and reconstructs the path.

    Args:
        dist: NxN distance matrix.
        N: Number of waypoints.

    Returns:
        Tuple of (path as list of indices, total cost of path).
    """
    # Initialize DP table: dp[u][mask] = min cost to reach city u with visited mask
    dp = [[float('inf')] * (1 << N) for _ in range(N)]
    # Initialize parent table to reconstruct path
    parent = [[-1] * (1 << N) for _ in range(N)]
    
    # Base case: Start at city 0 (waypoint ID 1), mask = 1 (only city 0 visited)
    dp[0][1] = 0  # Bit 0 is set for starting city
    
    # Fill DP table: iterate over all subsets of visited cities
    for mask in range(1 << N):
        for u in range(N):
            if dp[u][mask] == float('inf'):
                continue  # Skip unreachable states
            # Try visiting each unvisited city v
            for v in range(N):
                if not (mask & (1 << v)):  # If v is not visited
                    new_mask = mask | (1 << v)  # Mark v as visited
                    # Update cost if path through u to v is shorter
                    if dp[v][new_mask] > dp[u][mask] + dist[u][v]:
                        dp[v][new_mask] = dp[u][mask] + dist[u][v]
                        parent[v][new_mask] = u
    
    # Find minimum cost to complete tour by returning to start
    min_cost = float('inf')
    last_city = -1
    final_mask = (1 << N) - 1  # All cities visited
    for u in range(1, N):
        cost = dp[u][final_mask] + dist[u][0]
        if cost < min_cost:
            min_cost = cost
            last_city = u
    
    # Reconstruct path from parent pointers
    path = []
    mask = final_mask
    current = last_city
    while current != -1:
        path.append(current)
        next_current = parent[current][mask]
        mask ^= (1 << current)  # Remove current city from mask
        current = next_current
    path = path[::-1]  # Reverse to get path from start
    path.append(0)  # Complete tour by returning to start
    
    return path, min_cost

def explain_path(waypoints: list[tuple[int, float, float, float]], path: list[int], dist: list[list[float]]) -> float:
    """Print detailed explanation of the chosen path and its cost.

    Displays segment details, coordinates, and Euclidean distance calculations.

    Args:
        waypoints: List of waypoints (id, x, y, z).
        path: List of waypoint indices forming the path.
        dist: NxN distance matrix.

    Returns:
        Total fuel cost of the path.
    """
    print("\nLogic Behind Chosen Path:")
    print("The path was chosen as it minimizes the total Euclidean distance (fuel cost) using dynamic programming.")
    print("Path Details:")
    total_cost = 0.0
    # Iterate over consecutive pairs in the path
    for i in range(len(path) - 1):
        start_idx = path[i]
        end_idx = path[i + 1]
        # Extract IDs and coordinates
        start_id, x1, y1, z1 = waypoints[start_idx]
        end_id, x2, y2, z2 = waypoints[end_idx]
        distance = dist[start_idx][end_idx]
        total_cost += distance
        # Print segment details
        print(f"Segment {start_id} -> {end_id}:")
        print(f"  Coordinates: ({x1}, {y1}, {z1}) -> ({x2}, {y2}, {z2})")
        print(f"  Euclidean Distance = sqrt(({x2-x1})^2 + ({y2-y1})^2 + ({z2-z1})^2) = {distance:.2f}")
    print(f"Total Fuel Cost: {total_cost:.2f}")
    return total_cost

def visualize_path(waypoints: list[tuple[int, float, float, float]], path: list[int], fuel_cost: float) -> None:
    """Generate and save a 3D visualization of the TSP path.

    Plots waypoints as points, connects them with lines, and labels with IDs.
    Saves the plot as a PNG file.

    Args:
        waypoints: List of waypoints (id, x, y, z).
        path: List of waypoint indices forming the path.
        fuel_cost: Total cost of the path.
    """
    # Initialize 3D plot with specified figure size
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates for path waypoints
    x = [waypoints[i][1] for i in path]
    y = [waypoints[i][2] for i in path]
    z = [waypoints[i][3] for i in path]
    
    # Plot waypoints as red scatter points
    ax.scatter(x, y, z, c='red', marker='o', s=50, label='Waypoints')
    
    # Plot path as blue line connecting waypoints
    ax.plot(x, y, z, c='blue', linestyle='-', linewidth=2, label='Path')
    
    # Annotate each waypoint with its ID
    for i, (id, x_coord, y_coord, z_coord) in enumerate(waypoints):
        ax.text(x_coord, y_coord, z_coord, f'ID {id}', size=10, zorder=1, color='black')
    
    # Set axis labels and title with fuel cost
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Optimal Path (Fuel Cost: {fuel_cost:.2f})')
    ax.legend()
    
    # Display plot in Jupyter/Kaggle environment
    plt.show()
    
    # Save plot to file for offline access
    output_path = 'traditional approaches/optimal_path_approach_results/path_visualization.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"3D visualization saved as {output_path}")

def write_output(file_path: str, waypoints: list[tuple[int, float, float, float]], path: list[int], fuel_cost: float) -> None:
    """Write path and fuel cost to file.

    Writes space-separated waypoint IDs followed by cost (e.g., '1 2 3 4 5 1 51.46').

    Args:
        file_path: Output file path (e.g., path.txt).
        waypoints: List of waypoints (id, x, y, z).
        path: List of waypoint indices.
        fuel_cost: Total cost of the path.

    Raises:
        ValueError: If writing fails.
    """
    try:
        # Format output: 1-based IDs followed by cost (2 decimal places)
        output_str = " ".join(str(waypoints[i][0]) for i in path) + f" {fuel_cost:.2f}"
        print("\nOutput:", output_str)
        # Write to file
        with open(file_path, 'w') as f:
            f.write(output_str + "\n")
    except Exception as e:
        raise ValueError(f"Error writing to path.txt: {str(e)}")

def get_resource_usage() -> float:
    """Calculate current memory usage in MB.

    Uses psutil to measure resident set size (RSS) of the process.

    Returns:
        Memory usage in megabytes.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_used_mb = mem_info.rss / (1024 * 1024)  # Convert bytes to MB
    return memory_used_mb

def main() -> None:
    """Main function to orchestrate TSP solution and output generation.

    Steps:
    1. Read and validate waypoints.
    2. Compute distance matrix.
    3. Solve TSP using dynamic programming.
    4. Validate path, explain segments, visualize, and write output.
    5. Report performance metrics.
    """
    try:
        # Record start time for performance tracking
        start_time = time.time()
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        # Step 1: Read waypoints from file
        waypoints = read_waypoints("sample input/waypoints.txt")
        N = len(waypoints)
        
        # Step 2: Compute Euclidean distance matrix
        dist = compute_distance_matrix(waypoints)
        
        # Step 3: Solve TSP using Held-Karp algorithm
        path, fuel_cost = solve_tsp(dist, N)
        
        # Step 4: Validate path (each waypoint visited once, returns to start)
        visited = set(path[:-1])
        if len(visited) != N or path[0] != path[-1]:
            raise ValueError("Invalid path: must visit each waypoint once and return to start")
        
        # Step 5: Explain path details and calculate total cost
        total_cost = explain_path(waypoints, path, dist)
        
        # Step 6: Visualize path in 3D and save as PNG
        visualize_path(waypoints, path, fuel_cost)
        
        # Step 7: Write path and cost to path.txt
        write_output("traditional approaches/optimal_path_approach_results/path.txt", waypoints, path, fuel_cost)
        
        # Step 8: Report performance metrics
        end_time = time.time()
        print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Time Consumed: {end_time - start_time:.2f} seconds")
        
        memory_used = get_resource_usage()
        print(f"Memory Used: {memory_used:.2f} MB")
        
    except Exception as e:
        # Handle any errors during execution
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()
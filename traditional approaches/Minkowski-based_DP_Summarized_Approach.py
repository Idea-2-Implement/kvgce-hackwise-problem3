# Standard library imports for mathematical operations, timing, system resources, and file handling
import math  # For power and root operations in Minkowski distance calculations
import time  # For measuring execution time of TSP, visualization, and total runtime
import psutil  # For monitoring memory usage to report performance metrics
import os  # For file path handling (e.g., reading waypoints1.txt, writing path_minkowski.txt)

# Third-party imports for numerical computations and visualization
import numpy as np  # For efficient array-based distance matrix operations
import matplotlib.pyplot as plt  # For creating and displaying 3D path plots
from mpl_toolkits.mplot3d import Axes3D  # For 3D axis support in visualizations

# Typing imports for enhanced type hints
from typing import List, Tuple  # For type annotations in function signatures

def read_waypoints(file_path: str) -> List[Tuple[int, float, float, float]]:
    """Read waypoints from file with strict validation.

    Args:
        file_path (str): Path to waypoints file.

    Returns:
        List[Tuple[int, float, float, float]]: List of (id, x, y, z) tuples for each waypoint.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file format, IDs, or coordinates are invalid.
    """
    # Initialize empty list for waypoints
    waypoints = []
    try:
        # Read non-empty lines from file
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        # Parse each line
        for line in lines:
            # Validate line format: exactly 4 components (id, x, y, z)
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid waypoint format in line '{line}': expected 'id x y z'")
            try:
                # Convert to floats, ensuring ID is a positive integer
                id, x, y, z = map(float, parts)
            except ValueError:
                raise ValueError(f"Invalid numerical values in line '{line}': expected integer ID and float coordinates")
            if not id.is_integer() or id < 1:
                raise ValueError(f"Invalid waypoint ID {id} in line '{line}': must be a positive integer")
            # Store waypoint as tuple
            waypoints.append((int(id), x, y, z))
        
        # Validate number of waypoints (5 to 15 per problem constraints)
        N = len(waypoints)
        if N < 5 or N > 15:
            raise ValueError(f"Invalid number of waypoints {N}: must be between 5 and 15")
        # Ensure unique, consecutive IDs from 1 to N
        ids = set(wp[0] for wp in waypoints)
        if len(ids) != N or max(ids) != N or min(ids) != 1:
            raise ValueError(f"Waypoint IDs must be unique integers from 1 to {N}, got {sorted(ids)}")
        # Validate coordinate bounds: [-1000, 1000]
        for wp in waypoints:
            _, x, y, z = wp
            if not (-1000 <= x <= 1000 and -1000 <= y <= 1000 and -1000 <= z <= 1000):
                raise ValueError(f"Coordinates out of bounds in waypoint {wp}: must be in [-1000, 1000]")
        return waypoints
    except FileNotFoundError:
        raise FileNotFoundError(f"waypoints.txt not found at {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading waypoints: {str(e)}")

def minkowski_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float], p: float = 2.0) -> float:
    """Calculate Minkowski distance between two 3D points.

    Args:
        p1 (Tuple[float, float, float]): First point (x, y, z).
        p2 (Tuple[float, float, float]): Second point (x, y, z).
        p (float): Minkowski parameter (p=2: Euclidean for hackathon, p=3: innovation).

    Returns:
        float: Minkowski distance, computed as (|x2-x1|^p + |y2-y1|^p + |z2-z1|^p)^(1/p).
    """
    # Compute Minkowski distance (Euclidean for p=2): sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    return (abs(p1[0] - p2[0])**p + abs(p1[1] - p2[1])**p + abs(p1[2] - p2[2])**p)**(1/p)

def compute_distance_matrix(waypoints: List[Tuple[int, float, float, float]], p: float = 2.0) -> np.ndarray:
    """Calculate Minkowski distance matrix for waypoints.

    Args:
        waypoints (List[Tuple[int, float, float, float]]): List of (id, x, y, z) tuples.
        p (float): Minkowski parameter (p=2 for Euclidean distance).

    Returns:
        np.ndarray: N x N matrix of Minkowski distances (float32).
    """
    # Get number of waypoints
    N = len(waypoints)
    # Initialize NxN distance matrix with zeros
    distance_matrix = np.zeros((N, N), dtype=np.float32)
    # Compute Euclidean distances for non-diagonal elements
    for i in range(N):
        for j in range(N):
            if i != j:
                # Extract coordinates of waypoints i and j
                _, x1, y1, z1 = waypoints[i]
                _, x2, y2, z2 = waypoints[j]
                # Calculate Euclidean distance (p=2)
                distance_matrix[i, j] = minkowski_3d((x1, y1, z1), (x2, y2, z2), p)
    return distance_matrix

def compute_path_cost(distance_matrix: np.ndarray, path: List[int]) -> float:
    """Calculate total distance of a path, rounded to 2 decimal places.

    Args:
        distance_matrix (np.ndarray): N x N distance matrix.
        path (List[int]): List of waypoint indices forming the path (0-indexed).

    Returns:
        float: Total path cost, rounded to 2 decimal places.
    """
    # Initialize cost
    cost = 0.0
    # Sum distances between consecutive waypoints
    for i in range(len(path) - 1):
        cost += distance_matrix[path[i], path[i + 1]]
    # Round to 2 decimal places for output consistency
    return round(cost, 2)

def validate_path(path: List[int], N: int, distance_matrix: np.ndarray, reported_cost: float) -> None:
    """Validate path for uniqueness, cycle, and cost accuracy.

    Args:
        path (List[int]): List of waypoint indices (0-indexed).
        N (int): Number of waypoints.
        distance_matrix (np.ndarray): N x N distance matrix.
        reported_cost (float): Reported path cost.

    Raises:
        ValueError: If path length, visits, cycle, or cost are invalid.
    """
    # Check path length: N waypoints + return to start
    if len(path) != N + 1:
        raise ValueError(f"Path length {len(path)} does not match N+1 ({N+1})")
    # Verify each waypoint visited exactly once
    visited = set(path[:-1])
    if len(visited) != N or visited != set(range(N)):
        raise ValueError(f"Path does not visit each waypoint exactly once: {visited}")
    # Ensure path forms a cycle
    if path[0] != path[-1]:
        raise ValueError(f"Path does not return to start: start={path[0]}, end={path[-1]}")
    # Validate reported cost against computed cost
    computed_cost = compute_path_cost(distance_matrix, path)
    reported_cost = round(reported_cost, 2)
    if abs(computed_cost - reported_cost) > 1e-3:
        raise ValueError(f"Reported cost {reported_cost:.2f} differs from computed cost {computed_cost:.2f}")
    print(f"Debug: Path validation passed: cost={computed_cost:.2f}")

def solve_tsp(distance_matrix: np.ndarray, N: int) -> Tuple[List[int], float]:
    """Solve TSP using Held-Karp Dynamic Programming with Euclidean distance.

    Uses bitmask-based dynamic programming to find the optimal tour, minimizing the total Euclidean distance.
    Returns the path and its cost, starting and ending at waypoint 1 (index 0).

    Args:
        distance_matrix (np.ndarray): N x N distance matrix (p=2, Euclidean).
        N (int): Number of waypoints.

    Returns:
        Tuple[List[int], float]: (path, cost) where path is a list of 0-indexed indices and cost is the total distance.

    Raises:
        ValueError: If no valid tour is found or path is invalid.
    """
    # Track computation time and memory
    start_time = time.time()
    mem_before = get_resource_usage()
    
    # Initialize DP table: dp[mask][end] = min distance to reach 'end' with 'mask' visited
    dp = {}  # Key: (mask, end), Value: minimum distance
    parent = {}  # Key: (mask, end), Value: previous node for path reconstruction
    print("Debug: Initializing DP table")
    
    # Step 1: Initialize base cases (paths from city 0 to each city i)
    for i in range(1, N):
        mask = 1 << i  # Bitmask with only city i set
        dp[(mask, i)] = distance_matrix[0][i]
        parent[(mask, i)] = 0
    print(f"Debug: Base cases initialized for cities 1 to {N-1}")
    
    # Step 2: Fill DP table for all subsets
    for mask in range(1, 1 << N):
        if mask & 1:
            continue  # Skip masks including city 0 (handled separately)
        for end in range(1, N):
            if not (mask & (1 << end)):
                continue  # Skip if end city is not in mask
            prev_mask = mask & ~(1 << end)  # Remove end city from mask
            if prev_mask == 0:
                continue  # Skip if previous mask is empty
            for prev in range(1, N):
                if not (prev_mask & (1 << prev)):
                    continue  # Skip if prev city is not in prev_mask
                # Compute new distance: cost to prev + edge from prev to end
                new_dist = dp.get((prev_mask, prev), float('inf')) + distance_matrix[prev][end]
                if (mask, end) not in dp or new_dist < dp[(mask, end)]:
                    dp[(mask, end)] = new_dist
                    parent[(mask, end)] = prev
    print(f"Debug: DP table filled for {len(dp)} states")
    
    # Step 3: Find optimal last city to complete tour back to city 0
    all_cities_mask = (1 << N) - 2  # All cities except 0
    min_dist = float('inf')
    last_city = -1
    
    for end in range(1, N):
        if (all_cities_mask, end) in dp:
            current_dist = dp[(all_cities_mask, end)] + distance_matrix[end][0]
            if current_dist < min_dist:
                min_dist = current_dist
                last_city = end
    print(f"Debug: Optimal last city: {last_city}, min distance: {min_dist:.2f}")
    
    if last_city == -1:
        raise ValueError("No valid tour found")
    
    # Step 4: Reconstruct path (0-based indices)
    print("Debug: Reconstructing path")
    path = []
    mask = all_cities_mask
    curr = last_city
    
    while curr != 0:
        path.append(curr)
        new_mask = mask & ~(1 << curr)
        curr = parent.get((mask, curr), 0)
        mask = new_mask
    
    path.append(0)
    path.reverse()
    path.append(0)  # Complete the cycle
    print(f"Debug: Reconstructed path: {path}")
    
    # Step 5: Compute and validate path cost
    cost = compute_path_cost(distance_matrix, path)
    try:
        validate_path(path, N, distance_matrix, cost)
    except ValueError as e:
        raise ValueError(f"Held-Karp path invalid: {str(e)}")
    
    # Log performance for debugging
    print(f"Debug: TSP solved: path={path}, cost={cost:.2f}, time={time.time() - start_time:.4f}s, "
          f"memory increase={get_resource_usage() - mem_before:.2f} MB")
    
    return path, cost

def format_sequence_table(waypoints: List[Tuple[int, float, float, float]], path: List[int], distance_matrix: np.ndarray) -> str:
    """Format a table of the optimized visit sequence and segment details.

    Args:
        waypoints (List[Tuple[int, float, float, float]]): List of (id, x, y, z) tuples.
        path (List[int]): List of waypoint indices (0-indexed).
        distance_matrix (np.ndarray): N x N distance matrix.

    Returns:
        str: Formatted table string with aligned columns.
    """
    # Define table header with aligned columns
    header = "📍 Optimized Visit Sequence & Segment Details:\n" + \
             "-" * 60 + "\n" + \
             f"{'Step':<6}{'From':^6}{'To':^6}{'Distance':>10}{'To Coordinates (x,y,z)':<25}\n" + \
             "-" * 60
    rows = []
    # Generate table rows for each path segment
    for step, i in enumerate(range(len(path) - 1), 1):
        start_idx = path[i]
        end_idx = path[i + 1]
        start_id = waypoints[start_idx][0]
        end_id = waypoints[end_idx][0]
        distance = distance_matrix[start_idx, end_idx]
        # Format coordinates with 2 decimal places
        x, y, z = waypoints[end_idx][1:]
        coords = f"({x:7.2f}, {y:7.2f}, {z:7.2f})"
        # Add row with aligned columns
        rows.append(f"{step:<6}{start_id:^6}{end_id:^6}{distance:>10.2f}{coords:<25}")
    # Combine header, rows, and footer
    return header + "\n" + "\n".join(rows) + "\n" + "-" * 60

def explain_path(waypoints: List[Tuple[int, float, float, float]], path: List[int], distance_matrix: np.ndarray) -> float:
    """Explain the chosen path with a formatted table and detailed Euclidean calculations.

    Args:
        waypoints (List[Tuple[int, float, float, float]]): List of (id, x, y, z) tuples.
        path (List[int]): List of waypoint indices (0-indexed).
        distance_matrix (np.ndarray): N x N distance matrix (p=2, Euclidean).

    Returns:
        float: Total path cost, rounded to 2 decimal places.
    """
    # Explain path selection logic
    print("\nLogic Behind Chosen Path:")
    print("The path is computed using Held-Karp Dynamic Programming with Euclidean distance (Minkowski p=2), "
          "using bitmasks to track visited waypoints and computing the optimal tour starting from waypoint 1. "
          "Minkowski p=3 is available for innovation but not used for hackathon scoring.")
    
    # Display formatted sequence table
    print(format_sequence_table(waypoints, path, distance_matrix))
    
    # Provide detailed segment calculations using Euclidean formula
    print("\nPath Details:")
    total_cost = compute_path_cost(distance_matrix, path)
    for i in range(len(path) - 1):
        start_idx = path[i]
        end_idx = path[i + 1]
        # Extract IDs and coordinates
        start_id, x1, y1, z1 = waypoints[start_idx]
        end_id, x2, y2, z2 = waypoints[end_idx]
        distance = distance_matrix[start_idx, end_idx]
        # Print segment details with Euclidean formula
        print(f"Segment {start_id} -> {end_id}:")
        print(f"  Coordinates: ({x1:.2f}, {y1:.2f}, {z1:.2f}) -> ({x2:.2f}, {y2:.2f}, {z2:.2f})")
        print(f"  Euclidean Distance = sqrt((({x2:.2f}-{x1:.2f})^2 + ({y2:.2f}-{y1:.2f})^2 + "
              f"({z2:.2f}-{z1:.2f})^2)) = {distance:.2f}")
    print(f"Total Fuel Cost: {total_cost:.2f}")
    return total_cost

def visualize_path(waypoints: List[Tuple[int, float, float, float]], path: List[int], fuel_cost: float) -> Tuple[float, float]:
    """Display and save a 3D visualization of the Euclidean path.

    Args:
        waypoints (List[Tuple[int, float, float, float]]): List of (id, x, y, z) tuples.
        path (List[int]): List of waypoint indices (0-indexed).
        fuel_cost (float): Total path cost.

    Returns:
        Tuple[float, float]: (visualization_time, memory_increase) for performance logging.

    Raises:
        FileNotFoundError: If visualization file cannot be saved.
        ValueError: For other visualization errors.
    """
    # Track visualization performance
    mem_before = get_resource_usage()
    start_time = time.time()
    try:
        # Initialize 3D plot
        print("Debug: Creating 3D plot")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates for path
        x = [waypoints[i][1] for i in path]
        y = [waypoints[i][2] for i in path]
        z = [waypoints[i][3] for i in path]
        # Plot waypoints as red points
        ax.scatter(x, y, z, c='red', marker='o', s=50, label='Waypoints')
        # Plot path as blue line
        ax.plot(x, y, z, c='blue', linestyle='-', linewidth=2, label='Path')
        
        # Add ID labels to waypoints
        for i, (id, x_coord, y_coord, z_coord) in enumerate(waypoints):
            ax.text(x_coord, y_coord, z_coord, f'ID {id}', size=10, zorder=1, color='black')
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Euclidean DP Path (Fuel Cost: {fuel_cost:.2f})')
        ax.legend()
        
        # Display plot in Jupyter/Kaggle (if applicable)
        print("Debug: Displaying plot")
        plt.show()
        
        # Save plot to file
        output_path = 'traditional approaches/Minkowski-based_DP_Summarized_Approach/path_visualization_minkowski.png'
        print(f"Debug: Saving plot to {output_path}")
        plt.savefig(output_path, bbox_inches='tight')
        print("Debug: Plot saved successfully")
        
        # Close figure to free memory
        plt.close(fig)
        print("Debug: Figure closed")
        
        # Verify file creation
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Visualization file {output_path} not created")
        print(f"3D visualization saved as {output_path}")
        
        # Calculate performance metrics
        mem_after = get_resource_usage()
        return time.time() - start_time, mem_after - mem_before
    
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        raise ValueError(f"Error in visualization: {str(e)}")

def write_output(file_path: str, waypoints: List[Tuple[int, float, float, float]], path: List[int], fuel_cost: float) -> None:
    """Write path and fuel cost to file with validation.

    Args:
        file_path (str): Path to output file (path_minkowski.txt).
        waypoints (List[Tuple[int, float, float, float]]): List of (id, x, y, z) tuples.
        path (List[int]): List of waypoint indices (0-indexed).
        fuel_cost (float): Total path cost.

    Raises:
        ValueError: If output path length or cost is invalid.
    """
    try:
        # Convert path indices to 1-based IDs
        one_indexed_path = [waypoints[i][0] for i in path]
        # Validate output path length
        if len(one_indexed_path) != len(waypoints) + 1:
            raise ValueError(f"Output path length {len(one_indexed_path)} does not match N+1 ({len(waypoints)+1})")
        # Format output string
        output_str = " ".join(map(str, one_indexed_path)) + f" {fuel_cost:.2f}"
        print(f"Debug: Writing output to {file_path}: {output_str}")
        # Write to file
        with open(file_path, 'w') as f:
            f.write(output_str + "\n")
        # Verify written content
        with open(file_path, 'r') as f:
            written = f.read().strip().split()
        print(f"Debug: Read from {file_path}: {written}")
        # Check number of values (N+1 IDs + cost)
        if len(written) != len(waypoints) + 2:
            raise ValueError(f"{file_path} contains {len(written)} values, expected {len(waypoints)+2}")
        # Verify cost
        try:
            written_cost = float(written[-1])
            if abs(written_cost - fuel_cost) > 1e-3:
                raise ValueError(f"Written cost {written_cost} does not match computed cost {fuel_cost:.2f}")
        except ValueError:
            raise ValueError(f"Invalid fuel cost format in {file_path}")
    except Exception as e:
        raise ValueError(f"Error writing to {file_path}: {str(e)}")

def get_resource_usage() -> float:
    """Calculate current memory usage in MB.

    Returns:
        float: Memory usage in megabytes.
    """
    # Get process memory info
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Convert RSS from bytes to MB
    return mem_info.rss / (1024 * 1024)

def main() -> None:
    """Main function to orchestrate TSP solving and output generation.

    Steps:
    1. Initialize timing and logging.
    2. Read and validate waypoints.
    3. Compute Euclidean distance matrix (p=2).
    4. Solve TSP using Held-Karp Dynamic Programming.
    5. Explain and validate the optimal path.
    6. Print the optimal path sequence.
    7. Visualize the path in 3D.
    8. Write output to file.
    9. Log performance metrics.
    """
    try:
        # ------------------------------
        # Step 1: Initialize timing and logging
        # ------------------------------
        start_time = time.time()
        mem_before = get_resource_usage()
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        # ------------------------------
        # Step 2: Read and validate waypoints
        # ------------------------------
        waypoints = read_waypoints("sample input/waypoints.txt")
        N = len(waypoints)
        print(f"Read {N} waypoints successfully")
        
        # ------------------------------
        # Step 3: Compute Euclidean distance matrix
        # ------------------------------
        distance_matrix = compute_distance_matrix(waypoints, p=2.0)
        print("Euclidean distance matrix (p=2) computed")
        
        # ------------------------------
        # Step 4: Solve TSP
        # ------------------------------
        tsp_start = time.time()
        path, fuel_cost = solve_tsp(distance_matrix, N)
        tsp_time = time.time() - tsp_start
        print(f"Optimal path computed: {path}, cost={fuel_cost:.2f}")
        
        # ------------------------------
        # Step 5: Explain and validate path
        # ------------------------------
        total_cost = explain_path(waypoints, path, distance_matrix)
        validate_path(path, N, distance_matrix, total_cost)
        print("Path explanation and validation completed")
        
        # ------------------------------
        # Step 6: Print optimal path sequence
        # ------------------------------
        one_indexed_path = [waypoints[i][0] for i in path]
        print(f"\nOptimal Path Sequence: {' '.join(map(str, one_indexed_path))}")
        
        # ------------------------------
        # Step 7: Visualize path
        # ------------------------------
        vis_time, vis_mem_increase = visualize_path(waypoints, path, fuel_cost)
        print("Visualization generated")
        
        # ------------------------------
        # Step 8: Write output
        # ------------------------------
        write_output("traditional approaches/Minkowski-based_DP_Summarized_Approach/path_minkowski.txt", waypoints, path, fuel_cost)
        print("Output written to path_minkowski.txt")
        
        # ------------------------------
        # Step 9: Log performance metrics
        # ------------------------------
        end_time = time.time()
        total_time = end_time - start_time
        total_mem = get_resource_usage()
        print(f"\nPerformance Summary:")
        print(f"  TSP Solving Time: {tsp_time:.4f} seconds")
        print(f"  Visualization Time: {vis_time:.4f} seconds")
        print(f"  Total Time: {total_time:.2f} seconds")
        print(f"  Memory Used: {total_mem:.2f} MB")
        print(f"  Memory Increase (Visualization): {vis_mem_increase:.2f} MB")
        print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
    except Exception as e:
        # Handle and report any errors
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Entry point for script execution
    main()
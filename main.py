# Standard library imports for mathematical operations, timing, and file handling
import math  # For sqrt and other mathematical functions used in distance calculations
import time  # For measuring execution time of TSP and visualization steps
import psutil  # For monitoring memory usage to report performance metrics
import os  # For file path handling and checking existence of input/output files

# Third-party imports for numerical computations and visualization
import numpy as np  # For efficient array operations in distance matrix computations
import matplotlib.pyplot as plt  # For creating 3D plots of the TSP path
from mpl_toolkits.mplot3d import Axes3D  # For 3D axis support in path visualizations

# Typing imports for type hints to improve code readability and maintainability
from typing import List, Tuple  # For type annotations in function signatures

# Functools import for performance optimization
from functools import lru_cache  # For memoizing Minkowski distance calculations to reduce redundant computations

def read_waypoints(file_path: str) -> List[Tuple[int, float, float, float]]:
    """Read and validate waypoints from a file.

    Args:
        file_path: Path to the waypoints file (format: id x y z per line).

    Returns:
        List of tuples (id, x, y, z) representing waypoints.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file format or waypoint data is invalid.
    """
    waypoints = []
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid waypoint format in line '{line}': expected 'id x y z'")
            try:
                id, x, y, z = map(float, parts)
            except ValueError:
                raise ValueError(f"Invalid numerical values in line '{line}': expected integer ID and float coordinates")
            if not id.is_integer() or id < 1:
                raise ValueError(f"Invalid waypoint ID {id} in line '{line}': must be a positive integer")
            waypoints.append((int(id), x, y, z))
        
        num_waypoints = len(waypoints)
        # Validate number of waypoints (5 to 15 per problem constraints)
        if num_waypoints < 5 or num_waypoints > 15:
            raise ValueError(f"Invalid number of waypoints {num_waypoints}: must be between 5 and 15")
        # Ensure unique, consecutive IDs from 1 to N
        ids = set(wp[0] for wp in waypoints)
        if len(ids) != num_waypoints or max(ids) != num_waypoints or min(ids) != 1:
            raise ValueError(f"Waypoint IDs must be unique integers from 1 to {num_waypoints}, got {sorted(ids)}")
        # Check coordinate bounds
        for wp in waypoints:
            _, x, y, z = wp
            if not (-1000 <= x <= 1000 and -1000 <= y <= 1000 and -1000 <= z <= 1000):
                raise ValueError(f"Coordinates out of bounds in waypoint {wp}: must be in [-1000, 1000]")
        return waypoints
    except FileNotFoundError:
        raise FileNotFoundError(f"waypoints.txt not found at {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading waypoints: {str(e)}")

@lru_cache(maxsize=10000)
def minkowski_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float], p: float) -> float:
    """Calculate Minkowski distance between two 3D points with caching.

    Uses caching to optimize repeated distance calculations.
    Handles p=2 (Euclidean) efficiently and scales coordinates to prevent overflow for large p.

    Args:
        p1: First point coordinates (x, y, z).
        p2: Second point coordinates (x, y, z).
        p: Minkowski p-value (e.g., 2.0 for Euclidean).

    Returns:
        Minkowski distance between the points.

    Raises:
        OverflowError: If numerical overflow occurs.
    """
    try:
        max_diff = max(abs(p1[i] - p2[i]) for i in range(3))
        if max_diff == 0:
            return 0.0  # Identical points
        if p == 2.0:
            # Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
            return math.sqrt(sum((p1[i] - p2[i])**2 for i in range(3)))
        # Scale to prevent overflow for large p
        scaled = [(abs(p1[i] - p2[i]) / max_diff)**p for i in range(3)]
        result = max_diff * (sum(scaled)**(1/p))
        if math.isinf(result) or math.isnan(result):
            raise OverflowError("Minkowski distance overflow")
        return result
    except OverflowError as e:
        raise OverflowError(f"Error computing Minkowski distance (p={p}): {str(e)}")

def compute_distance_matrix(waypoints: List[Tuple[int, float, float, float]], p: float = 2.0) -> np.ndarray:
    """Compute distance matrix for waypoints using Minkowski distance.

    Args:
        waypoints: List of waypoints (id, x, y, z).
        p: Minkowski p-value (default: 2.0 for Euclidean).

    Returns:
        NxN NumPy array of distances between waypoints.
    """
    num_waypoints = len(waypoints)
    distance_matrix = np.zeros((num_waypoints, num_waypoints), dtype=np.float32)
    for i in range(num_waypoints):
        for j in range(num_waypoints):
            if i != j:
                _, x1, y1, z1 = waypoints[i]
                _, x2, y2, z2 = waypoints[j]
                distance_matrix[i, j] = minkowski_3d((x1, y1, z1), (x2, y2, z2), p)
    return distance_matrix

def compute_path_cost(distance_matrix: np.ndarray, path: List[int]) -> float:
    """Calculate total cost of a path, rounded to 2 decimal places.

    Args:
        distance_matrix: NxN matrix of distances.
        path: List of waypoint indices forming the path.

    Returns:
        Total path cost.
    """
    cost = 0.0
    for i in range(len(path) - 1):
        cost += distance_matrix[path[i], path[i + 1]]
    return round(cost, 2)

def validate_path(path: List[int], num_waypoints: int, distance_matrix: np.ndarray, reported_cost: float) -> None:
    """Validate path for correctness and cost accuracy.

    Ensures path visits each waypoint exactly once, returns to start, and matches reported cost.

    Args:
        path: List of waypoint indices.
        num_waypoints: Number of waypoints (N).
        distance_matrix: NxN matrix of distances.
        reported_cost: Reported total cost of the path.

    Raises:
        ValueError: If path is invalid or cost is incorrect.
    """
    # Check path length (N+1 for closed tour)
    if len(path) != num_waypoints + 1:
        raise ValueError(f"Path length {len(path)} does not match N+1 ({num_waypoints+1})")
    # Verify all waypoints visited exactly once
    visited = set(path[:-1])
    if len(visited) != num_waypoints or visited != set(range(num_waypoints)):
        raise ValueError(f"Path does not visit each waypoint exactly once: {visited}")
    # Ensure path returns to start
    if path[0] != path[-1]:
        raise ValueError(f"Path does not return to start: start={path[0]}, end={path[-1]}")
    # Validate cost accuracy
    computed_cost = compute_path_cost(distance_matrix, path)
    reported_cost = round(reported_cost, 2)
    if abs(computed_cost - reported_cost) > 1e-3:
        raise ValueError(f"Reported cost {reported_cost:.2f} differs from computed cost {computed_cost:.2f}")

def solve_tsp(distance_matrix: np.ndarray, num_waypoints: int, waypoints: List[Tuple[int, float, float, float]], p: float = 2.0) -> Tuple[List[int], float]:
    """Solve TSP using Held-Karp Dynamic Programming with branch-and-bound pruning.

    Held-Karp uses O(N^2 * 2^N) time to find the exact optimal tour.
    Branch-and-bound prunes suboptimal paths to meet performance constraints (<5s for N=15).
    Uses bitmask to represent visited waypoints and dynamic programming for minimum cost.

    Args:
        distance_matrix: NxN matrix of distances.
        num_waypoints: Number of waypoints (N).
        waypoints: List of waypoints (id, x, y, z).
        p: Minkowski p-value for distance metric.

    Returns:
        Tuple of (path, cost), where path is a list of indices and cost is the total distance.

    Raises:
        ValueError: If no valid tour is found.
    """
    dp = {}  # Memoization table: (mask, end) -> min distance
    parent = {}  # Track parent nodes for path reconstruction
    best_bound = float('inf')  # Best known tour cost for pruning
    
    # Initialize base cases: paths from start (index 0) to each city
    for i in range(1, num_waypoints):
        mask = 1 << (i - 1)  # Bitmask for city i (0-based index)
        dp[(mask, i)] = distance_matrix[0][i]
        parent[(mask, i)] = 0
    
    # Iterate over all subsets of cities (excluding start)
    for mask in range(1, 1 << (num_waypoints - 1)):
        for end in range(1, num_waypoints):
            if not (mask & (1 << (end - 1))):
                continue  # Skip if end city not in current subset
            prev_mask = mask & ~(1 << (end - 1))  # Remove end city from mask
            for prev in range(num_waypoints):
                if prev != 0 and not (prev_mask & (1 << (prev - 1))):
                    continue  # Skip if prev city not in previous subset
                if prev == 0 and prev_mask != 0:
                    continue  # Start city only for empty previous subset
                new_dist = dp.get((prev_mask, prev), float('inf')) + distance_matrix[prev][end]
                # Prune if exceeds best known bound
                if new_dist >= best_bound:
                    continue
                if (mask, end) not in dp or new_dist < dp[(mask, end)]:
                    dp[(mask, end)] = new_dist
                    parent[(mask, end)] = prev
    
    # Find minimum cost to complete the tour (return to start)
    all_cities_mask = (1 << (num_waypoints - 1)) - 1
    min_dist = float('inf')
    last_city = -1
    
    for end in range(1, num_waypoints):
        if (all_cities_mask, end) in dp:
            current_dist = dp[(all_cities_mask, end)] + distance_matrix[end][0]
            if current_dist < min_dist:
                min_dist = current_dist
                last_city = end
                best_bound = min(current_dist, best_bound)
    
    if last_city == -1:
        raise ValueError("No valid tour found")
    
    # Reconstruct path from parent pointers
    path = []
    mask = all_cities_mask
    curr = last_city
    
    while curr != 0:
        path.append(curr)
        prev = parent[(mask, curr)]
        mask = mask & ~(1 << (curr - 1))
        curr = prev
    
    path.append(0)
    path.reverse()
    path.append(0)  # Close the tour
    
    cost = compute_path_cost(distance_matrix, path)
    validate_path(path, num_waypoints, distance_matrix, cost)
    
    return path, cost

def format_sequence_table(waypoints: List[Tuple[int, float, float, float]], path: List[int], distance_matrix: np.ndarray) -> str:
    """Format a table of the optimized visit sequence.

    Args:
        waypoints: List of waypoints (id, x, y, z).
        path: List of waypoint indices forming the path.
        distance_matrix: NxN matrix of distances.

    Returns:
        Formatted table string.
    """
    header = "📍 Optimized Visit Sequence & Segment Details:\n" + \
             "-" * 60 + "\n" + \
             f"{'Step':<6}{'From':^8}{'To':^6}{'Distance':>8}{'To Coordinates (x,y,z)':<24}\n" + \
             "-" * 60
    rows = []
    for step, i in enumerate(range(len(path) - 1), 1):
        start_idx = path[i]
        end_idx = path[i + 1]
        start_id = waypoints[start_idx][0]
        end_id = waypoints[end_idx][0]
        distance = distance_matrix[start_idx, end_idx]
        x, y, z = waypoints[end_idx][1:]
        coords = f"({x:6.2f}, {y:6.2f}, {z:6.2f})"
        rows.append(f"{step:<6}{start_id:^8}{end_id:^6}{distance:>8.2f}{coords:<24}")
    return header + "\n" + "\n".join(rows) + "\n" + "-" * 60

def explain_path(waypoints: List[Tuple[int, float, float, float]], path: List[int], distance_matrix: np.ndarray, p: float) -> float:
    """Print detailed explanation of the chosen path and its cost.

    Displays sequence table, segment calculations, total cost, and optimal path.

    Args:
        waypoints: List of waypoints (id, x, y, z).
        path: List of waypoint indices forming the path.
        distance_matrix: NxN matrix of distances.
        p: Minkowski p-value.

    Returns:
        Total fuel cost of the path.
    """
    distance_type = "Euclidean" if p == 2.0 else f"Minkowski (p={p})"
    print(f"\nLogic Behind Chosen Path ({distance_type}):")
    print(f"The path is computed using Held-Karp Dynamic Programming with branch-and-bound pruning. "
          f"Best Minkowski path (lowest cost) is saved to path.txt for innovation.")
    
    # Display formatted table of visit sequence
    print(format_sequence_table(waypoints, path, distance_matrix))
    
    # Explain each segment's distance calculation
    print("\nPath Details:")
    total_cost = compute_path_cost(distance_matrix, path)
    for i in range(len(path) - 1):
        start_idx = path[i]
        end_idx = path[i + 1]
        start_id, x1, y1, z1 = waypoints[start_idx]
        end_id, x2, y2, z2 = waypoints[end_idx]
        distance = distance_matrix[start_idx, end_idx]
        print(f"Segment {start_id} -> {end_id}:")
        print(f"  Coordinates: ({x1:.2f}, {y1:.2f}, {z1:.2f}) -> ({x2:.2f}, {y2:.2f}, {z2:.2f})")
        if p == 2.0:
            print(f"  Euclidean Distance = sqrt(({x2:.2f}-{x1:.2f})^2 + ({y2:.2f}-{y1:.2f})^2 + "
                  f"({z2:.2f}-{z1:.2f})^2) = {distance:.2f}")
        else:
            print(f"  Minkowski Distance (p={p}) = ((|{x2:.2f}-{x1:.2f}|^{p} + |{y2:.2f}-{y1:.2f}|^{p} + "
                  f"|{z2:.2f}-{z1:.2f}|^{p}))^(1/{p}) = {distance:.2f}")
    print(f"Total Fuel Cost: {total_cost:.2f}")
    
    # Display the optimal path in ID format (1-based)
    one_indexed_path = [waypoints[i][0] for i in path]
    print(f"\nPath: {' '.join(map(str, one_indexed_path))}")
    
    return total_cost

def visualize_path(waypoints: List[Tuple[int, float, float, float]], path: List[int], fuel_cost: float, p: float) -> Tuple[float, float]:
    """Generate and save a 3D visualization of the path.

    Creates a 3D plot with waypoints as points, path as lines, and labels for IDs.
    Saves the plot as a PNG file and measures performance.

    Args:
        waypoints: List of waypoints (id, x, y, z).
        path: List of waypoint indices forming the path.
        fuel_cost: Total cost of the path.
        p: Minkowski p-value.

    Returns:
        Tuple of (visualization time, memory increase in MB).

    Raises:
        ValueError: If visualization fails.
    """
    mem_before = get_resource_usage()
    start_time = time.time()
    try:
        # Initialize 3D plot
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot waypoints as red scatter points
        x = [waypoints[i][1] for i in path]
        y = [waypoints[i][2] for i in path]
        z = [waypoints[i][3] for i in path]
        ax.scatter(x, y, z, c='red', marker='o', s=30, label='Waypoints')
        # Plot path as blue line
        ax.plot(x, y, z, c='blue', linestyle='-', linewidth=1.5, label='Path')
        
        # Add ID labels to waypoints
        for i, (id, x_coord, y_coord, z_coord) in enumerate(waypoints):
            ax.text(x_coord, y_coord, z_coord, f'ID {id}', size=8, zorder=1, color='black')
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        distance_type = "Euclidean" if p == 2.0 else f"Minkowski (p={p})"
        ax.set_title(f'{distance_type} Path (Cost: {fuel_cost:.2f})', fontsize=10)
        ax.legend(fontsize=8)
        
        # Save plot to file
        output_path = f'sample_output/process_results/path_visualization_3d_p{p}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.show(block=False)
        plt.pause(5)
        plt.close(fig)  # Free memory
        
        # Verify file creation
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Visualization file {output_path} not created")
        
        mem_after = get_resource_usage()
        return time.time() - start_time, max(mem_after - mem_before, 0.0)
    
    except Exception as e:
        raise ValueError(f"Error in visualization: {str(e)}")

def write_output(file_path: str, waypoints: List[Tuple[int, float, float, float]], path: List[int], fuel_cost: float) -> None:
    """Write path and fuel cost to file with validation.

    Writes path in 1-based ID format followed by cost (e.g., "1 4 5 3 2 1 40.18").

    Args:
        file_path: Output file path (e.g., path.txt).
        waypoints: List of waypoints (id, x, y, z).
        path: List of waypoint indices.
        fuel_cost: Total cost of the path.

    Raises:
        ValueError: If writing fails or output is invalid.
    """
    try:
        # Convert to 1-based IDs
        one_indexed_path = [waypoints[i][0] for i in path]
        # Validate path length
        if len(one_indexed_path) != len(waypoints) + 1:
            raise ValueError(f"Output path length {len(one_indexed_path)} does not match N+1 ({len(waypoints)+1})")
        output_str = " ".join(map(str, one_indexed_path)) + f" {fuel_cost:.2f}"
        with open(file_path, 'w') as f:
            f.write(output_str + "\n")
        # Verify written content
        with open(file_path, 'r') as f:
            written = f.read().strip().split()
        if len(written) != len(waypoints) + 2:
            raise ValueError(f"{file_path} contains {len(written)} values, expected {len(waypoints)+2}")
        try:
            written_cost = float(written[-1])
            if abs(written_cost - fuel_cost) > 1e-3:
                raise ValueError(f"Written cost {written_cost} does not match computed cost {fuel_cost:.2f}")
        except ValueError:
            raise ValueError(f"Invalid fuel cost format in {file_path}")
    except Exception as e:
        raise ValueError(f"Error writing to {file_path}: {str(e)}")

def write_results_summary(file_path: str, results: List[dict], waypoints: List[Tuple[int, float, float, float]]) -> None:
    """Write a summary of all results to a file.

    Summarizes paths and costs for each p-value tested.

    Args:
        file_path: Output file path (e.g., results.txt).
        results: List of result dictionaries {p, path, fuel_cost, tsp_time}.
        waypoints: List of waypoints (id, x, y, z).

    Raises:
        ValueError: If writing fails.
    """
    try:
        with open(file_path, 'w') as f:
            f.write("=== TSP Results Summary ===\n\n")
            for result in results:
                p = result['p']
                path = result['path']
                fuel_cost = result['fuel_cost']
                tsp_time = result['tsp_time']
                distance_type = "Euclidean" if p == 2.0 else f"Minkowski (p={p})"
                one_indexed_path = [waypoints[i][0] for i in path]
                f.write(f"Distance Metric: {distance_type}\n")
                f.write(f"Fuel Cost: {fuel_cost:.2f}\n")
                f.write(f"Path: {' '.join(map(str, one_indexed_path))}\n")
                f.write(f"TSP Time: {tsp_time:.4f} seconds\n\n")
    except Exception as e:
        print(f"Warning: Failed to write results summary to {file_path}: {str(e)}")

def get_resource_usage() -> float:
    """Calculate current memory usage in MB.

    Returns:
        Memory usage in megabytes.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

def main() -> None:
    """Main function to solve TSP with multiple p-values and generate outputs.

    Steps:
    1. Read waypoints from waypoints.txt.
    2. Compute optimal path for Euclidean distance (p=2.0) and display details.
    3. Visualize Euclidean path.
    4. Test other Minkowski p-values (1.0 to 100.0) to find the best path.
    5. Display and visualize the best Minkowski path.
    6. Save best path to path.txt and summary to results.txt.
    7. Print conclusion and performance metrics.

    Uses Held-Karp algorithm with branch-and-bound for exact TSP solution.
    """
    try:
        # Initialize performance tracking
        start_time = time.time()
        mem_before = get_resource_usage()
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        # Read input waypoints
        input_path = "sample input/waypoints.txt"
        if not os.path.exists(input_path):
            input_path = "sample input/TEST 1.txt"
        waypoints = read_waypoints(input_path)
        num_waypoints = len(waypoints)
        
        # Step 1: Compute and display Euclidean path (p=2.0, hackathon scoring)
        print("\n=== Optimal Path with Euclidean Distance (p=2.0) ===")
        print("TSP Algorithm: Held-Karp Dynamic Programming with branch-and-bound pruning")
        distance_matrix = compute_distance_matrix(waypoints, 2.0)
        tsp_start = time.time()
        euclidean_path, euclidean_cost = solve_tsp(distance_matrix, num_waypoints, waypoints, 2.0)
        tsp_time_euclidean = time.time() - tsp_start
        total_cost = explain_path(waypoints, euclidean_path, distance_matrix, 2.0)
        validate_path(euclidean_path, num_waypoints, distance_matrix, total_cost)
        
        # Step 2: Visualize Euclidean path
        vis_time, vis_mem_increase = visualize_path(waypoints, euclidean_path, euclidean_cost, 2.0)
        print(f"\nVisualization Time (p=2.0): {vis_time:.4f} seconds")
        print(f"Memory Increase (Visualization, p=2.0): {vis_mem_increase:.2f} MB")
        
        # Step 3: Test other Minkowski p-values to find best path
        p_values = [1.0, 3.0, 4.0, 6.0, 8.0, 10.0, 20.0, 50.0, 100.0]
        results = [{'p': 2.0, 'path': euclidean_path, 'fuel_cost': euclidean_cost, 'tsp_time': tsp_time_euclidean}]
        
        print("\n=== Testing Other Distance Metrics ===")
        for p in p_values:
            try:
                distance_matrix = compute_distance_matrix(waypoints, p)
                tsp_start = time.time()
                path, fuel_cost = solve_tsp(distance_matrix, num_waypoints, waypoints, p)
                tsp_time = time.time() - tsp_start
                results.append({'p': p, 'path': path, 'fuel_cost': fuel_cost, 'tsp_time': tsp_time})
            except OverflowError as e:
                print(f"Warning: Skipping p={p} due to numerical overflow: {str(e)}")
                continue
        
        if len(results) < 2:
            print("Warning: Limited results computed; proceeding with available data")
        
        # Display fuel costs for all metrics
        print("\nFuel Costs for Different Distance Metrics:")
        print("-" * 40)
        print(f"{'Distance Type':<15}{'p':>5}{'Fuel Cost':>10}{'TSP Time (s)':>15}")
        print("-" * 40)
        for result in sorted(results, key=lambda x: x['p']):
            distance_type = "Euclidean" if result['p'] == 2.0 else "Minkowski"
            print(f"{distance_type:<15}{result['p']:>5.1f}{result['fuel_cost']:>10.2f}{result['tsp_time']:>15.4f}")
        print("-" * 40)
        
        # Step 4: Display best Minkowski path (lowest cost)
        best_result = min(results, key=lambda x: x['fuel_cost'])
        print(f"\n=== Best Path with Minkowski Distance (p={best_result['p']}) ===")
        print(f"Innovation Note: Lowest fuel cost {best_result['fuel_cost']:.2f} achieved with p={best_result['p']}")
        print("TSP Algorithm: Held-Karp Dynamic Programming with branch-and-bound pruning")
        distance_matrix = compute_distance_matrix(waypoints, best_result['p'])
        total_cost = explain_path(waypoints, best_result['path'], distance_matrix, best_result['p'])
        validate_path(best_result['path'], num_waypoints, distance_matrix, total_cost)
        
        # Step 5: Visualize best Minkowski path
        vis_time, vis_mem_increase = visualize_path(waypoints, best_result['path'], best_result['fuel_cost'], best_result['p'])
        print(f"\nVisualization Time (p={best_result['p']}): {vis_time:.4f} seconds")
        print(f"Memory Increase (Visualization, p={best_result['p']}): {vis_mem_increase:.2f} MB")
        
        # Step 6: Save best path to path.txt
        output_path = "sample_output/path.txt" if os.path.exists("sample_output/") else "path.txt"
        write_output(output_path, waypoints, best_result['path'], best_result['fuel_cost'])
        
        # Save results summary
        results_path = "sample_output/process_results/results.txt" if os.path.exists("sample_output/") else "results.txt"
        write_results_summary(results_path, results, waypoints)
        
        # Step 7: Print conclusion comparing Euclidean and best Minkowski paths
        print(f"\n=== Conclusion ===")
        print(f"Euclidean Path (p=2.0, hackathon scoring):")
        one_indexed_path = [waypoints[i][0] for i in euclidean_path]
        print(f"  Path: {' '.join(map(str, one_indexed_path))}")
        print(f"  Fuel Cost: {euclidean_cost:.2f}")
        print(f"Best Minkowski Path (p={best_result['p']}, lowest cost, saved to path.txt):")
        one_indexed_path = [waypoints[i][0] for i in best_result['path']]
        print(f"  Path: {' '.join(map(str, one_indexed_path))}")
        print(f"  Fuel Cost: {best_result['fuel_cost']:.2f}")
        
        # Report overall performance
        end_time = time.time()
        total_time = end_time - start_time
        total_mem = get_resource_usage()  # Calculate final memory usage
        print(f"\nPerformance Summary (Overall):")
        print(f"  Total Time: {total_time:.2f} seconds")
        print(f"  Memory Used: {total_mem:.2f} MB")
        print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
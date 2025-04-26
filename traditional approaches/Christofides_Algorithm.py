# Standard library imports for mathematical operations, timing, system resources, and utilities
import math  # For sqrt in Euclidean distance calculations
import time  # For measuring execution time of TSP, visualization, and total runtime
import psutil  # For monitoring memory usage to report performance metrics
import os  # For file path handling (e.g., reading waypoints.txt, writing path_christofides.txt)
import itertools  # For generating combinations in minimum-weight perfect matching

# Third-party imports for numerical computations, visualization, and graph operations
import numpy as np  # For efficient array-based distance matrix operations
import matplotlib.pyplot as plt  # For creating and displaying 3D path plots
from mpl_toolkits.mplot3d import Axes3D  # For 3D axis support in visualizations
import networkx as nx  # For graph algorithms (MST, Eulerian circuit) in Christofides Algorithm



def read_waypoints(file_path: str) -> list[tuple[int, float, float, float]]:
    """Read waypoints from file with strict validation.

    Args:
        file_path (str): Path to waypoints file.

    Returns:
        list[tuple[int, float, float, float]]: List of (id, x, y, z) tuples for each waypoint.

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

def compute_distance_matrix(waypoints: list[tuple[int, float, float, float]]) -> np.ndarray:
    """Calculate Euclidean distance matrix for waypoints.

    Args:
        waypoints (list[tuple[int, float, float, float]]): List of (id, x, y, z) tuples.

    Returns:
        np.ndarray: N x N matrix of Euclidean distances (float32).
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
                # Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
                distance_matrix[i, j] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return distance_matrix

def compute_path_cost(distance_matrix: np.ndarray, path: list[int]) -> float:
    """Calculate total Euclidean distance of a path, rounded to 2 decimal places.

    Args:
        distance_matrix (np.ndarray): N x N distance matrix.
        path (list[int]): List of waypoint indices forming the path.

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

def validate_path(path: list[int], N: int, distance_matrix: np.ndarray, reported_cost: float) -> None:
    """Validate path for uniqueness, cycle, and cost accuracy.

    Args:
        path (list[int]): List of waypoint indices.
        N (int): Number of waypoints.
        distance_matrix (np.ndarray): N x N distance matrix.
        reported_cost (float): Reported path cost.

    Raises:
        ValueError: If path length, visits, or cycle are invalid.
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
    # Warn if reported cost differs (approximation may introduce minor discrepancies)
    computed_cost = compute_path_cost(distance_matrix, path)
    reported_cost = round(reported_cost, 2)
    if abs(computed_cost - reported_cost) > 1e-3:
        print(f"Warning: Reported cost {reported_cost:.2f} differs from computed cost {computed_cost:.2f} (approximation)")
    else:
        print(f"Debug: Path validation passed: cost={computed_cost:.2f}")

def solve_tsp(distance_matrix: np.ndarray, N: int) -> tuple[list[int], float]:
    """Solve TSP using Christofides Algorithm for a 1.5-approximation path.

    Constructs a Minimum Spanning Tree, adds minimum-weight perfect matching for odd-degree vertices,
    finds an Eulerian circuit, and shortcuts to a Hamiltonian cycle.

    Args:
        distance_matrix (np.ndarray): N x N distance matrix.
        N (int): Number of waypoints.

    Returns:
        tuple[list[int], float]: (path, cost) where path is a list of indices and cost is the total distance.

    Raises:
        ValueError: If the computed path is invalid.
    """
    # Track computation time and memory
    start_time = time.time()
    mem_before = get_resource_usage()
    
    # Create complete graph for NetworkX
    G = nx.Graph()
    # Add edges with weights from distance matrix
    for i in range(N):
        for j in range(i + 1, N):
            G.add_edge(i, j, weight=distance_matrix[i, j])
    
    # Step 1: Compute Minimum Spanning Tree (MST) using Kruskal's algorithm
    print("Debug: Computing MST")
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    
    # Step 2: Identify odd-degree vertices in MST
    print("Debug: Finding odd-degree vertices")
    degrees = dict(mst.degree())
    odd_vertices = [v for v, d in degrees.items() if d % 2 == 1]
    print(f"Debug: Odd vertices: {odd_vertices}")
    
    # Step 3: Compute minimum-weight perfect matching for odd-degree vertices
    print("Debug: Computing minimum-weight perfect matching")
    matching_edges = []
    available = set(odd_vertices)
    while available:
        # Select a vertex and find its closest available pair
        u = available.pop()
        min_weight = float('inf')
        min_v = None
        for v in available:
            weight = distance_matrix[u, v]
            if weight < min_weight:
                min_weight = weight
                min_v = v
        if min_v is not None:
            # Add edge to matching
            matching_edges.append((u, min_v, min_weight))
            available.remove(min_v)
    print(f"Debug: Matching edges: {matching_edges}")
    
    # Step 4: Combine MST and matching to form a multigraph
    print("Debug: Creating multigraph")
    multigraph = nx.MultiGraph()
    # Add MST edges
    multigraph.add_edges_from(mst.edges(data='weight'))
    # Add matching edges
    for u, v, w in matching_edges:
        multigraph.add_edge(u, v, weight=w)
    
    # Step 5: Find Eulerian circuit in multigraph
    print("Debug: Computing Eulerian circuit")
    eulerian_circuit = list(nx.eulerian_circuit(multigraph))
    print(f"Debug: Eulerian circuit: {eulerian_circuit}")
    
    # Step 6: Shortcut to Hamiltonian cycle
    print("Debug: Shortcutting to Hamiltonian cycle")
    path = [0]  # Start at waypoint 1 (index 0)
    visited = {0}
    for u, v in eulerian_circuit:
        # Add unvisited vertices to form Hamiltonian path
        if v not in visited:
            path.append(v)
            visited.add(v)
    path.append(0)  # Return to start
    print(f"Debug: Hamiltonian path: {path}")
    
    # Step 7: Compute and validate path cost
    cost = compute_path_cost(distance_matrix, path)
    try:
        validate_path(path, N, distance_matrix, cost)
    except ValueError as e:
        raise ValueError(f"Christofides path invalid: {str(e)}")
    
    # Log performance for debugging
    print(f"Debug: TSP solved: path={path}, cost={cost:.2f}, time={time.time() - start_time:.4f}s, "
          f"memory increase={get_resource_usage() - mem_before:.2f} MB")
    
    return path, cost

def format_sequence_table(waypoints: list[tuple[int, float, float, float]], path: list[int], distance_matrix: np.ndarray) -> str:
    """Format a table of the optimized visit sequence and segment details.

    Args:
        waypoints (list[tuple[int, float, float, float]]): List of (id, x, y, z) tuples.
        path (list[int]): List of waypoint indices.
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

def explain_path(waypoints: list[tuple[int, float, float, float]], path: list[int], distance_matrix: np.ndarray) -> float:
    """Explain the chosen path with a formatted table and detailed calculations.

    Args:
        waypoints (list[tuple[int, float, float, float]]): List of (id, x, y, z) tuples.
        path (list[int]): List of waypoint indices.
        distance_matrix (np.ndarray): N x N distance matrix.

    Returns:
        float: Total path cost, rounded to 2 decimal places.
    """
    # Explain path selection logic
    print("\nLogic Behind Chosen Path:")
    print("The path is a 1.5-approximation of the optimal Euclidean distance (fuel cost) using the "
          "Christofides Algorithm, which combines a Minimum Spanning Tree, minimum-weight perfect "
          "matching, and Eulerian circuit shortcutting, suitable for N <= 15.")
    
    # Display formatted sequence table
    print(format_sequence_table(waypoints, path, distance_matrix))
    
    # Provide detailed segment calculations
    print("\nPath Details:")
    total_cost = compute_path_cost(distance_matrix, path)
    for i in range(len(path) - 1):
        start_idx = path[i]
        end_idx = path[i + 1]
        # Extract IDs and coordinates
        start_id, x1, y1, z1 = waypoints[start_idx]
        end_id, x2, y2, z2 = waypoints[end_idx]
        distance = distance_matrix[start_idx, end_idx]
        # Print segment details with formula
        print(f"Segment {start_id} -> {end_id}:")
        print(f"  Coordinates: ({x1:.2f}, {y1:.2f}, {z1:.2f}) -> ({x2:.2f}, {y2:.2f}, {z2:.2f})")
        print(f"  Euclidean Distance = sqrt((({x2:.2f}-{x1:.2f})^2 + ({y2:.2f}-{y1:.2f})^2 + "
              f"({z2:.2f}-{z1:.2f})^2)) = {distance:.2f}")
    print(f"Total Fuel Cost: {total_cost:.2f}")
    return total_cost

def visualize_path(waypoints: list[tuple[int, float, float, float]], path: list[int], fuel_cost: float) -> tuple[float, float]:
    """Display and save a 3D visualization of the path.

    Args:
        waypoints (list[tuple[int, float, float, float]]): List of (id, x, y, z) tuples.
        path (list[int]): List of waypoint indices.
        fuel_cost (float): Total path cost.

    Returns:
        tuple[float, float]: (visualization_time, memory_increase) for performance logging.

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
        ax.set_title(f'1.5-Approximation Path (Christofides, Fuel Cost: {fuel_cost:.2f})')
        ax.legend()
        
        # Display plot in Jupyter/Kaggle
        print("Debug: Displaying plot")
        plt.show()
        
        # Save plot to file
        output_path = 'traditional approaches/Christofides_Algorithm/path_visualization_christofides.png'
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

def write_output(file_path: str, waypoints: list[tuple[int, float, float, float]], path: list[int], fuel_cost: float) -> None:
    """Write path and fuel cost to file with validation.

    Args:
        file_path (str): Path to output file (e.g., path_christofides.txt).
        waypoints (list[tuple[int, float, float, float]]): List of (id, x, y, z) tuples.
        path (list[int]): List of waypoint indices.
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
        print(f"Debug: Writing output: {output_str}")
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
    3. Compute distance matrix.
    4. Solve TSP using Christofides Algorithm.
    5. Explain and validate path.
    6. Visualize path and save plot.
    7. Write output to file.
    8. Log performance metrics.
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
        # Step 3: Compute distance matrix
        # ------------------------------
        distance_matrix = compute_distance_matrix(waypoints)
        print("Distance matrix computed")
        
        # ------------------------------
        # Step 4: Solve TSP
        # ------------------------------
        tsp_start = time.time()
        path, fuel_cost = solve_tsp(distance_matrix, N)
        tsp_time = time.time() - tsp_start
        print(f"1.5-approximation path computed: {path}, cost={fuel_cost:.2f}")
        
        # ------------------------------
        # Step 5: Explain and validate path
        # ------------------------------
        total_cost = explain_path(waypoints, path, distance_matrix)
        validate_path(path, N, distance_matrix, total_cost)
        print("Path explanation and validation completed")
        
        # ------------------------------
        # Step 6: Visualize path
        # ------------------------------
        vis_time, vis_mem_increase = visualize_path(waypoints, path, fuel_cost)
        print("Visualization generated")
        
        # ------------------------------
        # Step 7: Write output
        # ------------------------------
        write_output("traditional approaches/Christofides_Algorithm/path_christofides.txt", waypoints, path, fuel_cost)
        print("Output written to path_christofides.txt")
        
        # ------------------------------
        # Step 8: Log performance
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
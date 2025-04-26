# # Standard library imports for mathematical operations, timing, system resources, and randomization
# import math  # For sqrt in Euclidean distance calculations
# import time  # For measuring execution time of TSP, visualization, and total runtime
# import psutil  # For monitoring memory usage to report performance metrics
# import os  # For file path handling (e.g., reading waypoints.txt, writing path_ga.txt)
# import random  # For generating random tours, crossover, and mutation in Genetic Algorithm

# # Third-party imports for numerical computations and visualization
# import numpy as np  # For efficient array-based distance matrix operations
# import matplotlib.pyplot as plt  # For creating and displaying 3D path plots
# from mpl_toolkits.mplot3d import Axes3D  # For 3D axis support in visualizations

# # Enable inline plotting for Jupyter/Kaggle environments (not needed for standalone execution)
# %matplotlib inline

# def read_waypoints(file_path: str) -> list[tuple[int, float, float, float]]:
#     """Read waypoints from file with strict validation.

#     Args:
#         file_path (str): Path to waypoints file.

#     Returns:
#         list[tuple[int, float, float, float]]: List of (id, x, y, z) tuples for each waypoint.

#     Raises:
#         FileNotFoundError: If file_path does not exist.
#         ValueError: If file format, IDs, or coordinates are invalid.
#     """
#     # Initialize empty list for waypoints
#     waypoints = []
#     try:
#         # Read non-empty lines from file
#         with open(file_path, 'r') as f:
#             lines = [line.strip() for line in f if line.strip()]
#         # Parse each line
#         for line in lines:
#             # Validate line format: exactly 4 components (id, x, y, z)
#             parts = line.split()
#             if len(parts) != 4:
#                 raise ValueError(f"Invalid waypoint format in line '{line}': expected 'id x y z'")
#             try:
#                 # Convert to floats, ensuring ID is a positive integer
#                 id, x, y, z = map(float, parts)
#             except ValueError:
#                 raise ValueError(f"Invalid numerical values in line '{line}': expected integer ID and float coordinates")
#             if not id.is_integer() or id < 1:
#                 raise ValueError(f"Invalid waypoint ID {id} in line '{line}': must be a positive integer")
#             # Store waypoint as tuple
#             waypoints.append((int(id), x, y, z))
        
#         # Validate number of waypoints (5 to 15 per problem constraints)
#         N = len(waypoints)
#         if N < 5 or N > 15:
#             raise ValueError(f"Invalid number of waypoints {N}: must be between 5 and 15")
#         # Ensure unique, consecutive IDs from 1 to N
#         ids = set(wp[0] for wp in waypoints)
#         if len(ids) != N or max(ids) != N or min(ids) != 1:
#             raise ValueError(f"Waypoint IDs must be unique integers from 1 to {N}, got {sorted(ids)}")
#         # Validate coordinate bounds: [-1000, 1000]
#         for wp in waypoints:
#             _, x, y, z = wp
#             if not (-1000 <= x <= 1000 and -1000 <= y <= 1000 and -1000 <= z <= 1000):
#                 raise ValueError(f"Coordinates out of bounds in waypoint {wp}: must be in [-1000, 1000]")
#         return waypoints
#     except FileNotFoundError:
#         raise FileNotFoundError(f"waypoints.txt not found at {file_path}")
#     except Exception as e:
#         raise ValueError(f"Error reading waypoints: {str(e)}")

# def compute_distance_matrix(waypoints: list[tuple[int, float, float, float]]) -> np.ndarray:
#     """Calculate Euclidean distance matrix for waypoints.

#     Args:
#         waypoints (list[tuple[int, float, float, float]]): List of (id, x, y, z) tuples.

#     Returns:
#         np.ndarray: N x N matrix of Euclidean distances (float32).
#     """
#     # Get number of waypoints
#     N = len(waypoints)
#     # Initialize NxN distance matrix with zeros
#     distance_matrix = np.zeros((N, N), dtype=np.float32)
#     # Compute Euclidean distances for non-diagonal elements
#     for i in range(N):
#         for j in range(N):
#             if i != j:
#                 # Extract coordinates of waypoints i and j
#                 _, x1, y1, z1 = waypoints[i]
#                 _, x2, y2, z2 = waypoints[j]
#                 # Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
#                 distance_matrix[i, j] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
#     return distance_matrix

# def compute_path_cost(distance_matrix: np.ndarray, path: list[int]) -> float:
#     """Calculate total Euclidean distance of a path, rounded to 2 decimal places.

#     Args:
#         distance_matrix (np.ndarray): N x N distance matrix.
#         path (list[int]): List of waypoint indices forming the path.

#     Returns:
#         float: Total path cost, rounded to 2 decimal places.
#     """
#     # Initialize cost
#     cost = 0.0
#     # Sum distances between consecutive waypoints
#     for i in range(len(path) - 1):
#         cost += distance_matrix[path[i], path[i + 1]]
#     # Round to 2 decimal places for output consistency
#     return round(cost, 2)

# def validate_path(path: list[int], N: int, distance_matrix: np.ndarray, reported_cost: float) -> None:
#     """Validate path for uniqueness, cycle, and cost accuracy.

#     Args:
#         path (list[int]): List of waypoint indices.
#         N (int): Number of waypoints.
#         distance_matrix (np.ndarray): N x N distance matrix.
#         reported_cost (float): Reported path cost.

#     Raises:
#         ValueError: If path length, visits, or cycle are invalid.
#     """
#     # Check path length: N waypoints + return to start
#     if len(path) != N + 1:
#         raise ValueError(f"Path length {len(path)} does not match N+1 ({N+1})")
#     # Verify each waypoint visited exactly once
#     visited = set(path[:-1])
#     if len(visited) != N or visited != set(range(N)):
#         raise ValueError(f"Path does not visit each waypoint exactly once: {visited}")
#     # Ensure path forms a cycle
#     if path[0] != path[-1]:
#         raise ValueError(f"Path does not return to start: start={path[0]}, end={path[-1]}")
#     # Warn if reported cost differs (heuristic may not match exactly)
#     computed_cost = compute_path_cost(distance_matrix, path)
#     reported_cost = round(reported_cost, 2)
#     if abs(computed_cost - reported_cost) > 1e-3:
#         print(f"Warning: Reported cost {reported_cost:.2f} differs from computed cost {computed_cost:.2f} (heuristic)")
#     else:
#         print(f"Debug: Path validation passed: cost={computed_cost:.2f}")

# def solve_tsp(distance_matrix: np.ndarray, N: int) -> tuple[list[int], float]:
#     """Solve TSP using Genetic Algorithm for near-optimal path.

#     Evolves a population of tours via tournament selection, ordered crossover, and swap mutation.
#     Population size, generations, and mutation rate tuned for N <= 15.

#     Args:
#         distance_matrix (np.ndarray): N x N distance matrix.
#         N (int): Number of waypoints.

#     Returns:
#         tuple[list[int], float]: (path, cost) where path is a list of indices and cost is the total distance.

#     Raises:
#         ValueError: If the computed path is invalid.
#     """
#     # Track computation time and memory
#     start_time = time.time()
#     mem_before = get_resource_usage()
    
#     def create_random_tour() -> list[int]:
#         """Create a random valid tour starting and ending at index 0."""
#         # Generate shuffled tour for waypoints 1 to N-1
#         tour = list(range(1, N))
#         random.shuffle(tour)
#         # Ensure tour starts and ends at 0
#         return [0] + tour + [0]
    
#     def ordered_crossover(parent1: list[int], parent2: list[int]) -> list[int]:
#         """Ordered crossover (OX): Preserves subtours from parents while maintaining order."""
#         # Select random segment from parent1
#         start, end = sorted(random.sample(range(1, N), 2))
#         child = [None] * N
#         child[start:end] = parent1[start:end]
        
#         # Fill remaining positions with cities from parent2 in order, skipping those in child
#         parent2_cities = [city for city in parent2[1:-1] if city not in child[start:end]]
#         pos = end
#         for city in parent2_cities:
#             # Find next available position
#             while child[pos % N] is not None:
#                 pos += 1
#             child[pos % N] = city
        
#         # Ensure all positions are filled with remaining cities
#         for i in range(N):
#             if child[i] is None:
#                 child[i] = parent2_cities.pop(0)
        
#         # Return complete tour with start/end at 0
#         return [0] + child[1:] + [0]
    
#     def mutate(tour: list[int]) -> list[int]:
#         """Swap mutation: Swap two cities with probability mutation_rate."""
#         tour = tour[:]
#         if random.random() < mutation_rate:
#             # Select two distinct positions (excluding start/end)
#             i, j = random.sample(range(1, N), 2)
#             tour[i], tour[j] = tour[j], tour[i]
#         return tour
    
#     def tournament_selection(population: list[list[int]], k: int = 5) -> list[int]:
#         """Select best tour from k random candidates based on cost."""
#         # Randomly sample k tours from population
#         candidates = random.sample(population, k)
#         # Return tour with minimum cost
#         return min(candidates, key=lambda x: compute_path_cost(distance_matrix, x))
    
#     # Genetic Algorithm parameters
#     population_size = 100  # Number of tours in population
#     generations = 1000     # Number of evolution iterations
#     mutation_rate = 0.05   # Probability of mutating a tour
    
#     # Initialize population with random valid tours
#     population = [create_random_tour() for _ in range(population_size)]
    
#     # Verify initial population for correctness
#     for tour in population:
#         if set(tour[:-1]) != set(range(N)) or tour[0] != 0 or tour[-1] != 0:
#             raise ValueError(f"Invalid initial tour: {tour}")
    
#     # Find initial best tour
#     best_path = min(population, key=lambda x: compute_path_cost(distance_matrix, x))
#     best_cost = compute_path_cost(distance_matrix, best_path)
    
#     # Main Genetic Algorithm loop
#     for _ in range(generations):
#         # Create new population
#         new_population = []
#         for _ in range(population_size):
#             # Select parents via tournament selection
#             parent1 = tournament_selection(population)
#             parent2 = tournament_selection(population)
#             # Generate child via ordered crossover
#             child = ordered_crossover(parent1, parent2)
#             # Apply mutation
#             child = mutate(child)
#             # Verify child validity, fallback to random tour if invalid
#             if set 🙂

# System: It looks like the response was cut off. Below, I’ll complete the enhanced code with comments for the provided Genetic Algorithm (GA) TSP solver, ensuring consistency with prior submissions (dynamic programming, Nearest Neighbor, Branch and Bound, and Simulated Annealing). The code will maintain the same functionality, producing path_ga.txt with space-separated IDs and cost, a 3D visualization saved as path_visualization_ga.png, and detailed console output. The comments will enhance readability for the GitHub repository, align with the 20-mark code quality evaluation, and highlight the GA’s features for hackathon judges.

# ### Approach (Continued)
# - **Complete the solve_tsp Function**: Add comments for the remaining GA logic, including child validation, population updates, and performance logging.
# - *Other Functions*: Enhance comments in format_sequence_table, explain_path, visualize_path, write_output, get_resource_usage, and main to clarify their roles, align with prior submissions, and explain key operations (e.g., visualization, output validation).
# - *Consistency*: Use # for inline comments, section headers with dashes in main, and action-oriented phrasing, matching the style of previous artifacts (e.g., c4880438-018e-4b2c-8a26-af31d86bb234, 524e6e59-61f6-42ed-8931-7219d8a01bb6).
# - *Judge Appeal*: Emphasize the GA’s evolutionary approach (tournament selection, ordered crossover, mutation), its balance of speed and solution quality for \( N \leq 15 \), and hackathon compliance (e.g., output format, robust validation).

# ### Updated Code with Enhanced Comments
# Below is the complete code with added and improved comments. The artifact ID is new since this is a distinct GA implementation, unrelated to prior artifacts.

# <xaiArtifact artifact_id="50df0ec8-89e9-4793-bcd4-c5cc0e4dfc25" artifact_version_id="e4a0a6eb-a81d-4867-b2fe-846aa2459483" title="tsp_ga_solver.py" contentType="text/python">
# # Standard library imports for mathematical operations, timing, system resources, and randomization
import math  # For sqrt in Euclidean distance calculations
import time  # For measuring execution time of TSP, visualization, and total runtime
import psutil  # For monitoring memory usage to report performance metrics
import os  # For file path handling (e.g., reading waypoints.txt, writing path_ga.txt)
import random  # For generating random tours, crossover, and mutation in Genetic Algorithm

# Third-party imports for numerical computations and visualization
import numpy as np  # For efficient array-based distance matrix operations
import matplotlib.pyplot as plt  # For creating and displaying 3D path plots
from mpl_toolkits.mplot3d import Axes3D  # For 3D axis support in visualizations


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
    # Warn if reported cost differs (heuristic may not match exactly)
    computed_cost = compute_path_cost(distance_matrix, path)
    reported_cost = round(reported_cost, 2)
    if abs(computed_cost - reported_cost) > 1e-3:
        print(f"Warning: Reported cost {reported_cost:.2f} differs from computed cost {computed_cost:.2f} (heuristic)")
    else:
        print(f"Debug: Path validation passed: cost={computed_cost:.2f}")

def solve_tsp(distance_matrix: np.ndarray, N: int) -> tuple[list[int], float]:
    """Solve TSP using Genetic Algorithm for near-optimal path.

    Evolves a population of tours via tournament selection, ordered crossover, and swap mutation.
    Population size, generations, and mutation rate tuned for N <= 15.

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
    
    def create_random_tour() -> list[int]:
        """Create a random valid tour starting and ending at index 0."""
        # Generate shuffled tour for waypoints 1 to N-1
        tour = list(range(1, N))
        random.shuffle(tour)
        # Ensure tour starts and ends at 0
        return [0] + tour + [0]
    
    def ordered_crossover(parent1: list[int], parent2: list[int]) -> list[int]:
        """Ordered crossover (OX): Preserves subtours from parents while maintaining order."""
        # Select random segment from parent1
        start, end = sorted(random.sample(range(1, N), 2))
        child = [None] * N
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions with cities from parent2 in order, skipping those in child
        parent2_cities = [city for city in parent2[1:-1] if city not in child[start:end]]
        pos = end
        for city in parent2_cities:
            # Find next available position
            while child[pos % N] is not None:
                pos += 1
            child[pos % N] = city
        
        # Ensure all positions are filled with remaining cities
        for i in range(N):
            if child[i] is None:
                child[i] = parent2_cities.pop(0)
        
        # Return complete tour with start/end at 0
        return [0] + child[1:] + [0]
    
    def mutate(tour: list[int]) -> list[int]:
        """Swap mutation: Swap two cities with probability mutation_rate."""
        tour = tour[:]
        if random.random() < mutation_rate:
            # Select two distinct positions (excluding start/end)
            i, j = random.sample(range(1, N), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour
    
    def tournament_selection(population: list[list[int]], k: int = 5) -> list[int]:
        """Select best tour from k random candidates based on cost."""
        # Randomly sample k tours from population
        candidates = random.sample(population, k)
        # Return tour with minimum cost
        return min(candidates, key=lambda x: compute_path_cost(distance_matrix, x))
    
    # Genetic Algorithm parameters
    population_size = 100  # Number of tours in population
    generations = 1000     # Number of evolution iterations
    mutation_rate = 0.05   # Probability of mutating a tour
    
    # Initialize population with random valid tours
    population = [create_random_tour() for _ in range(population_size)]
    
    # Verify initial population for correctness
    for tour in population:
        if set(tour[:-1]) != set(range(N)) or tour[0] != 0 or tour[-1] != 0:
            raise ValueError(f"Invalid initial tour: {tour}")
    
    # Find initial best tour
    best_path = min(population, key=lambda x: compute_path_cost(distance_matrix, x))
    best_cost = compute_path_cost(distance_matrix, best_path)
    
    # Main Genetic Algorithm loop
    for _ in range(generations):
        # Create new population
        new_population = []
        for _ in range(population_size):
            # Select parents via tournament selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            # Generate child via ordered crossover
            child = ordered_crossover(parent1, parent2)
            # Apply mutation
            child = mutate(child)
            # Verify child validity, fallback to random tour if invalid
            if set(child[:-1]) != set(range(N)) or child[0] != 0 or child[-1] != 0:
                child = create_random_tour()
            new_population.append(child)
        # Update population
        population = new_population
        # Update best tour if a better one is found
        current_best = min(population, key=lambda x: compute_path_cost(distance_matrix, x))
        current_cost = compute_path_cost(distance_matrix, current_best)
        if current_cost < best_cost:
            best_path = current_best[:]
            best_cost = current_cost
    
    # Validate final path
    cost = round(best_cost, 2)
    try:
        validate_path(best_path, N, distance_matrix, cost)
    except ValueError as e:
        raise ValueError(f"Genetic Algorithm path invalid: {str(e)}")
    
    # Log performance for debugging
    print(f"Debug: TSP solved: path={best_path}, cost={cost:.2f}, time={time.time() - start_time:.4f}s, "
          f"memory increase={get_resource_usage() - mem_before:.2f} MB")
    
    return best_path, cost

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
    print("The path minimizes the total Euclidean distance (fuel cost) using a Genetic Algorithm, "
          "a heuristic that evolves a population of tours via tournament selection, ordered crossover, "
          "and swap mutation, tuned for N <= 15.")
    
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
        ax.set_title(f'Near-Optimal Path (Genetic Algorithm, Fuel Cost: {fuel_cost:.2f})')
        ax.legend()
        
        # Display plot in Jupyter/Kaggle
        print("Debug: Displaying plot")
        plt.show()
        
        # Save plot to file
        output_path = 'traditional approaches/Genetic_Algorithm_Implementation/path_visualization_ga.png'
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
        file_path (str): Path to output file (e.g., path_ga.txt).
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
    4. Solve TSP using Genetic Algorithm.
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
        print(f"Near-optimal path computed: {path}, cost={fuel_cost:.2f}")
        
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
        write_output("traditional approaches/Genetic_Algorithm_Implementation/path_ga.txt", waypoints, path, fuel_cost)
        print("Output written to path_ga.txt")
        
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
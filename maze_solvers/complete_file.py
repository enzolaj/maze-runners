import tkinter as tk
from tkinter import ttk
import random
import collections
import heapq
import math

# --- Configuration ---
CELL_SIZE = 20 # GUI
MAZE_WIDTH = 25 # Physical maze size
MAZE_HEIGHT = 25 # Physical maze size

# --- Monte Carlo Config ---
# These constants control the "Monte Carlo Lookahead" algorithm.
# Higher values are "smarter" but take longer per step.
MONTE_CARLO_ROLLOUTS = 10 # How many random simulations to run per possible move
MONTE_CARLO_DEPTH = 20    # How many steps to "look ahead" in each simulation

# --- Color Scheme ---
BG_COLOR = "#2c3e50"
WALL_COLOR = "#bdc3c7"
ROBOT_COLOR = "#f1c40f"
PLAYER_COLOR = "#3498db"
START_COLOR = "#1abc9c"
GOAL_COLOR = "#e74c3c"
REWARD_COLOR = "#e67e22"
FINAL_PATH_COLOR = "#f39c12"    # (Offline) The optimal path computed in advance
SEARCH_AREA_COLOR = "#34495e"   # (Online/Offline) All cells visited by the search
KNOWN_PATH_COLOR = "#95a5a6"    # (Online) The robot's meandering exploration trail
DEFAULT_CELL_COLOR = "#ecf0f1"  # Cell color for weight 1 or edge-weighted mazes
WEIGHT_COLOR_LOW = "#3498db"    # Blue (cheap terrain)
WEIGHT_COLOR_MID = "#2ecc71"    # Green (medium terrain)
WEIGHT_COLOR_HIGH = "#e74c3c"   # Red (expensive terrain)


class Maze:
    """
    Generates and stores the maze structure using a directional adjacency representation.
    
    Maze Representation:
      - self.grid is a defaultdict(set) where each cell (x, y) maps to a set of
        directions {'N', 'S', 'E', 'W'} representing open passages (knocked-down walls).
      - Initially, all cells have empty sets (all walls intact).
      - Generation algorithms "carve passages" by adding directions to cells' sets.
      - Example: if self.grid[(3, 5)] = {'N', 'E'}, cell (3,5) has open passages
        north and east (walls remain on south and west sides).
    
    Weight Systems:
      - Node-weighted: Each cell (x, y) has a terrain cost stored in self.node_weights.
        Cost is paid when ENTERING a cell. Used by DFS, BFS, Random Prim's, Greedy Frontier.
      - Edge-weighted: Each edge ((x1,y1), (x2,y2)) has a path cost in self.edge_weights.
        Cost is paid when MOVING BETWEEN cells. Used by Kruskal's MST generator.
    
    Coordinates: (x, y) where x is column (horizontal, 0 to width-1) and
                 y is row (vertical, 0 to height-1). Origin (0, 0) is top-left.
    """
    def __init__(self, width, height, gen_method='DFS', loop_percent=0, num_rewards=0, max_weight=1):
        """
        Initializes a maze and generates it using the specified algorithm.
        
        The constructor orchestrates the complete maze generation pipeline:
          1. Initialize grid (all walls present: empty sets for all cells)
          2. Run the chosen generation algorithm to carve a perfect maze (spanning tree)
          3. Optionally add loops by removing additional walls (creates cycles)
          4. Assign terrain weights (for node-weighted mazes)
          5. Place reward collectibles at random locations
        
        Parameters:
          width (int): Number of cells horizontally (columns)
          height (int): Number of cells vertically (rows)
          gen_method (str): Generation algorithm - 'DFS', 'BFS', 'Random Prim's',
                           'Greedy Frontier', or 'Kruskal's MST'
          loop_percent (int): Percentage of cells where we knock down extra walls
                             to create multiple paths (0 = perfect maze, no loops)
          num_rewards (int): Number of collectible reward cells to place
          max_weight (int): Maximum terrain cost for node-weighted mazes (1 = uniform cost)
        
        Implementation Flow:
          - Perfect maze generation creates a spanning tree (exactly one path between
            any two cells, width * height - 1 edges in the connectivity graph)
          - MST generators (Kruskal) use edge-weights; others use node-weights
          - Loop addition transforms perfect maze into imperfect maze by adding cycles
        """
        self.width = width
        self.height = height
        self.grid = collections.defaultdict(set)

        # 'node_weights' are "terrain costs" (cost to ENTER a cell).
        self.node_weights = {} # (x, y) -> cost
        # 'edge_weights' are "path costs" (cost to MOVE BETWEEN two cells).
        self.edge_weights = {} # ((x1,y1), (x2,y2)) -> cost
        # This flag tells the Robot which cost to use.
        self.weight_type = 'node'

        self.rewards = set()
        self.max_weight = max(1, max_weight)

        # --- Generation Pipeline ---
        if gen_method == 'Kruskal\'s MST':
            self.weight_type = 'edge'
            self._generate_kruskal()
        elif gen_method == 'Greedy Frontier':
            self.weight_type = 'node'
            self._generate_greedy_frontier()
            self._populate_node_weights(max_weight)
        elif gen_method == 'Random Prim\'s':
            self.weight_type = 'node'
            self._generate_random_prims()
            self._populate_node_weights(max_weight)
        elif gen_method == 'BFS': 
            self.weight_type = 'node'
            self._generate_bfs()
            self._populate_node_weights(max_weight)
        else: # Default to DFS
            self.weight_type = 'node'
            self._generate_dfs()
            self._populate_node_weights(max_weight)

        # After generation, optionally knock down walls to create loops.
        if loop_percent > 0:
            self._add_loops(loop_percent)

        # Place rewards. This is independent of generation method.
        self._populate_rewards(num_rewards)

    def _generate_dfs(self):
        """Generates a "perfect" maze using Randomized Depth-First Search.

        High-level overview:
          - We start with a grid where ALL cells are surrounded by walls
            (self.grid is a defaultdict(set), so each cell initially has an 
            empty set = no passages = walls on all 4 sides).
          - We randomly pick a starting cell and perform DFS with backtracking.
          - At each step, we pick an unvisited neighbor and "carve a passage"
            by adding the direction to both cells' sets (e.g., adding 'N' to 
            current cell and 'S' to neighbor removes the wall between them).
          - When stuck (no unvisited neighbors), we backtrack until we find a
            cell with unvisited neighbors, then continue carving.
          - The result is a "perfect maze" - a spanning tree where there's
            exactly one path between any two cells (no loops, no unreachable areas).
        """
        # Track which cells have been visited during maze generation
        visited = set()
        # Initialize stack with a random starting cell
        stack = [(random.randint(0, self.width - 1), random.randint(0, self.height - 1))]
        visited.add(stack[0])
        
        # Continue until we've backtracked to the start and exhausted all paths
        while stack:
            # Peek at the current cell (don't pop yet - we might add more to stack)
            current_x, current_y = stack[-1]
            
            # Find all unvisited neighbors in the 4 cardinal directions
            neighbors = []
            # (dx, dy, direction_from_current, direction_from_neighbor)
            for dx, dy, direction, opposite in [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'),
                                                (-1, 0, 'W', 'E'), (1, 0, 'E', 'W')]:
                nx, ny = current_x + dx, current_y + dy
                # Check if neighbor is in bounds and unvisited
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, direction, opposite))
            
            # If there are unvisited neighbors, carve a passage to one
            if neighbors:
                # Pick a random neighbor to maintain maze randomness
                nx, ny, direction, opposite = random.choice(neighbors)
                # Carve passage: remove wall from current cell in the chosen direction
                self.grid[(current_x, current_y)].add(direction)
                # Remove wall from neighbor cell in the opposite direction
                self.grid[(nx, ny)].add(opposite)
                # Mark neighbor as visited and push onto stack for DFS
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                # No unvisited neighbors - backtrack by popping from stack
                stack.pop()

    def _generate_random_prims(self):
        """Generates a "perfect" maze using Prim's Minimum Spanning Tree Algorithm.

        Maze Data Structure Context:
          - self.grid starts with all cells having empty sets (all walls present).
          - We carve passages by adding directions to cells' sets (e.g., 'N', 'S', 'E', 'W').
          - Adding 'E' to cell (x, y) and 'W' to cell (x+1, y) removes the wall between them.
          - Edge weights stored in self.edge_weights for pathfinding cost calculations.

        Algorithm (Prim's MST):
          1. Start with a random cell marked as visited
          2. Add all edges from start to unvisited neighbors to priority queue (with weights)
          3. While priority queue is not empty:
             a. Pop edge with minimum weight from queue
             b. If destination cell already visited, skip (stale entry)
             c. Mark destination as visited and carve passage
             d. Add all edges from destination to unvisited neighbors
          4. Result: Minimum Spanning Tree (minimizes total edge weight)
        
        Edge Weights:
          - Each edge assigned random weight + distance component
          - Formula: weight = random(1-10) + 0.5 * Euclidean_distance_from_start
          - Biases maze to expand near start first, then radiate outward
          - Creates more realistic "growing" pattern
        
        Mathematical Property:
          - Produces true Minimum Spanning Tree (greedy is optimal)
          - Total edges carved: width * height - 1 (perfect maze property)
          - Maze character: clusters near start, longer corridors further away
        
        Integration:
          - Called during Maze.__init__ when gen_method='Random Prim\'s'
          - Sets self.weight_type = 'edge' if using edge weights for pathfinding
          - Works with directional adjacency representation in self.grid
        """
        # Step 1: Initialize visited set and pick random starting cell
        visited = set()
        start_x, start_y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        start = (start_x, start_y)
        visited.add(start)
        
        # Step 2: Initialize priority queue with edges from start
        # Priority queue stores: (weight, from_cell, to_cell, direction, opposite_direction)
        frontier = []
        
        def add_frontier_edges(cx, cy):
            """Add all edges from (cx, cy) to unvisited neighbors to priority queue."""
            for dx, dy, direction, opposite in [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'),
                                                 (-1, 0, 'W', 'E'), (1, 0, 'E', 'W')]:
                nx, ny = cx + dx, cy + dy
                # Check if neighbor is in bounds and not yet visited
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    # Calculate edge weight: random + distance component
                    # Distance from start creates outward-radiating growth pattern
                    euclidean_dist = ((nx - start[0])**2 + (ny - start[1])**2)**0.5
                    weight = random.randint(1, 10) + 0.5 * euclidean_dist
                    
                    # Push edge to priority queue (heap orders by weight)
                    heapq.heappush(frontier, (weight, (cx, cy), (nx, ny), direction, opposite))
        
        # Add initial edges from start cell
        add_frontier_edges(start_x, start_y)
        
        # Step 3: Greedily build MST by always selecting lightest edge
        while frontier:
            # Pop edge with minimum weight (greedy choice)
            weight, cell, neighbor, direction, opposite = heapq.heappop(frontier)
            
            # Skip if destination already visited (stale entry from earlier insertion)
            if neighbor in visited:
                continue
            
            # Accept this edge: mark destination as visited
            visited.add(neighbor)
            
            # Carve passage bidirectionally between cell and neighbor
            cx, cy = cell
            nx, ny = neighbor
            self.grid[cell].add(direction)     # Add direction from cell
            self.grid[neighbor].add(opposite)   # Add opposite direction from neighbor
            
            # Add all edges from newly added cell to unvisited neighbors
            add_frontier_edges(nx, ny)

    def _generate_bfs(self):
        """Generates a maze using randomized Breadth-First Search.

        Maze Data Structure Context:
          - self.grid[(x, y)] stores a set of open directions for each cell.
          - Empty sets mean all walls intact; adding 'N'/'S'/'E'/'W' carves passages.
          - Passages are bidirectional: carving 'N' from (x, y) requires 'S' in (x, y-1).

        Algorithm (BFS-based Spanning Tree Generation):
          1. Start from a random cell, mark it visited, add to queue
          2. While queue is not empty:
             a. Dequeue a cell
             b. Shuffle the 4 cardinal directions (introduces randomness)
             c. For each direction, if neighbor is unvisited:
                - Carve passage between current and neighbor (bidirectional)
                - Mark neighbor as visited
                - Enqueue neighbor
          3. Result: Level-order spanning tree with randomized branching
        
        Mathematical Property:
          - Creates spanning tree with shorter average path lengths
          - Tends to create more "room-like" structures vs. DFS's long corridors
          - Total edges: width * height - 1 (perfect maze)
        
        Integration:
          - Called during Maze.__init__ when gen_method='BFS'
          - Followed by _populate_node_weights for terrain cost assignment
          - Result stored in self.grid as directional adjacency sets
        """
        # Initialize visited set and queue with random starting cell
        visited = set()
        start = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        visited.add(start)
        queue = collections.deque([start])  # FIFO queue for level-order traversal

        # Process cells in breadth-first order
        while queue:
            # Dequeue the next cell to process (oldest added = breadth-first)
            cx, cy = queue.popleft()
            
            # Randomize direction order to introduce variety (key for maze generation)
            directions = [(0,-1,'N','S'), (0,1,'S','N'), (-1,0,'W','E'), (1,0,'E','W')]
            random.shuffle(directions)  # Without this, mazes would have predictable patterns
            
            # Try to expand in each direction (randomized order)
            for dx, dy, d1, d2 in directions:
                nx, ny = cx + dx, cy + dy
                # Check if neighbor is in bounds and hasn't been visited yet
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    # Carve passage bidirectionally between current and neighbor
                    self.grid[(cx, cy)].add(d1)   # Add direction from current cell
                    self.grid[(nx, ny)].add(d2)   # Add opposite direction from neighbor
                    # Mark neighbor as discovered and enqueue for later processing
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    def _generate_greedy_frontier(self):
        """Generates a maze using a greedy frontier approach with heuristic bias.

        Maze Data Structure Context:
          - self.grid[(x, y)] = set of directions where walls have been removed.
          - We start with all cells walled off (empty sets) and carve by adding directions.
          - Carving is bidirectional: adding 'E' to (x, y) requires adding 'W' to (x+1, y).

        Algorithm (Greedy Best-First Spanning Tree):
          1. Pick random start and goal anchor cells
          2. Initialize frontier with start cell
          3. While frontier is not empty:
             a. Sort frontier by Manhattan distance to goal (greedy heuristic)
             b. Pop the closest cell from frontier
             c. If it has unvisited neighbors:
                - Pick a random neighbor and carve passage to it
                - Add neighbor to frontier
                - Re-add current cell to frontier (can branch again later)
          4. Result: Tree with long straight corridors biased toward goal direction
        
        Heuristic Bias:
          - Manhattan distance h(cell) = |cell.x - goal.x| + |cell.y - goal.y|
          - Always expanding the frontier cell closest to goal creates directional bias
          - Results in fewer branches and longer corridors compared to random selection
        
        Mathematical Property:
          - Still produces a perfect maze (spanning tree, width * height - 1 edges)
          - Not a uniformly random tree - biased toward straight paths
          - Tends to create clear "main paths" with sparse branching
        
        Integration:
          - Called during Maze.__init__ when gen_method='Greedy Frontier'
          - Followed by _populate_node_weights for terrain costs
          - Produces distinctive maze topology: river-like main paths
        """
        # Initialize visited set
        visited = set()
        
        # Pick random start position
        start = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        
        # Pick random goal anchor (different from start) to bias tree growth
        goal = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        while goal == start:
            goal = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

        # Initialize frontier with start cell
        frontier = [start]
        visited.add(start)

        # Process frontier using greedy best-first selection
        while frontier:
            # Sort frontier by Manhattan distance to goal (greedy heuristic)
            # This biases maze to grow toward the goal, creating long corridors
            frontier.sort(key=lambda cell: abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]))
            current_x, current_y = frontier.pop(0)  # Take closest cell to goal

            # Find all unvisited neighbors of current cell
            neighbors = []
            for dx, dy, direction, opposite in [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'), (-1, 0, 'W', 'E'), (1, 0, 'E', 'W')]:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, direction, opposite))

            # If current cell has unvisited neighbors, expand the tree
            if neighbors:
                # Pick one random neighbor to connect (introduces some randomness)
                nx, ny, direction, opposite = random.choice(neighbors)
                
                # Carve passage bidirectionally
                self.grid[(current_x, current_y)].add(direction)  # Open passage from current
                self.grid[(nx, ny)].add(opposite)                 # Open passage from neighbor

                # Mark neighbor as visited
                visited.add((nx, ny))
                
                # Re-add current to frontier (may have more neighbors to explore)
                frontier.append((current_x, current_y))
                
                # Add new neighbor to frontier
                frontier.append((nx, ny))

    # --- Kruskal's Generator Helper Methods ---
    def _find_set(self, parent, u):
        """
        Find operation for Disjoint Set Union (Union-Find) data structure.
        
        Purpose:
          - Determines which set (connected component) a cell belongs to
          - Each set has a representative "root" element
          - Used by Kruskal's algorithm to detect if adding an edge would create a cycle
        
        Algorithm:
          - Follow parent pointers until we reach a node that is its own parent (root)
          - Path compression optimization: during traversal, point all nodes directly
            to the root to flatten the tree structure
          - Time complexity: O(α(n)) amortized per operation (α is inverse Ackermann,
            effectively constant for all practical values)
        
        Parameters:
          parent (dict): Maps each cell to its parent cell in the union-find forest
          u (tuple): Cell coordinate (x, y) to find the root of
        
        Returns:
          tuple: Root cell coordinate representing the set containing u
        
        Integration:
          - Called by _union_sets to check if two cells are in same component
          - Essential for Kruskal's MST algorithm to avoid cycles when adding edges
        """
        if parent[u] != u:
            parent[u] = self._find_set(parent, parent[u]) # Path compression
        return parent[u]

    def _union_sets(self, parent, rank, u, v):
        """
        Union operation for Disjoint Set Union, merges two connected components.
        
        Purpose:
          - Merges the sets containing cells u and v into a single set
          - Used by Kruskal's algorithm when adding an edge to the spanning tree
          - Maintains forest structure efficiently using union-by-rank heuristic
        
        Algorithm:
          1. Find roots of both cells using _find_set
          2. If roots are the same, cells already connected (would create cycle)
          3. Otherwise, merge the two trees by rank (attach shorter to taller)
          4. If ranks equal, attach arbitrarily and increment rank of new root
        
        Union-by-Rank Heuristic:
          - rank[root] is an upper bound on tree height
          - Always attach tree with smaller rank to tree with larger rank
          - Prevents degenerate linear chains, keeps trees balanced
          - Combined with path compression, gives O(α(n)) amortized time
        
        Parameters:
          parent (dict): Parent pointers for union-find structure
          rank (dict): Rank (approximate height) of each tree root
          u, v (tuple): Cell coordinates to connect
        
        Returns:
          bool: True if sets were merged (edge added), False if already connected
        
        Integration:
          - Called by _generate_kruskal for each edge in sorted order
          - Return value determines if edge should be added to maze
          - False return means edge would create cycle (rejected)
        """
        u_root, v_root = self._find_set(parent, u), self._find_set(parent, v)
        if u_root == v_root:
            return False # Already in the same set
        # Union by rank
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        elif rank[v_root] < rank[u_root]:
            parent[v_root] = u_root
        else:
            parent[v_root] = u_root
            rank[u_root] += 1
        return True

    def _generate_kruskal(self):
        """
        Generates a maze using Kruskal's Minimum Spanning Tree algorithm.

        Maze Data Structure Context:
          - self.grid[(x, y)] stores open directions for passages (initially all empty)
          - self.edge_weights stores edge costs: self.edge_weights[((x1,y1), (x2,y2))] = cost
          - This is an EDGE-WEIGHTED maze (different from node-weighted DFS/BFS/Prim's)
          - Costs are assigned to passages between cells, not to cells themselves

        Algorithm (Kruskal's MST):
          1. Generate all possible edges between adjacent cells (2 * width * height - width - height edges)
          2. Assign random weights to each edge (simulates wall "removal difficulty")
          3. Sort all edges by weight (ascending order)
          4. Initialize Disjoint Set Union (DSU) with each cell in its own set
          5. For each edge in sorted order:
             a. Check if endpoints are in different sets (using DSU find)
             b. If different, add edge (carve passage bidirectionally) and merge sets (union)
             c. If same set, skip edge (would create cycle)
          6. Stop when we have width * height - 1 edges (spanning tree complete)
        
        Mathematical Properties:
          - Produces a Minimum Spanning Tree of the grid graph with random edge weights
          - MST has exactly n-1 edges for n vertices (perfect maze property)
          - Greedy algorithm: locally optimal choices (lightest available edge) lead to
            globally optimal solution (minimum total weight spanning tree)
          - Time complexity: O(E log E) for sorting edges, where E ≈ 2*width*height
        
        Edge-Weight System:
          - Edge weights stored bidirectionally: weight(u→v) = weight(v→u)
          - Robot pathfinding uses edge weights to compute path costs
          - Visual representation: tiles colored by average incident edge weights
        
        Integration:
          - Called during Maze.__init__ when gen_method='Kruskal\'s MST'
          - Sets self.weight_type = 'edge' (tells Robot to use edge_weights, not node_weights)
          - No node weights populated (not applicable for edge-weighted mazes)
          - Loops can be added afterwards via _add_loops() to create shortcuts in MST
        """
        # Step 1: Create a list of all potential edges with random weights
        # Each edge connects two adjacent cells (only East and South to avoid duplicates)
        edges = []
        for y in range(self.height):
            for x in range(self.width):
                u = (x, y)
                
                # Add edge to the East (right neighbor)
                if x < self.width - 1:
                    v = (x + 1, y)
                    weight = random.randint(1, 10)  # Random edge weight (passage difficulty)
                    edges.append((weight, u, v))
                
                # Add edge to the South (down neighbor)
                if y < self.height - 1:
                    v = (x, y + 1)
                    weight = random.randint(1, 10)
                    edges.append((weight, u, v))
        
        # Step 2: Sort all edges by weight (ascending order)
        # Kruskal's greedy approach: always try lightest edges first
        edges.sort()  # Python sorts tuples by first element (weight)

        # Step 3: Initialize Disjoint Set Union (Union-Find) data structure
        # Initially, each cell is in its own set (disconnected components)
        parent = {}  # parent[node] -> representative of the set containing node
        rank = {}    # rank[node] -> approximate tree height for balancing
        for y in range(self.height):
            for x in range(self.width):
                node = (x, y)
                parent[node] = node  # Each node is its own parent initially
                rank[node] = 0       # All trees start with rank 0

        # Step 4: Process edges in sorted order, adding them if they don't create cycles
        num_edges = 0
        total_cells = self.width * self.height
        for weight, u, v in edges:
            # Try to merge the sets containing u and v
            # _union_sets returns True if they were in different sets (no cycle)
            if self._union_sets(parent, rank, u, v):
                # Edge accepted! Add to maze by carving passage and storing weight
                x1, y1 = u
                x2, y2 = v
                
                # Store edge weight bidirectionally for pathfinding
                self.edge_weights[(u, v)] = weight
                self.edge_weights[(v, u)] = weight

                # Carve passage based on edge orientation
                if x1 == x2:  # Vertical edge (same column)
                    if y1 < y2:
                        self.grid[u].add('S')  # u can go South to v
                        self.grid[v].add('N')  # v can go North to u
                    else:
                        self.grid[u].add('N')
                        self.grid[v].add('S')
                else:  # Horizontal edge (same row)
                    if x1 < x2:
                        self.grid[u].add('E')  # u can go East to v
                        self.grid[v].add('W')  # v can go West to u
                    else:
                        self.grid[u].add('W')
                        self.grid[v].add('E')

                num_edges += 1
                
                # Optimization: A spanning tree has exactly n-1 edges for n vertices
                # Stop early once we've connected all cells
                if num_edges >= total_cells - 1:
                    break

    def _add_loops(self, percentage):
        """
        Creates an "imperfect" maze by removing additional walls to introduce cycles.

        Maze Data Structure Context:
          - Perfect maze (spanning tree) has exactly one path between any two cells
          - self.grid[(x, y)] contains open directions; closed directions have walls
          - Adding a direction that doesn't exist creates a new passage (removes wall)
        
        Algorithm:
          1. Calculate target: num_walls_to_remove = (width * height * percentage) / 100
          2. For each removal attempt:
             a. Pick a random cell (x, y)
             b. Find all 4 directions that currently have walls (not in self.grid[(x, y)])
             c. Pick a random walled direction
             d. Carve passage bidirectionally (add direction to both cells)
        
        Mathematical Effect:
          - Perfect maze has width * height - 1 edges (tree property)
          - Each loop adds one edge, creating a cycle
          - With k loops: total edges = width * height - 1 + k
          - Graph becomes cyclic: multiple paths exist between some cell pairs
        
        Pathfinding Impact:
          - Multiple paths enable route choices for robots
          - Optimal path may be shorter than in perfect maze version
          - DFS exploration might explore more before finding goal
          - A* benefits from alternative routes
        
        Parameters:
          percentage (int): Percentage of total cells where we remove an extra wall
                           (0 = no loops/perfect maze, 100 = very dense connectivity)
        
        Integration:
          - Called after initial maze generation if loop_percent > 0
          - Now works with all generators including Kruskal's MST
          - Works with all maze representations (node-weighted and edge-weighted)
          - For MST mazes, loops transform the optimal tree into a graph with shortcuts
        """
        # Calculate how many walls to remove based on percentage
        num_walls_to_remove = int(self.width * self.height * percentage / 100)
        
        # Attempt to remove the specified number of walls
        for _ in range(num_walls_to_remove):
            # Pick a random cell
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            # Find all directions that currently have walls (not in grid set)
            possible_walls = []
            # Check North: wall exists if 'N' not in grid set
            if y > 0 and 'N' not in self.grid[(x, y)]:
                possible_walls.append(('N', 'S', x, y - 1))
            # Check South: wall exists if 'S' not in grid set
            if y < self.height - 1 and 'S' not in self.grid[(x, y)]:
                possible_walls.append(('S', 'N', x, y + 1))
            # Check West: wall exists if 'W' not in grid set
            if x > 0 and 'W' not in self.grid[(x, y)]:
                possible_walls.append(('W', 'E', x - 1, y))
            # Check East: wall exists if 'E' not in grid set
            if x < self.width - 1 and 'E' not in self.grid[(x, y)]:
                possible_walls.append(('E', 'W', x + 1, y))
            
            # If there are walls to remove, pick one randomly and carve passage
            if possible_walls:
                direction, opposite, nx, ny = random.choice(possible_walls)
                # Carve bidirectionally: add direction to both cells
                self.grid[(x, y)].add(direction)      # Remove wall from current cell
                self.grid[(nx, ny)].add(opposite)     # Remove wall from neighbor cell

    def _populate_node_weights(self, max_weight):
        """
        Assigns terrain costs (node weights) to every cell in the maze.
        
        Maze Data Structure Context:
          - self.node_weights[(x, y)] = cost to ENTER cell (x, y)
          - Used by node-weighted maze generators (DFS, BFS, Random Prim's, Greedy Frontier)
          - Not used by edge-weighted generators (Kruskal's MST)
        
        Algorithm:
          - Iterate through all cells in row-major order
          - Assign each cell a random integer cost in range [1, max_weight]
          - Uniform random distribution across the range
        
        Pathfinding Impact:
          - Cost 1: Cheap terrain (fast traversal), visualized in blue
          - Cost max_weight: Expensive terrain (slow traversal), visualized in red
          - Optimal paths favor lower-weight cells
          - Uniform weights (max_weight=1) makes all paths equal cost
        
        Parameters:
          max_weight (int): Maximum terrain cost (minimum is always 1)
                           Setting to 1 creates unweighted maze
        
        Integration:
          - Called after generation for non-MST methods
          - Affects Robot pathfinding cost calculations via get_cost()
          - Visual rendering: cells colored by weight (blue=cheap, red=expensive)
        """
        for y in range(self.height):
            for x in range(self.width):
                self.node_weights[(x, y)] = random.randint(1, max_weight)

    def _populate_rewards(self, num_rewards):
        """
        Places collectible reward items at random distinct locations in the maze.
        
        Maze Data Structure Context:
          - self.rewards is a set of (x, y) coordinates
          - Rewards are waypoints that robots must visit before reaching the goal
          - Independent of maze topology (can be placed after generation)
        
        Algorithm:
          - For each reward to place:
            a. Generate random coordinates (x, y)
            b. If coordinates already contain a reward, retry
            c. Otherwise add to self.rewards set
          - Guarantees no duplicate reward locations
        
        Robot Behavior Impact:
          - Robots must visit all rewards before goal is considered reached
          - Pathfinding becomes multi-stage: start → nearest_reward → ... → goal
          - Increases exploration and path complexity
          - Offline solvers compute optimal multi-waypoint tour
          - Online explorers dynamically retarget to nearest unvisited reward
        
        Parameters:
          num_rewards (int): Number of rewards to place (0 = no rewards)
        
        Integration:
          - Called at end of Maze.__init__ after all generation complete
          - Rewards tracked separately in Robot.unvisited_rewards
          - Visual rendering: orange circles for unvisited rewards
        """
        for _ in range(num_rewards):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in self.rewards: self.rewards.add((x, y)); break

    def get_valid_moves(self, x, y):
        """
        Returns the set of open passage directions from a given cell.
        
        Maze Data Structure Context:
          - self.grid[(x, y)] is a set containing 'N', 'S', 'E', 'W' for open passages
          - Empty set means all walls present (no passages)
          - If 'N' in set, passage exists from (x, y) to (x, y-1)
        
        Direction Mapping:
          - 'N' (North): y decreases by 1 (upward in grid coordinates)
          - 'S' (South): y increases by 1 (downward in grid coordinates)
          - 'W' (West): x decreases by 1 (leftward)
          - 'E' (East): x increases by 1 (rightward)
        
        Returns:
          set: Subset of {'N', 'S', 'E', 'W'} representing available moves
               Empty set if cell is completely walled (shouldn't happen in valid maze)
        
        Integration:
          - Used by pathfinding algorithms to determine legal moves
          - Used by visualization to determine which walls to draw
          - Core interface between maze representation and navigation logic
        """
        return self.grid.get((x,y), set())

    def get_neighbors(self, x, y):
        """
        Returns list of reachable neighbor cell coordinates from a given position.
        
        Maze Data Structure Context:
          - Uses self.grid[(x, y)] to find open passages (via get_valid_moves)
          - Converts directional strings ('N', 'S', 'E', 'W') to coordinate offsets
          - Only returns cells accessible through open passages (no walls blocking)
        
        Coordinate Transformations:
          - 'N': (x, y) → (x, y-1)  [move up]
          - 'S': (x, y) → (x, y+1)  [move down]
          - 'W': (x, y) → (x-1, y)  [move left]
          - 'E': (x, y) → (x+1, y)  [move right]
        
        Returns:
          list: Neighbor coordinates as [(x1, y1), (x2, y2), ...] 
                Length 0-4 depending on number of open passages
                Order not guaranteed
        
        Integration:
          - Primary interface for pathfinding algorithms (BFS, A*, DFS, etc.)
          - Used to expand frontier in search algorithms
          - Abstracts directional representation from coordinate-based logic
          - Essential for graph traversal operations
        """
        neighbors = []
        for move in self.get_valid_moves(x, y):
            nx, ny = x, y
            if move == 'N': ny -= 1; 
            elif move == 'S': ny += 1
            elif move == 'W': nx -= 1; 
            elif move == 'E': nx += 1
            neighbors.append((nx, ny))
        return neighbors

    def get_cost(self, pos_from, pos_to):
        """
        Returns the cost of moving from one cell to an adjacent cell.
        
        Maze Weight System Context:
          - self.weight_type determines which cost system is active:
            * 'node': cost to ENTER pos_to (terrain cost stored in node_weights)
            * 'edge': cost of EDGE from pos_from to pos_to (stored in edge_weights)
          - Node-weighted: used by DFS, BFS, Random Prim's, Greedy Frontier
          - Edge-weighted: used by Kruskal's MST generator
        
        Cost Semantics:
          Node-weighted: Crossing from A to B costs node_weights[B]
                        (pay cost when entering destination cell)
          Edge-weighted: Crossing from A to B costs edge_weights[(A, B)]
                        (pay cost for using the passage between cells)
        
        Parameters:
          pos_from (tuple): Starting cell coordinate (x, y)
          pos_to (tuple): Destination cell coordinate (x, y)
        
        Returns:
          int/float: Movement cost (≥1, default 1 if not specified)
        
        Integration:
          - Called by all pathfinding algorithms to compute path costs
          - Used in A*, Greedy cost accumulation
          - Critical for weighted shortest path computation
          - Determines optimal path selection when multiple routes exist
        """
        if self.weight_type == 'edge': return self.edge_weights.get((pos_from, pos_to), 1)
        else: return self.node_weights.get(pos_to, 1)


class Robot:
    """
    Represents the AI agent that can operate in two paradigms:

    1) Offline (Solver) — "knows the full maze" and computes an end-to-end plan
       before moving. Examples in this file:
         - A* (offline mode)
         - Prim Solver (MST-based, builds a global tree and follows it)
         - Kruskal Solver (MST-based)

       In this mode, the agent first computes a solution path (possibly in
       multiple stages when rewards are present). Once a final path exists,
       the visualization animates following that path step-by-step. Metrics
       are attributed to the search process (nodes explored, etc.), not the
       animation itself.

    2) Online (Explorer) — does NOT know the maze; it discovers the maze while
       moving. Examples:
         - BFS (unweighted shortest-path search discovered incrementally)
         - Greedy BFS (heuristic-only best-first)
         - Depth-First Search (systematic exploration with backtracking)
         - Monte Carlo lookahead (sample-based forward probing)
         - A* (online mode)

       In this mode, the agent maintains a frontier (queue/PQ/stack), a notion
       of the currently best next node to expand, and parent pointers for the
       discovered tree of predecessors. To make the animation physically faithful,
       the agent will "walk" along discovered edges to the next expansion node
       instead of teleporting. Importantly, the cost metric and the algorithmic
       step counts are separated from this animation walk to avoid inflating
       algorithmic metrics with purely presentational motion.

    Metrics (key distinctions):
      - steps: total physical animation moves (every visual move one cell).
      - nodes_expanded: number of nodes removed from the frontier and processed.
      - unique_explored: count of unique cells discovered/visited by the search.
      - algorithm_steps: a normalized notion of "logical steps" the algorithm
        performs. For BFS/A*/Greedy this increments on expansion (pop). For DFS
        and Monte Carlo it increments per deliberate move (including backtracking)
        because the algorithm must physically check new branches.
      - total_cost: cumulative algorithmic cost to produce the final solution,
        excluding animation-only motion. For online A*/Greedy we aggregate per
        segment (reward to reward to goal). For BFS we reconstruct cost along
        the parent chain on finish. For offline solvers it's the sum along the
        computed final path.
    """
    def __init__(self, maze, algorithm="Depth-First Search", knows_maze=False): # DFS default
        # Store maze reference (contains grid structure with directional passages)
        self.maze = maze
        self.algorithm = algorithm

        # Determine if robot has full maze knowledge (offline) or discovers it (online)
        # Online-only algorithms cannot use the 'knows_maze' = True mode
        if self.algorithm in ["Monte Carlo", "Depth-First Search", "Greedy BFS", "BFS"]:
            self.knows_maze = False  # Force online mode for these algorithms
        else:
            self.knows_maze = knows_maze  # A* and solvers can be online or offline
        
        # Alias: is_solver means "knows full maze and computes path in advance"
        self.is_solver = self.knows_maze

        # --- Basic Robot State ---
        # Pick random starting position that doesn't overlap with rewards
        self.start_pos = (random.randint(0, maze.width - 1), random.randint(0, maze.height - 1))
        while self.start_pos in self.maze.rewards:
            self.start_pos = (random.randint(0, maze.width-1), random.randint(0, maze.height-1))
        
        # Initialize current position at start
        self.x, self.y = self.start_pos
        
        # Pick distant goal position (different from start and rewards)
        self.goal_pos = self._get_distant_pos()
        while self.goal_pos in self.maze.rewards or self.goal_pos == self.start_pos:
            self.goal_pos = self._get_distant_pos()

        # Track robot's movement history (for visualization trail)
        self.path = [(self.x, self.y)]
        
        # Track which rewards haven't been collected yet
        self.unvisited_rewards = set(self.maze.rewards)
        
        # Track all cells discovered by search algorithm (for fog-of-war visualization)
        self.search_area = set()
        
        # Flag indicating if robot has reached goal with all rewards
        self.is_done = False

        # --- State for "Offline" (Solver) Mode ---
        # Pre-computed path from start to goal (via rewards)
        self.final_path = []

        # --- State for "Online" (Explorer) Mode ---
        # Stack for DFS/Monte Carlo backtracking
        self.backtrack_stack = []
        
        # Current waypoint target (next reward or final goal)
        self.current_target = None
        
        # Priority queue for A*/Greedy BFS: stores (priority, cost, node) tuples
        self.online_pq = []
        
        # FIFO queue for BFS: stores nodes in discovery order
        self.online_queue = collections.deque()
        
        # g-cost tracking for A*: maps node -> cost from segment start
        self.online_cost_so_far = {}
        
        # Parent pointers for path reconstruction: maps node -> parent node
        self.online_came_from = {}
        
        # Target cell for single-step backtracking animation
        self.backtrack_target = None
        
        # Queued animation steps for walking along discovered tree edges
        self.walk_path = collections.deque()
        
        # Accumulated solution cost across all segments (start → rewards → goal)
        self.solution_cost_accum = 0
        
        # Starting position of current search segment (for cost calculation)
        self.stage_start = None

        # --- Performance Metrics ---
        self.metrics = {
            'algorithm_steps': 0,      # Logical algorithm iterations (expansions/moves)
            'unique_explored': 0,      # Number of unique cells discovered
            'baseline_optimal_cost': 0, # Baseline optimal cost (computed via offline A*)
            'solution_cost': 0,        # Final path cost of algorithm's solution
            'animation_walk_cost': 0,  # Cumulative cost of all physical animation moves
            'total_path_length': 0,    # Number of edges physically traversed
            'nodes_expanded': 0,       # Nodes popped from frontier (= algorithm_steps)
            'frontier_max': 0,         # Maximum frontier size reached
        }

        # Compute baseline optimal cost for performance comparison
        # (uses offline A* regardless of robot's actual algorithm)
        self.metrics['baseline_optimal_cost'] = self._compute_min_path_cost_baseline(self.start_pos, self.goal_pos)

        # --- Initialize Algorithm State ---
        if self.is_solver:
            # Offline mode: compute full path before any movement
            self.solve_maze_offline()
        else:
            # Online mode: initialize search structures for incremental discovery
            self._setup_online_search()

    def _build_parent_chain(self, node):
        """Returns the chain of nodes following parent pointers up to the root.

        This is used by the physical animation system to construct a walk
        between the agent's current position and the next expansion node using
        the lowest common ancestor (LCA) of their parent chains. This keeps
        the animation faithful to discovered connectivity without changing the
        algorithm's semantics.
        """
        # Initialize empty chain to store path from node to root
        chain = []
        cur = node
        parents = self.online_came_from  # Parent pointers from search tree
        
        # Safety limit to prevent infinite loops (shouldn't happen in valid maze)
        limit = self.maze.width * self.maze.height + 2
        
        # Follow parent pointers until we reach root or hit a stopping condition
        while cur is not None and cur not in chain and limit > 0:
            chain.append(cur)  # Add current node to chain
            
            # Stop if we reached a root node (has no parent)
            if cur not in parents:
                break
            
            # Move to parent node
            cur = parents[cur]
            limit -= 1
        
        return chain  # Returns list: [node, parent, grandparent, ..., root]

    def _plan_walk_to(self, target):
        """Plan a physical path along parent pointers from current position to target.

        Strategy:
          1) Build parent chains for current position and the target.
          2) Find the LCA where these chains meet.
          3) Walk up from current to LCA, then down from LCA to target.

        The generated waypoints are stored in self.walk_path and consumed by
        step(), advancing one cell per tick.
        """
        # Early exit if already at target
        if target == (self.x, self.y):
            return
        
        # Build parent chains: node → parent → ... → root
        a_chain = self._build_parent_chain((self.x, self.y))  # Current position's ancestry
        b_chain = self._build_parent_chain(target)            # Target's ancestry
        
        # Safety check: both chains must exist
        if not a_chain or not b_chain:
            return
        
        # Find Lowest Common Ancestor (LCA) where chains intersect
        b_set = set(b_chain)  # Convert to set for O(1) lookup
        lca = None
        for n in a_chain:
            if n in b_set:  # Found first common ancestor
                lca = n
                break
        
        # If no LCA found, nodes aren't connected (shouldn't happen in valid tree)
        if lca is None:
            return
        
        # Calculate path segments
        i_lca_a = a_chain.index(lca)  # Position of LCA in current's chain
        i_lca_b = b_chain.index(lca)  # Position of LCA in target's chain
        
        # Path up: current → LCA (exclude current, include LCA)
        path_up = a_chain[1:i_lca_a+1]
        
        # Path down: LCA → target (exclude LCA, include target)
        path_down = list(reversed(b_chain[:i_lca_b]))
        
        # Combine: current → LCA → target
        walk_seq = path_up + path_down
        
        # Queue the walk sequence for step-by-step animation
        self.walk_path.clear()
        for node in walk_seq:
            self.walk_path.append(node)

    def _record_move(self, prev_pos, new_pos, is_algorithm_move=False):
        """Record a single visual move and optionally attribute algorithmic cost.

        - total_path_length/animation_walk_cost always include animation moves.
        - unique_explored reflects the size of the discovered set (search_area).
        """
        # Update path metrics only if robot actually moved
        if prev_pos != new_pos:
            # Path length = number of edges = number of nodes - 1
            self.metrics['total_path_length'] = max(0, len(self.path) - 1)
            
            # Accumulate cost of all physical moves (including animation walks)
            self.metrics['animation_walk_cost'] += self.maze.get_cost(prev_pos, new_pos)
        
        # Update exploration metric (always reflects current search_area size)
        self.metrics['unique_explored'] = len(self.search_area)

    def _bump_frontier_metric(self):
        """
        Updates the maximum frontier size metric based on current algorithm.
        
        Frontier definitions:
          - BFS: Queue of discovered but not expanded nodes
          - A*/Greedy: Priority queue of discovered nodes
          - DFS/Monte Carlo: Backtrack stack of branch points
        """
        # Measure current frontier size based on algorithm's data structure
        size = 0
        if self.algorithm == 'BFS':
            size = len(self.online_queue)
        elif self.algorithm in ['A*', 'Greedy BFS']:
            size = len(self.online_pq)
        elif self.algorithm in ['Depth-First Search', 'Monte Carlo']:
            size = len(self.backtrack_stack)
        
        # Update max if current frontier is larger than previous maximum
        if size > self.metrics['frontier_max']:
            self.metrics['frontier_max'] = size

    def _get_distant_pos(self):
        """
        Finds a suitable goal position far from the robot's starting position.
        
        Purpose:
          - Ensures interesting maze navigation (not trivially short paths)
          - Makes algorithm performance differences more observable
          - Creates challenging scenarios for pathfinding algorithms
        
        Algorithm:
          - Repeatedly generate random coordinates until Manhattan distance
            from start exceeds threshold
          - Threshold: (width + height) / 2 (roughly half the maximum distance)
          - Maximum possible Manhattan distance: width + height - 2
        
        Integration:
          - Called during Robot.__init__ to set self.goal_pos
          - Goal must not overlap with start or rewards (checked externally)
          - Used by visualization to draw goal marker
        """
        while True:
            gx, gy = random.randint(0, self.maze.width-1), random.randint(0, self.maze.height-1)
            dist = abs(self.x - gx) + abs(self.y - gy)
            if dist > (self.maze.width + self.maze.height) / 2: return (gx, gy)

    def _heuristic(self, a, b):
        """
        Computes Manhattan distance heuristic between two positions.
        
        Mathematical Definition:
          - h(a, b) = |a.x - b.x| + |a.y - b.y|
          - Also called L1 distance or taxicab distance
          - Admissible: never overestimates true shortest path in grid graphs
          - Consistent: h(a, b) ≤ cost(a, c) + h(c, b) for any intermediate c
        
        Properties:
          - Lower bound on actual path cost in 4-connected grid
          - Admissibility guarantees A* optimality (finds shortest path)
          - Consistency ensures A* never re-expands nodes
          - In unweighted maze: h(a, b) ≤ actual_path_length(a, b)
        
        Parameters:
          a (tuple): First position (x, y)
          b (tuple): Second position (x, y)
        
        Returns:
          int: Manhattan distance (sum of absolute coordinate differences)
        
        Integration:
          - Used by A* for f = g + h evaluation function
          - Used by Greedy BFS for pure heuristic-based ordering (f = h)
          - Used by online explorers to select next reward target
          - Critical for informed search algorithm efficiency
        """
        (x1, y1) = a; (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)

    def _compute_min_path_cost_baseline(self, start, goal):
        """
        Computes optimal path cost from start to goal using offline A*.
        
        Purpose:
          - Establishes baseline metric: theoretical minimum cost for comparison
          - Used to evaluate algorithm efficiency (how close to optimal?)
          - Runs independently of main robot algorithm (doesn't affect exploration)
        
        Algorithm:
          - Standard A* with f = g + h, where h is Manhattan distance
          - Priority queue ordered by f-cost
          - Explores full maze (knows all passages and weights)
          - Terminates when goal is reached
          - Returns cumulative path cost (g-value at goal)
        
        Assumptions:
          - Full maze knowledge (offline computation)
          - Uses maze.get_neighbors for connectivity
          - Uses maze.get_cost for edge/node weights
          - Single-goal pathfinding (no rewards considered here)
        
        Returns:
          int/float: Optimal path cost from start to goal, or 0 if unreachable
        
        Integration:
          - Called during Robot.__init__ to set metrics['baseline_optimal_cost']
          - Provides reference for algorithm performance evaluation
          - Independent of whether robot is online or offline mode
        """
        # Initialize A* search structures
        pq = []  # Priority queue: (f_cost, g_cost, node)
        g = {start: 0}  # g-cost: actual cost from start to each node
        parent = {start: None}  # Parent pointers (unused but standard A* structure)
        
        # Push start node with f = g + h (g=0 initially)
        heapq.heappush(pq, (self._heuristic(start, goal), 0, start))
        
        # Standard A* expansion loop
        while pq:
            # Pop node with lowest f-cost
            f, cost, node = heapq.heappop(pq)
            
            # Goal test: return g-cost when goal reached
            if node == goal:
                return cost  # This is the optimal path cost
            
            # Expand all reachable neighbors through maze passages
            for nb in self.maze.get_neighbors(node[0], node[1]):
                # Calculate tentative g-cost via current node
                new_cost = cost + self.maze.get_cost(node, nb)
                
                # Update if this is better path or neighbor is newly discovered
                if nb not in g or new_cost < g[nb]:
                    g[nb] = new_cost
                    parent[nb] = node
                    
                    # Push with f = g + h
                    f_cost = new_cost + self._heuristic(nb, goal)
                    heapq.heappush(pq, (f_cost, new_cost, nb))
        
        # Goal unreachable (shouldn't happen in connected maze)
        return 0

    def _build_mst(self, method):
        """
        Constructs a Minimum Spanning Tree over the entire maze graph.
        
        Purpose:
          - Used by offline MST solvers (Prim Solver, Kruskal Solver)
          - Builds global tree structure respecting maze connectivity and weights
          - Tree defines unique paths between any two cells (no cycles)
        
        MST Context:
          - Input: Maze graph where nodes are cells, edges are passages
          - Edge weights: from maze.get_cost (respects node/edge weight system)
          - Output: Tree adjacency list with all cells connected
        
        Algorithms Supported:
          1. Prim Solver:
             - Start from robot's start_pos
             - Maintain priority queue of edges from visited to unvisited nodes
             - Greedily add lightest edge that expands the tree
             - Time: O(E log V) with binary heap
          
          2. Kruskal Solver:
             - Sort all edges by weight
             - Use Union-Find to track connected components
             - Add edges that connect different components (avoid cycles)
             - Time: O(E log E) for sorting
        
        Mathematical Properties:
          - MST has exactly |V| - 1 edges for |V| vertices
          - Minimizes total weight while maintaining connectivity
          - Unique path exists between any two nodes in the tree
          - Both Prim and Kruskal produce optimal MSTs (may differ if ties)
        
        Parameters:
          method (str): 'Prim Solver' or 'Kruskal Solver'
        
        Returns:
          tuple: (tree_adj, visited_nodes) where:
            - tree_adj: dict mapping node → list of adjacent nodes in MST
            - visited_nodes: set of all nodes included in MST
        
        Integration:
          - Called by solve_maze_offline() for MST-based solvers
          - Result used to extract unique tree paths for robot navigation
          - Tree structure respects maze weights, producing cost-efficient routes
        """
        # Step 1: Collect all maze cells as graph vertices
        nodes = [(x, y) for y in range(self.maze.height) for x in range(self.maze.width)]
        
        # Step 2: Build undirected edge list from maze passages
        edges = []
        for (x, y) in nodes:
            # Get all neighbors reachable through open passages in maze.grid
            for (nx, ny) in self.maze.get_neighbors(x, y):
                # Only add each edge once (use lexicographic ordering to avoid duplicates)
                if (x, y) < (nx, ny):
                    # Get edge cost (respects node/edge weight system)
                    cost = self.maze.get_cost((x, y), (nx, ny))
                    edges.append((cost, (x, y), (nx, ny)))
        
        # Initialize MST adjacency list and visited set
        tree_adj = collections.defaultdict(list)
        visited_nodes = set()
        
        # Branch based on MST algorithm choice
        if method == 'Prim Solver':
            # --- Prim's Algorithm: Greedy frontier expansion ---
            # Start from robot's initial position
            start = self.start_pos
            visited = {start}
            visited_nodes.add(start)
            
            # Initialize priority queue with edges from start
            pq = []
            for cost, u, v in edges:
                # Add edge to PQ if one endpoint is start and other is unvisited
                if u == start and v not in visited:
                    heapq.heappush(pq, (cost, u, v))
                elif v == start and u not in visited:
                    heapq.heappush(pq, (cost, v, u))
            
            # Greedily add lightest edges that expand the tree
            while pq and len(visited) < len(nodes):
                # Pop lightest edge from frontier
                cost, u, v = heapq.heappop(pq)
                
                # Skip if destination already in tree (stale entry)
                if v in visited:
                    continue
                
                # Add edge to MST
                visited.add(v)
                visited_nodes.add(v)
                tree_adj[u].append(v)  # Bidirectional adjacency
                tree_adj[v].append(u)
                
                # Add all edges from newly added vertex v to unvisited neighbors
                for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                    nx, ny = v[0]+dx, v[1]+dy
                    # Check bounds, visited status, and passage existence
                    if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height and (nx, ny) not in visited and (nx, ny) in self.maze.get_neighbors(v[0], v[1]):
                        c = self.maze.get_cost(v, (nx, ny))
                        heapq.heappush(pq, (c, v, (nx, ny)))
        
        else:  # Kruskal Solver
            # --- Kruskal's Algorithm: Sort edges, add if no cycle ---
            # Initialize Union-Find data structure
            parent = {}
            rank = {}
            
            # Find with path compression
            def find(a):
                parent.setdefault(a, a)
                if parent[a] != a:
                    parent[a] = find(parent[a])  # Path compression
                return parent[a]
            
            # Union by rank
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return False  # Already in same set (would create cycle)
                rank.setdefault(ra, 0)
                rank.setdefault(rb, 0)
                # Attach smaller tree to larger tree
                if rank[ra] < rank[rb]:
                    parent[ra] = rb
                elif rank[rb] < rank[ra]:
                    parent[rb] = ra
                else:
                    parent[rb] = ra
                    rank[ra] += 1
                return True
            
            # Initialize each node as its own set
            for n in nodes:
                parent[n] = n
                rank[n] = 0
            
            # Sort edges by weight (greedy: always try lightest first)
            edges.sort(key=lambda e: e[0])
            
            # Add edges in sorted order until we have spanning tree
            added = 0
            need = len(nodes) - 1  # Spanning tree has n-1 edges
            
            for cost, u, v in edges:
                # Try to merge sets containing u and v
                if union(u, v):
                    # Edge accepted (doesn't create cycle)
                    tree_adj[u].append(v)  # Bidirectional adjacency
                    tree_adj[v].append(u)
                    visited_nodes.add(u)
                    visited_nodes.add(v)
                    added += 1
                    
                    # Early termination: spanning tree complete
                    if added >= need:
                        break
        
        return tree_adj, visited_nodes

    # --- OFFLINE (SOLVER) LOGIC ---
    def _run_offline_search(self, start, end):
        """Core pathfinding engine for Offline algorithms (A*, BFS).

        This routine abstracts the differences between A* and BFS in the offline
        context. It returns:
          - path: a list of nodes from start to end (inclusive) if reachable
          - visited_nodes: set of nodes processed during the search (for metrics)

        Mechanics:
          - BFS: treats all edges as unit cost. Priority is equal to path length.
          - A*: uses f = g + h, where h is Manhattan distance to 'end'. g accumulates
                maze cost via Maze.get_cost (node-weight or edge-weight based on generator).

        Parent pointers (came_from) are used for path reconstruction. This function
        does not animate; it performs pure computation suitable for offline solvers.
        """
        # Initialize priority queue based on algorithm type
        pq = []
        if self.algorithm == 'BFS':
            heapq.heappush(pq, (0, start))  # BFS: priority = path length (starts at 0)
        elif self.algorithm == 'A*':
            heapq.heappush(pq, (self._heuristic(start, end), start))  # A*: priority = f = g + h (g=0 initially)
        else:
            return [], {start}  # Unsupported algorithm

        # Initialize search data structures
        came_from = {start: None}    # Parent pointers for path reconstruction
        cost_so_far = {start: 0}     # g-cost: actual cost from start to each node
        visited_nodes = set()        # Track all nodes processed (for metrics)
        
        # Main search loop
        while pq:
            # Pop node with lowest priority (f-cost for A*, path length for BFS)
            if self.algorithm == 'A*':
                _, current = heapq.heappop(pq)  # A* only needs node (f-cost already factored in)
            else:
                current_cost, current = heapq.heappop(pq)  # BFS tracks cost explicitly

            # Mark node as processed
            visited_nodes.add(current)
            
            # Goal test: terminate if we reached the end
            if current == end:
                break

            # Expand all reachable neighbors
            for neighbor in self.maze.get_neighbors(current[0], current[1]):
                # Calculate g-cost to reach neighbor through current
                if self.algorithm == 'BFS':
                    new_cost = cost_so_far[current] + 1  # BFS: uniform cost (counts edges)
                else:
                    new_cost = cost_so_far[current] + self.maze.get_cost(current, neighbor)  # A*: actual maze cost

                # Update neighbor if we found a better path or it's undiscovered
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    
                    # Calculate priority based on algorithm
                    if self.algorithm == 'BFS':
                        priority = new_cost  # BFS: priority = g (no heuristic)
                    else:
                        priority = new_cost + self._heuristic(neighbor, end)  # A*: priority = g + h
                    
                    heapq.heappush(pq, (priority, neighbor))
        
        # Reconstruct path from end to start using parent pointers
        path = []
        current = end
        if end in came_from:  # Only reconstruct if goal was reached
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()  # Reverse to get start → end order
        
        return (path, visited_nodes)

    def solve_maze_offline(self):
        """Computes the full multi-stage path for an Offline solver.

        When rewards exist, the solver computes a sequence of segments:
          current -> nearest next reward -> ... -> goal.

        Two flavors supported:
          - A* (or BFS) solver: repeatedly runs _run_offline_search for each
            segment using the full maze (knows_maze=True).
          - MST solvers (Prim/Kruskal): construct a global spanning tree of the
            maze graph first, then navigate along unique tree paths between
            waypoints (rewards and goal). This is not necessarily globally shortest
            in the original weighted graph but is consistent with the MST's structure.

        After planning, the visualization animates following self.final_path.
        Metrics reflect the planning workload and the consolidated final cost,
        not the animation walk cost.
        """
        if self.algorithm in ['Prim Solver', 'Kruskal Solver']:
            # --- MST-Based Solver Path ---
            # Step 1: Build Minimum Spanning Tree over entire maze graph
            tree_adj, visited_nodes = self._build_mst(self.algorithm)
            self.search_area.update(visited_nodes)  # All nodes processed during MST construction

            # Helper: Find unique path between two nodes in the tree
            def tree_path(a, b):
                """
                BFS traversal in tree to find unique path from a to b.
                Since MST is a tree, there's exactly one path between any two nodes.
                """
                # Initialize BFS from node a
                parent = {a: None}
                dq = collections.deque([a])
                
                # Explore tree until we find b
                while dq:
                    u = dq.popleft()
                    if u == b:
                        break  # Found target node
                    # Expand to tree neighbors not yet visited
                    for v in tree_adj.get(u, []):
                        if v not in parent:
                            parent[v] = u
                            dq.append(v)
                
                # Reconstruct path if b was reached
                if b not in parent:
                    return []  # No path exists (shouldn't happen in connected tree)
                
                # Build path from b back to a using parent pointers
                path = [b]
                cur = b
                while cur != a:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()  # Reverse to get a → b order
                return path

            # Step 2: Plan multi-waypoint tour: start → rewards → goal
            current_loc = self.start_pos
            rewards_to_visit = set(self.maze.rewards)
            full_path = []
            
            # Greedily visit nearest reward until all collected
            while rewards_to_visit:
                # Find reward with lowest-cost tree path from current location
                best_path, best_target, min_cost = [], None, float('inf')
                
                for reward in rewards_to_visit:
                    # Get unique tree path from current to this reward
                    p = tree_path(current_loc, reward)
                    if p:
                        # Calculate actual path cost (sum of edge costs)
                        path_cost = sum(self.maze.get_cost(p[i], p[i+1]) for i in range(len(p)-1))
                        # Track best option
                        if path_cost < min_cost:
                            min_cost, best_path, best_target = path_cost, p, reward
                
                # Add best path segment to solution (skip first node to avoid duplicates)
                if best_target:
                    full_path.extend(best_path[1:])
                    current_loc = best_target
                    rewards_to_visit.remove(best_target)
                else:
                    break  # No path to remaining rewards (shouldn't happen)
            
            # Step 3: Add final segment from last reward to goal
            p_goal = tree_path(current_loc, self.goal_pos)
            if p_goal:
                full_path.extend(p_goal[1:])  # Skip first to avoid duplicate
            
            self.final_path = full_path
            
            # Step 4: Compute metrics
            # Algorithm steps = all nodes processed during MST construction
            self.metrics['algorithm_steps'] = len(self.search_area)
            
            # Calculate total cost of final path
            cost_sum = 0
            cur = self.start_pos
            for nxt in self.final_path:
                cost_sum += self.maze.get_cost(cur, nxt)
                cur = nxt
            # Store the solution cost found by the offline solver
            self.metrics['solution_cost'] = cost_sum
        else:
            # --- A* / BFS Solver Path ---
            # Use standard pathfinding (A* or BFS) to compute segments
            
            # Step 1: Initialize multi-waypoint tour planning
            current_loc = self.start_pos
            rewards_to_visit = set(self.maze.rewards)
            full_path = []
            
            # Step 2: Greedily visit rewards (nearest by actual search cost)
            while rewards_to_visit:
                # Evaluate all remaining rewards to find closest by search cost
                best_path, best_target, min_cost = [], None, float('inf')
                
                for reward in rewards_to_visit:
                    # Run offline search (A* or BFS) from current to this reward
                    path, visited = self._run_offline_search(current_loc, reward)
                    self.search_area.update(visited)  # Accumulate explored nodes for metrics
                    
                    if path:
                        # Calculate path cost based on algorithm
                        if self.algorithm == 'BFS':
                            # BFS: cost = number of edges (unweighted)
                            path_cost = len(path)
                        else:
                            # A*: cost = sum of actual maze edge/node costs
                            path_cost = sum(self.maze.get_cost(path[i], path[i+1]) for i in range(len(path)-1))
                        
                        # Track best reward option
                        if path_cost < min_cost:
                            min_cost, best_path, best_target = path_cost, path, reward
                
                # Add best path segment to solution
                if best_target:
                    full_path.extend(best_path[1:])  # Skip first node (already at it)
                    current_loc = best_target
                    rewards_to_visit.remove(best_target)
                else:
                    break  # No path to remaining rewards (unreachable)
            
            # Step 3: Add final segment from last location to goal
            path_to_goal, visited = self._run_offline_search(current_loc, self.goal_pos)
            self.search_area.update(visited)
            
            if path_to_goal:
                full_path.extend(path_to_goal[1:])  # Skip first to avoid duplicate
            
            # Store complete path for animation
            self.final_path = full_path
            
            # Step 4: Compute metrics
            # Algorithm steps = total unique nodes explored across all searches
            self.metrics['algorithm_steps'] = len(self.search_area)
            
            # Calculate total cost of concatenated path
            cost_sum = 0
            cur = self.start_pos
            for nxt in self.final_path:
                cost_sum += self.maze.get_cost(cur, nxt)
                cur = nxt
            # Store the solution cost found by the offline solver
            self.metrics['solution_cost'] = cost_sum

    def _step_follow_path(self):
        """
        Step function for OFFLINE solver (animates pre-computed path).
        
        Offline solvers compute the full solution path during initialization.
        This function simply animates the robot following that path step-by-step.
        Each call moves the robot one cell along self.final_path.
        """
        # If there's still path to follow, take one step
        if self.final_path:
            prev = (self.x, self.y)
            
            # Pop next position from pre-computed path
            next_pos = self.final_path.pop(0)
            self.x, self.y = next_pos
            
            # Record position in movement history
            self.path.append((self.x, self.y))
            
            # Update metrics (doesn't count as algorithm step - just animation)
            self._record_move(prev, next_pos, is_algorithm_move=False)
            
            # Collect reward if robot stepped on one
            if (self.x, self.y) in self.unvisited_rewards:
                self.unvisited_rewards.remove((self.x, self.y))
        
        # Check for completion: at goal with all rewards collected
        if (self.x, self.y) == self.goal_pos and not self.unvisited_rewards:
            self.is_done = True
    # --- END OFFLINE LOGIC ---

    # --- ONLINE (EXPLORER) LOGIC ---
    def _update_online_target(self, from_pos):
        """Find the next target (closest remaining reward or the goal).

        Online explorers pursue a moving objective: the nearest unvisited reward
        (by Manhattan distance heuristic) until rewards are exhausted, then the
        final goal. This target is re-evaluated when a reward is reached.
        """
        # If no rewards left, target the final goal
        if not self.unvisited_rewards:
            self.current_target = self.goal_pos
        else:
            # Find nearest unvisited reward using Manhattan distance
            min_dist = float('inf')
            best_reward = None
            
            for reward in self.unvisited_rewards:
                # Calculate heuristic distance from current position
                dist = self._heuristic(from_pos, reward)
                
                # Track closest reward
                if dist < min_dist:
                    min_dist = dist
                    best_reward = reward
            
            # Set closest reward as current target
            self.current_target = best_reward

    def _setup_online_search(self):
        """Initialize the data structures for the chosen Online algorithm.

        Highlights:
          - search_area: visual/logical visited set.
          - online_queue / online_pq: BFS queue or priority queues for A*/Greedy.
          - online_came_from / online_cost_so_far: parent pointers and g-costs.
          - stage_start: marks the beginning of a reward-to-reward (or start-to-goal)
            segment so we can attribute per-segment costs for heuristic-driven methods.

        We also avoid teleporting by queuing a physical walk to the next expansion
        node (handled in step()).
        """
        # Determine initial target (nearest reward or goal if no rewards)
        self._update_online_target((self.x, self.y))
        
        # Record starting position for this search segment
        start_pos = (self.x, self.y)
        self.stage_start = start_pos
        
        # Initialize cost tracking (for A* g-cost accumulation)
        self.online_cost_so_far = {start_pos: 0}
        
        # Initialize parent pointers (for path reconstruction)
        self.online_came_from = {start_pos: None}  # Start has no parent
        
        # Initialize algorithm-specific data structures
        self.online_pq = []                        # Priority queue for A*/Greedy BFS
        self.online_queue = collections.deque()    # FIFO queue for BFS
        
        # Mark start as explored for most algorithms
        # Exception: Greedy BFS uses search_area as closed set, marks on expansion
        if self.algorithm != 'Greedy BFS':
            self.search_area.add(start_pos)

        # Prime the appropriate frontier data structure with starting position
        if self.algorithm == 'BFS':
            # BFS: Add start to FIFO queue
            self.online_queue.append(start_pos)
        
        elif self.algorithm == 'A*':
            # A*: Priority = f = g + h, where g=0 at start
            priority = self._heuristic(start_pos, self.current_target)
            heapq.heappush(self.online_pq, (priority, 0, start_pos))  # (f, g, node)
        
        elif self.algorithm == 'Greedy BFS':
            # Greedy: Priority = h only (no g-cost consideration)
            priority = self._heuristic(start_pos, self.current_target)
            heapq.heappush(self.online_pq, (priority, start_pos))  # (h, node)
        
        # Note: DFS and Monte Carlo don't use queue/PQ, they use backtrack_stack
        # which starts empty and grows as they explore

    def _step_online_bfs(self):
        """A single step of an Online BFS explorer.

        BFS Semantics:
          - Frontier: FIFO queue of discovered but not yet expanded nodes.
          - Expansion: Pop the head, expand all undiscovered neighbors, set parents.
          - Optimality: Finds shortest path in unweighted graphs.

        Visualization policy:
          - If the agent is not physically at the node to be expanded, plan a
            physical walk to it (consumed by step()) and return.
          - When at the node, perform expansion (nodes_expanded++, algorithm_steps++).
          - On finishing at the goal, reconstruct cost along the parent tree and
            assign to total_cost (animation moves excluded).
        """
        # Early exit if queue is empty
        if not self.online_queue:
            return
        
        # Peek at next node to expand (front of FIFO queue)
        current = self.online_queue[0]
        
        # Physical animation: must be at node before expanding it
        if (self.x, self.y) != current:
            self._plan_walk_to(current)  # Queue walk steps along discovered tree
            return
        
        # Dequeue node for expansion (FIFO: oldest discovered = shallowest level)
        current = self.online_queue.popleft()
        
        # Move robot to current node
        prev = (self.x, self.y)
        self.x, self.y = current
        self.path.append(current)
        
        # Update metrics for expansion
        self.metrics['nodes_expanded'] += 1
        self.metrics['algorithm_steps'] += 1
        self._record_move(prev, current, is_algorithm_move=False)
        
        # Check if reached final goal with all rewards collected
        if current == self.goal_pos and not self.unvisited_rewards:
            # Reconstruct path cost from start to goal using parent pointers
            cost_sum = 0
            node = current
            while node is not None and node in self.online_came_from:
                parent = self.online_came_from[node]
                if parent is None:
                    break
                # Add edge cost from parent to node
                cost_sum += self.maze.get_cost(parent, node)
                node = parent
            
            # Store the solution cost found by the algorithm
            self.metrics['solution_cost'] = cost_sum
            self.is_done = True
            return
        
        # Expand neighbors: discover and enqueue all reachable unvisited cells
        for neighbor in self.maze.get_neighbors(current[0], current[1]):
            # Only process if not already discovered
            if neighbor not in self.online_came_from:
                # Set parent pointer for path reconstruction
                self.online_came_from[neighbor] = current
                
                # Mark as discovered (for visualization)
                self.search_area.add(neighbor)
                
                # Enqueue for future expansion (FIFO order)
                self.online_queue.append(neighbor)
        
        # Update frontier size metric
        self._bump_frontier_metric()

    def _step_online_a_star(self):
        """A single step of an Online A* explorer.

        A* Semantics (online):
          - Priority queue ordered by f = g + h, with Manhattan heuristic.
          - We clean stale PQ entries (those with worse g than currently known).
          - We maintain parent pointers and g-costs for discovered nodes.

        Visualization policy:
          - Physically walk to the PQ top node before expanding it.
          - When a reward/goal is reached, attribute the incremental g-cost of
            that segment to solution_cost_accum. The final total_cost is the
            sum across segments.

        Note: We do not add animation walking cost to total_cost; it only reflects
        the cost of the algorithmically chosen solution path.
        """
        # Early exit if priority queue is empty (shouldn't happen in valid maze)
        if not self.online_pq:
            return
        
        # Clean stale PQ entries: remove entries with outdated g-costs
        # Multiple entries for same node can exist; only process one with best g-cost
        while self.online_pq and self.online_pq[0][2] in self.online_cost_so_far and self.online_pq[0][1] > self.online_cost_so_far[self.online_pq[0][2]]:
            heapq.heappop(self.online_pq)  # Discard stale entry
        if not self.online_pq:
            return
        
        # Peek at next node to expand (lowest f-cost in PQ)
        peek = self.online_pq[0]
        current = peek[2]  # Extract node coordinates from (f, g, node) tuple
        
        # Physical animation: walk to node before expanding it
        if (self.x, self.y) != current:
            self._plan_walk_to(current)  # Queue intermediate steps along discovered tree
            return
        
        # Pop the node for expansion
        priority, cost, current = heapq.heappop(self.online_pq)  # cost = g-value
        
        # Double-check this isn't a stale entry (another path found better cost)
        if current in self.online_cost_so_far and cost > self.online_cost_so_far[current]:
            return  # Skip stale entry
        
        # Move robot to current node
        prev = (self.x, self.y)
        self.x, self.y = current
        self.path.append(current)
        
        # Update metrics
        self.metrics['nodes_expanded'] += 1
        self.metrics['algorithm_steps'] += 1
        self._record_move(prev, current, is_algorithm_move=False)

        # Check if we reached a target (reward or goal)
        if current == self.current_target:
            # Collect reward if present
            if current in self.unvisited_rewards:
                self.unvisited_rewards.remove(current)
            
            # Check for final goal with all rewards collected
            if not self.unvisited_rewards and current == self.goal_pos:
                # Reconstruct cost of final segment using parent pointers
                cost_sum = 0
                node = current
                while node is not None and node in self.online_came_from:
                    parent = self.online_came_from[node]
                    if parent is None:
                        break
                    cost_sum += self.maze.get_cost(parent, node)
                    node = parent
                # Add final segment cost to total
                self.solution_cost_accum += cost_sum
                # Store the solution cost found by the algorithm
                self.metrics['solution_cost'] = self.solution_cost_accum
                self.is_done = True
                return
            
            # Reached intermediate reward: retarget to next objective
            self._update_online_target(current)
            
            # Calculate cost of completed segment (stage_start → current)
            cost_sum = 0
            node = current
            while node is not None and node in self.online_came_from:
                parent = self.online_came_from[node]
                if parent is None:
                    break
                cost_sum += self.maze.get_cost(parent, node)
                node = parent
            self.solution_cost_accum += cost_sum
            
            # Reset search structures for next segment (fresh A* from current position)
            self.online_cost_so_far = {current: 0}
            self.online_came_from = {current: None}
            self.online_pq = []
            new_priority = self._heuristic(current, self.current_target)  # f = h (g=0 at start)
            heapq.heappush(self.online_pq, (new_priority, 0, current))
            self.stage_start = current
            self._bump_frontier_metric()
            return

        # Expand neighbors: add/update them in the search frontier
        for neighbor in self.maze.get_neighbors(current[0], current[1]):
            # Calculate tentative g-cost via current node
            new_cost = cost + self.maze.get_cost(current, neighbor)
            
            # Update if this is a better path or neighbor is newly discovered
            if neighbor not in self.online_cost_so_far or new_cost < self.online_cost_so_far[neighbor]:
                self.online_cost_so_far[neighbor] = new_cost
                self.search_area.add(neighbor)  # Mark as discovered
                self.online_came_from[neighbor] = current  # Set parent for path reconstruction
                
                # Calculate f-cost: f = g + h
                new_priority = new_cost + self._heuristic(neighbor, self.current_target)
                heapq.heappush(self.online_pq, (new_priority, new_cost, neighbor))
        
        self._bump_frontier_metric()

    def _step_online_greedy_bfs(self):
        """A single step of an Online Greedy Best-First Search explorer.

        Greedy BFS Semantics:
          - Priority queue ordered by heuristic h(n) only (no g-cost). Tends to
            rush toward the goal; not optimal in general.
          - We maintain a closed set via search_area to avoid re-expanding nodes.

        Visualization policy mirrors A*: walk to the selected node first; on
        reaching a reward/goal, attribute the cost of the discovered tree segment
        (via parents) to total_cost and reset for the next segment.
        """
        if not self.online_pq: return
        # Peek; physically walk to top before expanding
        peek = self.online_pq[0]
        current = peek[1]
        if (self.x, self.y) != current:
            self._plan_walk_to(current); return
        priority, current = heapq.heappop(self.online_pq) # Pop based only on h-cost

        # Need to check if visited because greedy can revisit with lower h-cost
        # but we only want to expand each node once visually in online mode.
        # `search_area` acts as our closed set here.
        if current in self.search_area:
             return # Skip if already processed

        self.search_area.add(current) # Add to closed set equivalent
        prev = (self.x, self.y)
        self.x, self.y = current; self.path.append(current)
        self.metrics['nodes_expanded'] += 1
        self.metrics['algorithm_steps'] += 1
        self._record_move(prev, current, is_algorithm_move=False)

        # Target check (same logic as A*)
        if current == self.current_target:
            if current in self.unvisited_rewards: self.unvisited_rewards.remove(current)
            if not self.unvisited_rewards and current == self.goal_pos:
                # Final segment: accumulate cost from stage_start to current using parents
                cost_sum = 0
                node = current
                while node is not None and node in self.online_came_from:
                    parent = self.online_came_from[node]
                    if parent is None: break
                    cost_sum += self.maze.get_cost(parent, node)
                    node = parent
                self.solution_cost_accum += cost_sum
                # Store the solution cost found by the algorithm
                self.metrics['solution_cost'] = self.solution_cost_accum
                self.is_done = True; return
            self._update_online_target(current)
            # Reset search, only need heuristic for priority
            self.online_came_from = {current: None}; self.online_pq = []
            new_priority = self._heuristic(current, self.current_target) # Priority = h
            heapq.heappush(self.online_pq, (new_priority, current))
            # Accumulate cost for finished segment from stage_start to current
            cost_sum = 0
            node = current
            while node is not None and node in self.online_came_from:
                parent = self.online_came_from[node]
                if parent is None: break
                cost_sum += self.maze.get_cost(parent, node)
                node = parent
            self.solution_cost_accum += cost_sum
            self.stage_start = current
            self._bump_frontier_metric(); return

        # Explore neighbors
        for neighbor in self.maze.get_neighbors(current[0], current[1]):
             # Add to PQ if not already processed (in search_area)
            if neighbor not in self.search_area and neighbor not in self.online_came_from: # Prevent adding duplicates to PQ if not processed
                self.online_came_from[neighbor] = current # Store parent (optional for pure greedy)
                new_priority = self._heuristic(neighbor, self.current_target) # Priority = h
                heapq.heappush(self.online_pq, (new_priority, neighbor)) # Add with h-cost priority
        self._bump_frontier_metric()

    def _step_explore_dfs(self, randomize=True):
        """A single step of Online DFS (or Random Walk when randomize=True).

        DFS Semantics:
          - Systematically explores as deep as possible along a branch before
            backtracking to the last branching point. The algorithm must
            physically "walk back" to try a different direction (unlike PQ-based
            planners which can jump logically to far nodes).

        Implementation details:
          - We treat backtrack_stack as the trail of decisions to unwind.
          - algorithm_steps increments on each deliberate forward move (and in
            step() on backtrack moves) because DFS must physically check the
            next path.
        """
        # Mark current position as explored
        self.search_area.add((self.x, self.y))
        
        # Update frontier size metric before any state changes
        self._bump_frontier_metric()
        
        # Check if we reached a target (reward or goal)
        if (self.x, self.y) == self.current_target:
            # Collect reward if present
            if (self.x, self.y) in self.unvisited_rewards:
                self.unvisited_rewards.remove((self.x, self.y))
            
            # Check for completion: at goal with all rewards collected
            if not self.unvisited_rewards and (self.x, self.y) == self.goal_pos:
                self.is_done = True
                return
            
            # Retarget to next reward/goal and clear backtrack stack (fresh exploration)
            self._update_online_target((self.x, self.y))
            self.backtrack_stack.clear()

        # Find all unvisited neighbors from current position
        unvisited_neighbors = []
        directions = ['N', 'S', 'W', 'E']
        
        # Randomize direction order for random walk variant
        if randomize:
            random.shuffle(directions)
        
        # Check each direction for valid, unvisited neighbors
        for move in directions:
            # Check if passage exists in this direction (in maze.grid set)
            if move in self.maze.get_valid_moves(self.x, self.y):
                # Calculate neighbor coordinates
                nx, ny = self.x, self.y
                if move == 'N':
                    ny -= 1
                elif move == 'S':
                    ny += 1
                elif move == 'W':
                    nx -= 1
                elif move == 'E':
                    nx += 1
                
                # Add to list if not yet explored
                if (nx, ny) not in self.search_area:
                    unvisited_neighbors.append((nx, ny))

        # If there are unvisited neighbors, move to the first one (DFS: go deeper)
        if unvisited_neighbors:
            nx, ny = unvisited_neighbors[0]  # Take first (or random if shuffled)
            prev = (self.x, self.y)
            
            # Push current position onto backtrack stack before moving
            self.backtrack_stack.append((self.x, self.y))
            
            # Move to neighbor
            self.x, self.y = nx, ny
            self.path.append((self.x, self.y))
            
            # Update metrics
            self.metrics['nodes_expanded'] += 1
            self.metrics['algorithm_steps'] += 1
            self._record_move(prev, (self.x, self.y), is_algorithm_move=True)
            
            # Update frontier metric after expanding
            self._bump_frontier_metric()
        
        # No unvisited neighbors: backtrack to last branching point
        elif self.backtrack_stack:
            # Pop from stack to initiate single-step backtrack
            self.backtrack_target = self.backtrack_stack.pop()
            
            # Update frontier metric after popping
            self._bump_frontier_metric()

    def _step_monte_carlo(self):
        """A single step of Online Monte Carlo Lookahead.

        Monte Carlo Lookahead (online, sample-based):
          - For each unvisited neighbor of the agent, run several random rollouts
            of limited depth. Score a neighbor by the best (lowest) heuristic
            distance to the current target achieved during its rollouts.
          - Choose the neighbor with the best score and move there.

        This method prefers moves that are more likely to quickly reduce the
        heuristic distance in the near future without computing exact shortest
        paths. We count algorithm_steps and, when moving, attribute cost only for
        those exploratory moves—not for any intermediate animation.
        """
        # Mark current position as explored
        self.search_area.add((self.x, self.y))
        
        # Update frontier metric before potential state changes
        self._bump_frontier_metric()
        
        # Check if reached target (reward or goal)
        if (self.x, self.y) == self.current_target:
            # Collect reward if present
            if (self.x, self.y) in self.unvisited_rewards:
                self.unvisited_rewards.remove((self.x, self.y))
            
            # Check for completion
            if not self.unvisited_rewards and (self.x, self.y) == self.goal_pos:
                self.is_done = True
                return
            
            # Retarget and clear backtrack stack for fresh exploration
            self._update_online_target((self.x, self.y))
            self.backtrack_stack.clear()

        # Find all unvisited neighbors
        neighbor_scores = []
        valid_unvisited_neighbors = []
        for neighbor in self.maze.get_neighbors(self.x, self.y):
            if neighbor not in self.search_area:
                valid_unvisited_neighbors.append(neighbor)

        # If there are unvisited neighbors, evaluate each with Monte Carlo rollouts
        if valid_unvisited_neighbors:
            # For each neighbor, run multiple random simulations
            for neighbor in valid_unvisited_neighbors:
                best_rollout_dist = float('inf')
                
                # Run MONTE_CARLO_ROLLOUTS simulations from this neighbor
                for _ in range(MONTE_CARLO_ROLLOUTS):
                    # Start simulation from neighbor position
                    sim_x, sim_y = neighbor
                    
                    # Track minimum heuristic distance achieved in this rollout
                    min_dist_rollout = self._heuristic((sim_x, sim_y), self.current_target)
                    
                    # Simulate MONTE_CARLO_DEPTH random moves
                    for _ in range(MONTE_CARLO_DEPTH):
                        # Get valid moves from current simulation position
                        sim_neighbors = self.maze.get_neighbors(sim_x, sim_y)
                        if not sim_neighbors:
                            break  # Dead end in simulation
                        
                        # Take random move (simulating random exploration)
                        sim_x, sim_y = random.choice(sim_neighbors)
                        
                        # Calculate heuristic at new simulation position
                        dist = self._heuristic((sim_x, sim_y), self.current_target)
                        
                        # Track best (lowest) distance achieved in this rollout
                        min_dist_rollout = min(dist, min_dist_rollout)
                    
                    # Update best distance across all rollouts for this neighbor
                    best_rollout_dist = min(best_rollout_dist, min_dist_rollout)
                
                # Store neighbor with its best rollout score (lower = better)
                heapq.heappush(neighbor_scores, (best_rollout_dist, neighbor))
            
            # Choose neighbor with best (lowest) Monte Carlo score
            if neighbor_scores:
                _, (nx, ny) = heapq.heappop(neighbor_scores)
                prev = (self.x, self.y)
                
                # Push current onto backtrack stack before moving
                self.backtrack_stack.append((self.x, self.y))
                
                # Move to best neighbor
                self.x, self.y = nx, ny
                self.path.append((self.x, self.y))
                
                # Update metrics
                self.metrics['nodes_expanded'] += 1
                self.metrics['algorithm_steps'] += 1
                self._record_move(prev, (self.x, self.y), is_algorithm_move=True)
                self._bump_frontier_metric()
                return  # End step after moving

        # No unvisited neighbors: initiate backtracking
        if self.backtrack_stack:
            self.backtrack_target = self.backtrack_stack.pop()
        
        self._bump_frontier_metric()

    # --- MAIN STEP FUNCTION ---
    def step(self):
        """Main robot tick.

        Execution order per tick:
          1) If a physical walk is planned (walk_path), consume one step. This
             realizes the "no teleport" policy for all algorithms.
          2) If backtracking is in progress (backtrack_target), perform one step
             and attribute algorithm_steps for DFS/Monte Carlo (which must walk
             physically to explore).
          3) Otherwise execute one logical algorithm step appropriate to the
             current algorithm (BFS/A*/Greedy/DFS/Monte Carlo or offline follow).

        This split ensures metrics distinguish presentational animation from the
        algorithm's actual decision steps and cost.
        """
        # Don't step if robot has finished (reached goal with all rewards)
        if self.is_done:
            return

        # Priority 1: Physical animation walk (for online algorithms)
        # Consume one cell from queued walk path between expansion nodes
        if self.walk_path:
            prev = (self.x, self.y)
            nxt = self.walk_path.popleft()  # Take next step in animation
            
            # Move robot one cell
            self.x, self.y = nxt
            self.path.append((self.x, self.y))
            
            # Record move (doesn't count as algorithm step, just animation)
            self._record_move(prev, nxt, is_algorithm_move=False)
            return

        # Priority 2: Backtracking step (for DFS/Monte Carlo)
        # Execute single-step backtrack when algorithm is stuck
        if self.backtrack_target:
            prev = (self.x, self.y)
            
            # Jump to backtrack target (one step back in exploration tree)
            self.x, self.y = self.backtrack_target
            self.path.append((self.x, self.y))
            
            # Record move and clear backtrack target
            self._record_move(prev, (self.x, self.y))
            self.backtrack_target = None
            return  # Backtrack takes full tick

        # Priority 3: Algorithm logic step
        if self.is_solver:
            # Offline mode: simply follow pre-computed path
            self._step_follow_path()
        else:
            # Online mode: execute one algorithmic step based on chosen algorithm
            if self.algorithm == "Random Walk":
                # DFS with randomized direction selection
                self._step_explore_dfs(randomize=True)
            
            elif self.algorithm == "Depth-First Search":
                # Systematic DFS (consistent direction order)
                self._step_explore_dfs(randomize=False)
            
            elif self.algorithm == "BFS":
                # Breadth-first expansion (FIFO queue)
                self._step_online_bfs()
            
            elif self.algorithm == "A*":
                # Best-first with f = g + h heuristic
                self._step_online_a_star()
            
            elif self.algorithm == "Greedy BFS":
                # Best-first with f = h only (no g-cost)
                self._step_online_greedy_bfs()
            
            elif self.algorithm == "Monte Carlo":
                # Sample-based lookahead with random rollouts
                self._step_monte_carlo()


class MazeApp:
    """
    Main application class for the Tkinter-based maze visualization and simulation.
    
    Architecture:
      - Manages GUI with Tkinter (control panels, canvases, event handlers)
      - Orchestrates maze generation and robot pathfinding simulation
      - Provides real-time visualization of algorithm execution
      - Tracks and displays performance metrics
    
    Components:
      1. Control Panel: User inputs for algorithms, generators, parameters
      2. Dual Canvas Display:
         - Left: Full maze view (robot vs maze) or robot's perspective (vs player)
         - Right: Robot's discovered view (robot vs maze) or player's view (vs player)
      3. Simulation Loop: Animates robot movement step-by-step
      4. Metrics Display: Real-time algorithm performance statistics
    
    Modes:
      - Robot vs Maze: Single robot pathfinding with algorithm visualization
      - Robot vs Player: Competitive mode where player races against robot AI
    
    Visualization Details:
      - Maze representation: self.maze.grid with directional passages (N/S/E/W)
      - Cell coloring: terrain weights (blue=cheap, red=expensive)
      - Path highlighting: search area, final path, known path
      - Markers: start (teal), goal (red), rewards (orange), agents (yellow/blue)
    
    Integration:
      - Creates Maze objects with selected generation algorithm
      - Creates Robot objects with selected pathfinding algorithm
      - Animates by calling robot.step() in update_loop
      - Renders using _draw_maze_on_canvas with maze.grid data structure
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Algorithm Visualizer")
        self.root.configure(bg=BG_COLOR)
        # Start larger to accommodate metrics without wrapping/truncation
        try:
            self.root.geometry("1400x700")
            self.root.minsize(1200, 700)
        except Exception:
            pass

        # UI Variables
        self.mode = tk.StringVar(value="robot_vs_maze")
        # Speed slider config (visual range) and mapping to delays (logic range)
        self.speed_min = 1; self.speed_max = 100000  # visual range; logic below clamps to [delay_fast_ms..delay_slow_ms]
        self.delay_fast_ms = 1   # allow ASAP scheduling for max speed
        self.delay_slow_ms = 200 # slowest frame delay (roughly previous behavior)
        self.speed = tk.IntVar(value=self.speed_max // 2)  # default mid-speed
        # Steps per tick: do multiple algorithm steps per UI frame for higher apparent FPS
        self.steps_per_tick = tk.IntVar(value=1)
        self.algorithm_var = tk.StringVar(value="Depth-First Search") # DFS default
        self.knows_maze_var = tk.BooleanVar(value=False)
        self.gen_method_var = tk.StringVar(value="DFS")
        self.loop_percent_var = tk.IntVar(value=0)
        self.num_rewards_var = tk.IntVar(value=0)
        self.max_weight_var = tk.IntVar(value=1)
        self.is_running = False

        self._setup_ui()
        self.start_new_simulation()

    def _setup_ui(self):
        # Configure widget styles
        style = ttk.Style(); style.configure("TButton", padding=6, relief="flat", background="#34495e", foreground="white"); style.map("TButton", background=[('active', '#4a627a')])
        # Custom button style with black text
        style.configure("Black.TButton", padding=6, relief="flat", background="#34495e", foreground="black"); style.map("Black.TButton", background=[('active', '#4a627a')])
        style.configure("TRadiobutton", background=BG_COLOR, foreground="white"); style.configure("TLabel", background=BG_COLOR, foreground="white")
        style.configure("TScale", background=BG_COLOR); style.configure("TCombobox", padding=5); style.configure("TSpinbox", padding=5)
        style.configure("TCheckbutton", background=BG_COLOR, foreground="white"); style.map("TCheckbutton", background=[('active', BG_COLOR)])
        style.map('TCheckbutton', indicatorcolor=[('disabled', '#7f8c8d')]); style.map('TCombobox', fieldbackground=[('disabled', '#34495e')]); style.map('TSpinbox', fieldbackground=[('disabled', '#34495e')])
        self.root.option_add('*TCombobox*Listbox.background', '#34495e'); self.root.option_add('*TCombobox*Listbox.foreground', 'white')
        self.root.option_add('*TCombobox*Listbox.selectBackground', '#4a627a'); self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')

        # Create UI Frames
        control_frame_top = tk.Frame(self.root, bg=BG_COLOR, padx=10, pady=5); control_frame_top.pack(side=tk.TOP, fill=tk.X)
        control_frame_bottom = tk.Frame(self.root, bg=BG_COLOR, padx=10, pady=5); control_frame_bottom.pack(side=tk.TOP, fill=tk.X)
        maze_frame = tk.Frame(self.root, bg=BG_COLOR, padx=10, pady=10); maze_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Top Control Row
        ttk.Button(control_frame_top, text="New Simulation", command=self.start_new_simulation, style="Black.TButton").pack(side=tk.LEFT, padx=5)
        self.knows_maze_check = ttk.Checkbutton(control_frame_top, text="Knows Full Maze?", variable=self.knows_maze_var, command=self.on_knows_maze_change); self.knows_maze_check.pack(side=tk.LEFT, padx=10)
        ttk.Label(control_frame_top, text="Algorithm:").pack(side=tk.LEFT, padx=(10, 5))
        # Updated algorithm list
        self.algo_combo = ttk.Combobox(control_frame_top, textvariable=self.algorithm_var, values=["A*", "Greedy BFS", "BFS", "Depth-First Search", "Monte Carlo"], state="readonly", width=18); self.algo_combo.pack(side=tk.LEFT, padx=5); self.algo_combo.bind("<<ComboboxSelected>>", self.on_algo_change)
        ttk.Label(control_frame_top, text="Mode:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(control_frame_top, text="Robot vs Maze", variable=self.mode, value="robot_vs_maze", command=self.start_new_simulation).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame_top, text="Robot vs Player", variable=self.mode, value="robot_vs_player", command=self.start_new_simulation).pack(side=tk.LEFT)
        ttk.Label(control_frame_top, text="Speed:").pack(side=tk.LEFT, padx=(10, 5)); ttk.Scale(control_frame_top, from_=self.speed_min, to=self.speed_max, orient=tk.HORIZONTAL, variable=self.speed).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(control_frame_top, text="Steps/tick:").pack(side=tk.LEFT, padx=(10, 5)); ttk.Spinbox(control_frame_top, from_=1, to=500, textvariable=self.steps_per_tick, width=6).pack(side=tk.LEFT)

        # Bottom Control Row
        ttk.Label(control_frame_bottom, text="Generator:").pack(side=tk.LEFT, padx=5)
        # Updated generator list (remove legacy Weighted Prim's)
        self.gen_combo = ttk.Combobox(control_frame_bottom, textvariable=self.gen_method_var, values=["DFS", "BFS", "Greedy Frontier", "Random Prim's", "Kruskal's MST"], state="readonly", width=22); self.gen_combo.pack(side=tk.LEFT, padx=5); self.gen_combo.bind("<<ComboboxSelected>>", self.on_generator_change)
        ttk.Label(control_frame_bottom, text="Loop %:").pack(side=tk.LEFT, padx=(10, 5)); self.loop_spinbox = ttk.Spinbox(control_frame_bottom, from_=0, to=100, increment=5, textvariable=self.loop_percent_var, width=5, command=self.start_new_simulation); self.loop_spinbox.pack(side=tk.LEFT)
        ttk.Label(control_frame_bottom, text="Rewards:").pack(side=tk.LEFT, padx=(10, 5)); self.reward_spinbox = ttk.Spinbox(control_frame_bottom, from_=0, to=10, textvariable=self.num_rewards_var, width=5, command=self.start_new_simulation); self.reward_spinbox.pack(side=tk.LEFT)
        ttk.Label(control_frame_bottom, text="Max Weight:").pack(side=tk.LEFT, padx=(10, 5)); self.max_weight_spinbox = ttk.Spinbox(control_frame_bottom, from_=1, to=10, textvariable=self.max_weight_var, width=5, command=self.start_new_simulation); self.max_weight_spinbox.pack(side=tk.LEFT)
        # Metrics label (right side) with fixed width to prevent window auto-widening
        self.metrics_var = tk.StringVar(value="")
        # No wrapping; make label wider so full metrics fit
        self.metrics_label = ttk.Label(control_frame_bottom, textvariable=self.metrics_var, width=140, anchor=tk.E)
        self.metrics_label.pack(side=tk.RIGHT)

        # Canvases
        canvas_size_w = CELL_SIZE * MAZE_WIDTH; canvas_size_h = CELL_SIZE * MAZE_HEIGHT
        self.left_canvas = tk.Canvas(maze_frame, width=canvas_size_w, height=canvas_size_h, bg=BG_COLOR, highlightthickness=0); self.left_canvas.pack(side=tk.LEFT, padx=10)
        self.right_canvas = tk.Canvas(maze_frame, width=canvas_size_w, height=canvas_size_h, bg=BG_COLOR, highlightthickness=0); self.right_canvas.pack(side=tk.RIGHT, padx=10)
        # Redraw on resize to maintain aspect without stretching
        self.left_canvas.bind("<Configure>", lambda e: self.draw_left_maze())
        self.right_canvas.bind("<Configure>", lambda e: self.draw_right_maze())

        # Player Keybinds
        self.root.bind("<KeyPress-Up>", lambda e: self.move_player('N')); self.root.bind("<KeyPress-Down>", lambda e: self.move_player('S'))
        self.root.bind("<KeyPress-Left>", lambda e: self.move_player('W')); self.root.bind("<KeyPress-Right>", lambda e: self.move_player('E'))

        self._update_ui_state() # Set initial UI state

    # --- UI Event Handlers ---
    def on_algo_change(self, event=None): self._update_ui_state(); self.start_new_simulation()
    def on_knows_maze_change(self, event=None): self._update_ui_state(); self.start_new_simulation()
    def on_generator_change(self, event=None): self._update_ui_state(); self.start_new_simulation()

    def _update_ui_state(self):
        """Manages enabling/disabling UI controls based on selections."""
        # Generator controls: Disable Max Weight for MST generators (use edge weights)
        # Loops are now enabled for all generators including Kruskal's
        is_mst_gen = self.gen_method_var.get() in ["Kruskal's MST"]
        self.max_weight_spinbox.configure(state='disabled' if is_mst_gen else 'normal')
        # Loop spinbox always enabled
        self.loop_spinbox.configure(state='normal')

        # Algorithm controls: Disable "Knows Maze" for online-only algorithms
        online_only_algos = ["Monte Carlo", "Depth-First Search", "Greedy BFS", "BFS"]
        is_online_only = self.algorithm_var.get() in online_only_algos

        if is_online_only:
            self.knows_maze_var.set(False); self.knows_maze_check.configure(state='disabled')
        else: # A*, GFS can be either
            self.knows_maze_check.configure(state='normal')

        # Filter algorithm list based on knows_maze state
        if self.knows_maze_var.get() == True: # Offline mode selected
            valid_offline_algos = ["A*", "Prim Solver", "Kruskal Solver"]
            self.algo_combo.configure(values=valid_offline_algos)
            if self.algorithm_var.get() not in valid_offline_algos:
                self.algorithm_var.set("A*") # Default to A*
        else: # Online mode selected
            # A* and Greedy BFS added
            all_algos = ["A*", "Greedy BFS", "BFS", "Depth-First Search", "Monte Carlo"]
            self.algo_combo.configure(values=all_algos)

    def start_new_simulation(self):
        """
        Initializes a new simulation with fresh maze and robot instances.
        
        Maze Data Structure Context:
          - Creates new Maze object with self.maze.grid (directional passages)
          - Each cell (x,y) in grid has set of {'N','S','E','W'} for open passages
          - Generation method determines maze topology and weight distribution
        
        Initialization Sequence:
          1. Stop any running simulation
          2. Read UI parameters (algorithm, generator, weights, rewards, loops)
          3. Create new Maze with selected generator (populates grid structure)
          4. Create primary Robot with selected algorithm and mode (online/offline)
          5. If robot-vs-player mode, create player Robot at same start/goal
          6. Render initial state on both canvases
          7. Start animation loop
        
        Parameter Handling:
          - Loops now enabled for all generators including Kruskal's MST
          - Online-only algorithms force knows_maze=False
          - Player robot always uses online exploration (fair competition)
        
        Integration:
          - Called on app startup and when user clicks "New Simulation"
          - Called automatically when UI parameters change (algorithm, generator)
          - Synchronizes all UI state with simulation objects
          - Resets all metrics and visualization state
        """
        self.is_running = False
        loop_percent = self.loop_percent_var.get()
        # Loops now allowed for all generators
        
        self.maze = Maze(MAZE_WIDTH, MAZE_HEIGHT, self.gen_method_var.get(), loop_percent, self.num_rewards_var.get(), self.max_weight_var.get())
        self.robot = Robot(self.maze, self.algorithm_var.get(), self.knows_maze_var.get())
        self.player = None
        if self.mode.get() == "robot_vs_player":
            # Create player and align start/goal to robot's for a fair race
            self.player = Robot(self.maze, algorithm="Depth-First Search", knows_maze=False); self.player.color = PLAYER_COLOR
            self.player.goal_pos = self.robot.goal_pos
            self.player.unvisited_rewards = set(self.maze.rewards)
            self.player.start_pos = self.robot.start_pos
            self.player.x, self.player.y = self.player.start_pos
            self.player.path = [self.player.start_pos]
            # Reinitialize online search structures with the new start
            self.player.search_area.clear()
            self.player._setup_online_search()
        self.draw_all(); self._update_metrics_label(); self.is_running = True; self.root.after(100, self.update_loop)

    def _color_lerp(self, c1, c2, t):
        c1_rgb = tuple(int(c1[i:i+2], 16) for i in (1, 3, 5)); c2_rgb = tuple(int(c2[i:i+2], 16) for i in (1, 3, 5))
        r = int(c1_rgb[0] + (c2_rgb[0] - c1_rgb[0]) * t); g = int(c1_rgb[1] + (c2_rgb[1] - c1_rgb[1]) * t); b = int(c1_rgb[2] + (c2_rgb[2] - c1_rgb[2]) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    def _get_weight_color(self, w, max_w):
        if max_w <= 1: return DEFAULT_CELL_COLOR
        t = (w - 1) / max(1, max_w - 1)
        if t < 0.5: return self._color_lerp(WEIGHT_COLOR_LOW, WEIGHT_COLOR_MID, t * 2)
        else: return self._color_lerp(WEIGHT_COLOR_MID, WEIGHT_COLOR_HIGH, (t - 0.5) * 2)

    def update_loop(self):
        """
        Main simulation loop executing robot steps and rendering updates.
        
        Execution Flow:
          1. Check if simulation is still running
          2. Execute multiple robot steps per tick (controlled by steps_per_tick)
          3. Render updated maze state on appropriate canvas(es)
          4. Update metrics display
          5. Check for completion conditions
          6. Schedule next tick with calculated delay
        
        Modes:
          - Robot vs Maze: Robot steps and right canvas updates
          - Robot vs Player: Robot steps, player controlled by keyboard
        
        Animation Control:
          - Speed slider: Maps linearly to delay range [delay_fast_ms, delay_slow_ms]
          - Steps per tick: Multiple algorithm steps per frame for faster execution
          - Delay calculation: Interpolates between 1ms (fast) and 200ms (slow)
        
        Integration:
          - Scheduled via root.after() for non-blocking animation
          - Each tick calls robot.step() which advances algorithm state
          - Rendering calls use maze.grid to draw walls/passages
          - Metrics updated from robot.metrics dictionary
        """
        if not self.is_running: return
        steps = max(1, int(self.steps_per_tick.get()))
        if self.mode.get() == "robot_vs_maze":
            for _ in range(steps):
                if self.robot.is_done: break
                self.robot.step()
            self.draw_right_maze(); self._update_metrics_label()
            if self.robot.is_done: self.is_running = False; self.draw_winner_message("Robot Finished!")
        elif self.mode.get() == "robot_vs_player":
            for _ in range(steps):
                if self.robot.is_done or (self.player and self.player.is_done): break
                self.robot.step()
            self.draw_left_maze(); self._update_metrics_label(); self.check_winner()
        # Map slider linearly to a bounded delay range independent of speed_max magnitude
        span = max(1, self.speed_max - self.speed_min)
        norm = (self.speed.get() - self.speed_min) / span  # 0..1
        delay = int(self.delay_slow_ms - norm * (self.delay_slow_ms - self.delay_fast_ms))
        self.root.after(max(0, delay), self.update_loop)  # Ensure non-negative delay

    def _update_metrics_label(self):
        m = self.robot.metrics
        text = (
            f"Algorithm steps: {m['algorithm_steps']}  |  "
            f"Unique explored: {m['unique_explored']}  |  "
            f"Baseline cost: {m['baseline_optimal_cost']}  |  "
            f"Solution cost: {m['solution_cost']}  |  "
            f"Animation cost: {m['animation_walk_cost']}  |  "
            f"Path length: {m['total_path_length']}  |  "
            f"Frontier max: {m['frontier_max']}"
        )
        self.metrics_var.set(text)

    def check_winner(self):
        if self.robot.is_done: self.is_running = False; self.draw_winner_message("Robot Wins!")
        elif self.player and self.player.is_done: self.is_running = False; self.draw_winner_message("Player Wins!")

    def move_player(self, direction):
        """
        Handles human player movement in robot-vs-player mode via keyboard input.
        
        Maze Data Structure Context:
          - Uses maze.get_valid_moves(x, y) to check if direction is passable
          - Valid moves returned as set of {'N', 'S', 'E', 'W'} from maze.grid
          - Can only move if requested direction is in the valid moves set
        
        Movement Logic:
          - 'N': Decreases y by 1 (move up)
          - 'S': Increases y by 1 (move down)
          - 'W': Decreases x by 1 (move left)
          - 'E': Increases x by 1 (move right)
          - Only moves if passage exists (direction in valid_moves set)
        
        State Updates:
          - Appends new position to player.path (movement history)
          - Adds position to player.search_area (discovered cells)
          - Removes reward if player lands on one
          - Checks win condition: at goal with all rewards collected
        
        Parameters:
          direction (str): One of 'N', 'S', 'E', 'W' from arrow key binding
        
        Integration:
          - Bound to arrow keys: Up→N, Down→S, Left→W, Right→E
          - Only active in robot-vs-player mode while simulation running
          - Triggers re-render of right canvas to show updated player state
          - Checks for winner after each move
        """
        if self.mode.get() == "robot_vs_player" and self.is_running:
            if direction in self.maze.get_valid_moves(self.player.x, self.player.y):
                if direction=='N': self.player.y-=1; 
                elif direction=='S': self.player.y+=1
                elif direction=='W': self.player.x-=1; 
                elif direction=='E': self.player.x+=1
                self.player.path.append((self.player.x, self.player.y)); self.player.search_area.add((self.player.x, self.player.y))
                if (self.player.x,self.player.y) in self.player.unvisited_rewards: self.player.unvisited_rewards.remove((self.player.x,self.player.y))
                if (self.player.x,self.player.y)==self.player.goal_pos and not self.player.unvisited_rewards: self.player.is_done=True
                self.draw_right_maze(); self.check_winner()

    def draw_all(self):
        if self.mode.get()=="robot_vs_maze": self._draw_maze_on_canvas(self.left_canvas,self.maze,True,self.robot); self._draw_maze_on_canvas(self.right_canvas,self.maze,False,self.robot)
        else: self.draw_left_maze(); self.draw_right_maze()
    def draw_left_maze(self):
        show_full = True if self.mode.get()=="robot_vs_maze" else False
        self._draw_maze_on_canvas(self.left_canvas, self.maze, show_full, self.robot)
    def draw_right_maze(self): agent = self.robot if self.mode.get()=="robot_vs_maze" else self.player; self._draw_maze_on_canvas(self.right_canvas, self.maze, False, agent)

    def _draw_maze_on_canvas(self, canvas, maze, show_full_maze, agent):
        """
        Renders the maze and agent state onto a Tkinter canvas.
        
        Maze Data Structure Context:
          - maze.grid[(x, y)] contains set of open directions {'N','S','E','W'}
          - Empty set = all walls present, full set = no walls (open cell)
          - Walls are drawn where directions are ABSENT from the set
          - Passages exist where directions are PRESENT in the set
        
        Rendering Algorithm (Two-Pass):
          Pass 1 - Cell Backgrounds:
            - Iterate all cells, draw rectangles with appropriate fill colors
            - Known cells: colored by weight (node/edge-weighted)
            - Unknown cells (online mode): dark background
            - Search area: darker gray overlay
            - Path cells: highlighted based on algorithm state
          
          Pass 2 - Walls:
            - For each cell, check which directions are ABSENT from grid set
            - Draw wall lines on sides where no passage exists
            - Wall drawing on top prevents overlap artifacts
        
        Coordinate System:
          - Canvas coordinates: (0, 0) at top-left, pixel-based
          - Maze coordinates: (x, y) where x is column, y is row
          - Transformation: canvas_x = x_offset + x * cell_size
          - Maintains square cells with integer pixel alignment
        
        Visibility Logic:
          - show_full_maze=True: render entire maze (oracle view)
          - show_full_maze=False: render only discovered cells (agent.search_area)
          - Offline solvers: show full maze (knows everything)
          - Online explorers: show only explored regions (fog of war)
        
        Visual Elements:
          - Cell colors: weight-based gradient (blue→green→red)
          - Walls: thick gray lines where passages don't exist
          - Paths: highlighted cells (search area, final path, known path)
          - Markers: start (teal), goal (red), rewards (orange), agent (yellow/blue)
        
        Parameters:
          canvas: Tkinter Canvas widget to draw on
          maze: Maze object containing grid structure and weights
          show_full_maze (bool): If True, show entire maze; if False, show only discovered
          agent: Robot object whose perspective to render
        
        Integration:
          - Called by draw_all(), draw_left_maze(), draw_right_maze()
          - Renders maze.grid directional representation as visual walls
          - Updates every simulation tick to animate agent movement
        """
        canvas.delete("all"); c_width=canvas.winfo_width(); c_height=canvas.winfo_height()
        if c_width<=1 or c_height<=1: canvas.after(50, lambda: self._draw_maze_on_canvas(canvas,maze,show_full_maze,agent)); return
        # Maintain square cells with integer pixel grid to avoid jitter
        cell_size_f = min(c_width / self.maze.width, c_height / self.maze.height)
        cell_size = max(1, int(cell_size_f))
        draw_w = cell_size * self.maze.width; draw_h = cell_size * self.maze.height
        x_off = int((c_width - draw_w) // 2); y_off = int((c_height - draw_h) // 2)
        wall_width = max(5, int(cell_size / 8))
        # For edge-weighted mazes (Kruskal), prepare color scaling
        edge_color_max = None
        if maze.weight_type == 'edge' and maze.edge_weights:
            try:
                edge_color_max = max(maze.edge_weights.values())
            except Exception:
                edge_color_max = 1
            if edge_color_max < 1: edge_color_max = 1
        # Pass 1: draw all rectangles (background and known cells)
        for y in range(maze.height):
            for x in range(maze.width):
                x1,y1,x2,y2 = int(x_off + x*cell_size), int(y_off + y*cell_size), int(x_off + (x+1)*cell_size), int(y_off + (y+1)*cell_size)
                is_known = show_full_maze or agent.is_solver or (x,y) in agent.search_area
                if is_known:
                    if maze.weight_type=='node' and maze.max_weight>1:
                        fill_color = self._get_weight_color(maze.node_weights.get((x,y),1), maze.max_weight)
                    elif maze.weight_type=='edge' and edge_color_max:
                        nbs = maze.get_neighbors(x, y)
                        if nbs:
                            weights = [maze.edge_weights.get(((x, y), nb), 1) for nb in nbs]
                            avg_w = sum(weights) / max(1, len(weights))
                        else:
                            avg_w = 1
                        fill_color = self._get_weight_color(avg_w, edge_color_max)
                    else:
                        fill_color = DEFAULT_CELL_COLOR
                    if not show_full_maze:
                        if (x,y) in agent.search_area: fill_color = SEARCH_AREA_COLOR
                        if agent.is_solver and (x,y) in agent.final_path: fill_color = FINAL_PATH_COLOR
                        elif not agent.is_solver and (x,y) in agent.path: fill_color = KNOWN_PATH_COLOR
                    canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="")
                else:
                    if not show_full_maze:
                        canvas.create_rectangle(x1, y1, x2, y2, fill=BG_COLOR, outline="")

        # Pass 2: draw walls on top to avoid being covered by later rectangles
        for y in range(maze.height):
            for x in range(maze.width):
                x1,y1,x2,y2 = int(x_off + x*cell_size), int(y_off + y*cell_size), int(x_off + (x+1)*cell_size), int(y_off + (y+1)*cell_size)
                is_known = show_full_maze or agent.is_solver or (x,y) in agent.search_area
                if not show_full_maze and not is_known:
                    continue
                open_walls = maze.get_valid_moves(x, y)
                if 'N' not in open_walls: canvas.create_line(x1, y1, x2, y1, fill=WALL_COLOR, width=wall_width)
                if 'S' not in open_walls: canvas.create_line(x1, y2, x2, y2, fill=WALL_COLOR, width=wall_width)
                if 'W' not in open_walls: canvas.create_line(x1, y1, x1, y2, fill=WALL_COLOR, width=wall_width)
                if 'E' not in open_walls: canvas.create_line(x2, y1, x2, y2, fill=WALL_COLOR, width=wall_width)
        # For edge-weighted mazes, weights are visualized via tile colors above (no numeric/edge overlays)
        for (rx,ry) in maze.rewards:
            if show_full_maze:
                self._draw_marker(canvas,(rx,ry),REWARD_COLOR,shape='oval')
            else:
                if (rx,ry) in agent.unvisited_rewards: self._draw_marker(canvas,(rx,ry),REWARD_COLOR,shape='oval')
                elif (rx,ry) in agent.search_area or agent.is_solver: self._draw_marker(canvas,(rx,ry),SEARCH_AREA_COLOR,shape='oval')
        # Start/goal markers: full view shows robot's; per-agent views show only that agent's markers
        if show_full_maze:
            self._draw_marker(canvas,self.robot.start_pos,START_COLOR)
            self._draw_marker(canvas,self.robot.goal_pos,GOAL_COLOR)
        else:
            if agent==self.robot:
                self._draw_marker(canvas,self.robot.start_pos,START_COLOR)
                self._draw_marker(canvas,self.robot.goal_pos,GOAL_COLOR)
            else:
                if self.player:
                    self._draw_marker(canvas,self.player.start_pos,START_COLOR)
                    self._draw_marker(canvas,self.player.goal_pos,GOAL_COLOR)
        # Only draw the moving agent on the non-full view
        if not show_full_maze:
            agent_color = ROBOT_COLOR if agent==self.robot else PLAYER_COLOR
            self._draw_marker(canvas,(agent.x,agent.y),agent_color,is_agent=True)

    def _draw_marker(self, canvas, pos, color, is_agent=False, shape='rect'):
        c_width=canvas.winfo_width(); c_height=canvas.winfo_height()
        # Match maze drawing mapping (square cells centered on integer grid)
        cell_size_f = min(c_width / self.maze.width, c_height / self.maze.height)
        cell_size = max(1, int(cell_size_f))
        draw_w = cell_size * self.maze.width; draw_h = cell_size * self.maze.height
        x_off = int((c_width - draw_w) // 2); y_off = int((c_height - draw_h) // 2)
        x, y = pos; margin_frac = 0.2 if not is_agent else 0.25; m = int(cell_size*margin_frac)
        x1, y1 = int(x_off + x*cell_size + m), int(y_off + y*cell_size + m)
        x2, y2 = int(x_off + (x+1)*cell_size - m), int(y_off + (y+1)*cell_size - m)
        if shape=='oval': canvas.create_oval(x1,y1,x2,y2,fill=color,outline="")
        else: canvas.create_rectangle(x1,y1,x2,y2,fill=color,outline="")

    def draw_winner_message(self, message):
        canvas = self.right_canvas; w = canvas.winfo_width(); h = canvas.winfo_height()
        if self.mode.get()=='robot_vs_player': canvas = self.left_canvas if "Robot" in message else self.right_canvas
        if w<=1 or h<=1: self.root.after(100, lambda: self.draw_winner_message(message)); return
        canvas.create_rectangle(w/2-100,h/2-30,w/2+100,h/2+30,fill=BG_COLOR,outline="white",width=2)
        canvas.create_text(w/2,h/2,text=message,fill="white",font=("Helvetica",16,"bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
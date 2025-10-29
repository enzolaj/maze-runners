import tkinter as tk
from tkinter import ttk
import random
import collections
import heapq
import math

# --- Configuration ---
CELL_SIZE = 20
MAZE_WIDTH = 25
MAZE_HEIGHT = 25

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
    Generates and stores the maze structure.

    Two weight models:
      - Node-weighted: the cost is associated to entering a cell (terrain cost).
        Methods: DFS generation, Random Prim's, BFS, Greedy Frontier. In this
        mode we color cells by weight (if max_weight > 1), and the cost of a
        path is the sum of node weights along the entered cells.

      - Edge-weighted: the cost is associated to traversing an edge between
        neighboring cells. Methods: Weighted Prim's (MST), Kruskal's (MST).
        In this mode we draw edge weights in the full-maze view.

    Minimum Spanning Tree (MST) refresher:
      - Given a connected, undirected graph G=(V,E) with non-negative edge
        weights w(e), an MST is a subset T ⊆ E that connects all |V| vertices,
        has no cycles, and minimizes the total weight sum_{e∈T} w(e).
      - MSTs are not necessarily unique; when weights are distinct the MST is
        unique. MSTs optimize total tree weight, not shortest-path distances for
        arbitrary pairs (thats a different problem: shortest paths).
      - Two classic algorithms:
          • Prims: grows a single tree by repeatedly adding the lightest edge
            crossing the cut between the growing tree and the rest of the graph
            (a greedy cut-optimal strategy). With a binary heap, complexity is
            O(E log V).
          • Kruskals: sorts all edges by weight and adds them in ascending order
            while skipping those that would form a cycle, using a Disjoint Set
            Union (Union-Find) data structure (path compression + union by rank)
            for near-linear performance: O(E log E) for sorting plus almost-
            constant amortized Union-Find operations.
    """
    def __init__(self, width, height, gen_method='DFS', loop_percent=0, num_rewards=0, max_weight=1):
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
        if gen_method == "Weighted Prim's (MST)":
            self.weight_type = 'edge'
            self._generate_weighted_prims()
        elif gen_method == "Kruskal's MST":
            self.weight_type = 'edge'
            self._generate_kruskal()
        elif gen_method == 'Random Prim\'s':
            self.weight_type = 'node'
            self._generate_random_prims()
            self._populate_node_weights(max_weight)
        elif gen_method == 'Greedy Frontier':
            self.weight_type = 'node'
            self._generate_greedy_frontier()
            self._populate_node_weights(max_weight)
        elif gen_method == 'BFS':  # <-- ADDED BFS
            self.weight_type = 'node'
            self._generate_bfs()
            self._populate_node_weights(max_weight)
        else: # Default to DFS
            self.weight_type = 'node'
            self._generate_dfs()
            self._populate_node_weights(max_weight)

        # After generation, optionally knock down walls to create loops.
        # Note: Loops disabled for MST generators in MazeApp._update_ui_state
        if loop_percent > 0:
            self._add_loops(loop_percent)

        # Place rewards. This is independent of generation method.
        self._populate_rewards(num_rewards)

class Maze:
    """
    Generates and stores the maze structure.
    Supports node-weighted (DFS, BFS, Random Prim's) and
    edge-weighted (Weighted Prim's MST, Kruskal's MST) generators.
    """
    def __init__(self, width, height, gen_method='DFS', loop_percent=0, num_rewards=0, max_weight=1):
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
        if gen_method == "Weighted Prim's (MST)":
            self.weight_type = 'edge'
            self._generate_weighted_prims()
        elif gen_method == "Kruskal's MST":
            self.weight_type = 'edge'
            self._generate_kruskal()
        elif gen_method == 'Random Prim\'s':
            self.weight_type = 'node'
            self._generate_random_prims()
            self._populate_node_weights(max_weight)
        elif gen_method == 'BFS':  # <-- ADDED BFS
            self.weight_type = 'node'
            self._generate_bfs()
            self._populate_node_weights(max_weight)
        else: # Default to DFS
            self.weight_type = 'node'
            self._generate_dfs()
            self._populate_node_weights(max_weight)

        # After generation, optionally knock down walls to create loops.
        # Note: Loops disabled for MST generators in MazeApp._update_ui_state
        if loop_percent > 0:
            self._add_loops(loop_percent)

        # Place rewards. This is independent of generation method.
        self._populate_rewards(num_rewards)

    def _generate_dfs(self):
        """Generates a "perfect" maze using Randomized Depth-First Search.

        Idea (backtracking/recursive backtracker):
          - Start from a random cell, perform DFS, and whenever you find an
            unvisited neighbor, carve a passage to it and descend. When stuck,
            backtrack until another unvisited neighbor exists. The union of
            carved passages forms a spanning tree of the grid graph, i.e., a
            perfect maze (exactly one simple path between any two cells).
        """
        visited = set()
        stack = [(random.randint(0, self.width - 1), random.randint(0, self.height - 1))]
        visited.add(stack[0])
        while stack:
            current_x, current_y = stack[-1]
            neighbors = []
            for dx, dy, direction, opposite in [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'),
                                                (-1, 0, 'W', 'E'), (1, 0, 'E', 'W')]:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, direction, opposite))
            if neighbors:
                nx, ny, direction, opposite = random.choice(neighbors)
                self.grid[(current_x, current_y)].add(direction)
                self.grid[(nx, ny)].add(opposite)
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

    def _generate_random_prims(self):
        """Generates a "perfect" maze using Randomized Prim's Algorithm.

        High-level:
          - Maintain a frontier set of cells adjacent to the visited region.
          - Repeatedly select a random frontier cell and connect it to one of
            its already-visited neighbors. This yields a uniform-ish spanning
            tree with different texture than the DFS backtracker.
        """
        visited = set()
        start_x, start_y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        visited.add((start_x, start_y))
        frontier = set()
        for dx, dy, _, _ in [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'), (-1, 0, 'W', 'E'), (1, 0, 'E', 'W')]:
            nx, ny = start_x + dx, start_y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                frontier.add((nx, ny))
        while frontier:
            current_x, current_y = random.choice(list(frontier))
            frontier.remove((current_x, current_y))
            visited.add((current_x, current_y))
            visited_neighbors = []
            for dx, dy, direction, opposite in [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'),
                                                (-1, 0, 'W', 'E'), (1, 0, 'E', 'W')]:
                nx, ny = current_x + dx, current_y + dy
                if (nx, ny) in visited:
                    visited_neighbors.append((nx, ny, direction, opposite))
            if visited_neighbors:
                nx, ny, direction, opposite = random.choice(visited_neighbors)
                self.grid[(current_x, current_y)].add(direction)
                self.grid[(nx, ny)].add(opposite)
            for dx, dy, _, _ in [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'), (-1, 0, 'W', 'E'), (1, 0, 'E', 'W')]:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    frontier.add((nx, ny))

    def _generate_bfs(self):
        """Generates a maze using randomized Breadth-First Search.

        Note:
          - Unlike shortest-path BFS, here BFS is used as a spanning-tree
            generator by enqueuing newly discovered neighbors and immediately
            carving passages as they are discovered. Randomizing neighbor order
            changes the maze’s texture while preserving tree property.
        """
        visited = set()
        start = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        visited.add(start)
        queue = collections.deque([start])

        while queue:
            cx, cy = queue.popleft()
            directions = [(0,-1,'N','S'), (0,1,'S','N'), (-1,0,'W','E'), (1,0,'E','W')]
            random.shuffle(directions)
            for dx, dy, d1, d2 in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    # connect current to neighbor
                    self.grid[(cx, cy)].add(d1)
                    self.grid[(nx, ny)].add(d2)
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    def _generate_greedy_frontier(self):
        """Generates a maze using a greedy frontier approach (matches gfs.py).

        Greedy Frontier heuristic:
          - Maintain a list of frontier cells (discovered, not finalized).
          - Always pick the frontier cell that is closest (by Manhattan distance)
            to a random goal anchor. This biases the tree to grow toward that
            anchor, yielding long corridors with occasional branches.
        """
        visited = set()
        start = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        goal = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        while goal == start:
            goal = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

        frontier = [start]
        visited.add(start)

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        while frontier:
            frontier.sort(key=lambda cell: manhattan(cell, goal))
            current_x, current_y = frontier.pop(0)

            neighbors = []
            for dx, dy, direction, opposite in [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'), (-1, 0, 'W', 'E'), (1, 0, 'E', 'W')]:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, direction, opposite))

            if neighbors:
                nx, ny, direction, opposite = random.choice(neighbors)
                self.grid[(current_x, current_y)].add(direction)
                self.grid[(nx, ny)].add(opposite)
                visited.add((nx, ny))
                frontier.append((current_x, current_y))
                frontier.append((nx, ny))

    def _generate_weighted_prims(self):
        """Generates a maze using weighted Prim’s (Minimum Spanning Tree).

        Graph model:
          - Grid cells are vertices; edges exist between 4-neighbors.
          - We assign a random edge weight (optionally modulated by distance,
            in variants) and run Prim’s algorithm using a min-heap frontier.

        Algorithm sketch:
          - Start from a random seed cell, push all incident edges with weights.
          - Repeatedly pop the lightest edge joining the current tree to a new
            cell; carve that passage and push new incident edges.
          - The carved passages form an MST in the grid graph.
        """
        visited = set()
        start = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        visited.add(start)
        frontier = [] # A heapq
        def add_frontier_edges(cx, cy):
            for dx, dy, direction, opposite in [(0,-1,'N','S'),(0,1,'S','N'),(-1,0,'W','E'),(1,0,'E','W')]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    w = random.randint(1, 10)
                    self.edge_weights[((cx, cy), (nx, ny))] = w
                    self.edge_weights[((nx, ny), (cx, cy))] = w
                    heapq.heappush(frontier, (w, (cx, cy), (nx, ny), direction, opposite))
        add_frontier_edges(*start)
        while frontier:
            w, cell, neighbor, direction, opposite = heapq.heappop(frontier)
            if neighbor in visited: continue
            visited.add(neighbor)
            cx, cy = cell; nx, ny = neighbor
            self.grid[(cx, cy)].add(direction)
            self.grid[(nx, ny)].add(opposite)
            add_frontier_edges(nx, ny)

    # --- Kruskal's Generator Helper Methods ---
    def _find_set(self, parent, u):
        """Find operation for Disjoint Set Union (Union-Find).

        Path compression:
          - During find, reassign parent[u] directly to the representative.
            This yields inverse-Ackermann amortized time per operation.
        """
        if parent[u] != u:
            parent[u] = self._find_set(parent, parent[u]) # Path compression
        return parent[u]

    def _union_sets(self, parent, rank, u, v):
        """Union operation for DSU with union by rank.

        Union by rank:
          - Attach the root with smaller rank under the root with larger rank.
            If ranks are equal, arbitrarily pick one and increment its rank.
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
        Generates a maze using Kruskal's algorithm (Minimum Spanning Tree).

        Kruskal’s MST algorithm:
          1) Build the edge list (each grid-adjacent pair) with random weights.
          2) Sort edges by ascending weight: O(E log E).
          3) Initialize DSU with each vertex in its own set.
          4) Scan edges in order; for an edge (u,v), if find(u) != find(v),
             then add the edge (carve the wall) and union(u,v). Otherwise skip
             to avoid cycles. Stop when we connected all vertices (|V|-1 edges).

        The resulting carved walls form a spanning tree minimizing total edge
        weight in the grid graph.
        """
        # 1. Create a list of all potential edges with random weights
        edges = []
        for y in range(self.height):
            for x in range(self.width):
                u = (x, y)
                # Edge to the East
                if x < self.width - 1:
                    v = (x + 1, y)
                    weight = random.randint(1, 10)
                    edges.append((weight, u, v))
                # Edge to the South
                if y < self.height - 1:
                    v = (x, y + 1)
                    weight = random.randint(1, 10)
                    edges.append((weight, u, v))

        # 2. Sort edges by weight
        edges.sort() # Sorts by the first element (weight)

        # 3. Initialize Disjoint Set Union (DSU) data structure
        parent = {} # parent[node] -> representative of the set
        rank = {}   # rank[node] -> upper bound on tree height for optimization
        for y in range(self.height):
            for x in range(self.width):
                node = (x, y)
                parent[node] = node
                rank[node] = 0

        # 4. Iterate through sorted edges and add to MST if they connect different sets
        num_edges = 0
        total_cells = self.width * self.height
        for weight, u, v in edges:
            # If adding this edge connects two previously unconnected components
            if self._union_sets(parent, rank, u, v):
                # Add edge to maze (carve wall) and store weight
                x1, y1 = u; x2, y2 = v
                self.edge_weights[(u, v)] = weight
                self.edge_weights[(v, u)] = weight # Bidirectional

                if x1 == x2: # Vertical edge
                    if y1 < y2: self.grid[u].add('S'); self.grid[v].add('N')
                    else:       self.grid[u].add('N'); self.grid[v].add('S')
                else: # Horizontal edge
                    if x1 < x2: self.grid[u].add('E'); self.grid[v].add('W')
                    else:       self.grid[u].add('W'); self.grid[v].add('E')

                num_edges += 1
                # Optimization: Stop when we have enough edges for a spanning tree
                if num_edges >= total_cells - 1:
                    break

    def _add_loops(self, percentage):
        """Creates an "imperfect" maze by removing extra walls."""
        num_walls_to_remove = int(self.width * self.height * percentage / 100)
        for _ in range(num_walls_to_remove):
            x = random.randint(0, self.width - 1); y = random.randint(0, self.height - 1)
            possible_walls = []
            # Check potential walls to remove (that currently exist)
            if y > 0 and 'N' not in self.grid[(x, y)]: possible_walls.append(('N', 'S', x, y - 1))
            if y < self.height - 1 and 'S' not in self.grid[(x, y)]: possible_walls.append(('S', 'N', x, y + 1))
            if x > 0 and 'W' not in self.grid[(x, y)]: possible_walls.append(('W', 'E', x - 1, y))
            if x < self.width - 1 and 'E' not in self.grid[(x, y)]: possible_walls.append(('E', 'W', x + 1, y))
            if possible_walls:
                direction, opposite, nx, ny = random.choice(possible_walls)
                self.grid[(x, y)].add(direction); self.grid[(nx, ny)].add(opposite) # Add connection

    def _populate_node_weights(self, max_weight):
        """Assigns a "terrain cost" (node weight) to every cell."""
        for y in range(self.height):
            for x in range(self.width):
                self.node_weights[(x, y)] = random.randint(1, max_weight)

    def _populate_rewards(self, num_rewards):
        """Places 'num_rewards' at random locations."""
        for _ in range(num_rewards):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in self.rewards: self.rewards.add((x, y)); break

    def get_valid_moves(self, x, y):
        """Returns the set of open directions ('N', 'S', 'E', 'W') for a cell."""
        return self.grid.get((x,y), set())

    def get_neighbors(self, x, y):
        """Returns a list of (nx, ny) coordinates reachable from (x, y)."""
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
        """Gets the cost of a move, respecting maze's weight type."""
        if self.weight_type == 'edge': return self.edge_weights.get((pos_from, pos_to), 1)
        else: return self.node_weights.get(pos_to, 1)

    # --- Kruskal's Generator Helper Methods ---
    def _find_set(self, parent, u):
        """Find operation for disjoint set union (DSU)."""
        if parent[u] != u:
            parent[u] = self._find_set(parent, parent[u]) # Path compression
        return parent[u]

    def _union_sets(self, parent, rank, u, v):
        """Union operation for disjoint set union (DSU)."""
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
        Generates a maze using Kruskal's algorithm (Minimum Spanning Tree).
        Assigns random weights to all potential walls (edges) and adds
        edges to the tree in increasing order of weight, as long as they
        don't form a cycle.
        """
        # 1. Create a list of all potential edges with random weights
        edges = []
        for y in range(self.height):
            for x in range(self.width):
                u = (x, y)
                # Edge to the East
                if x < self.width - 1:
                    v = (x + 1, y)
                    weight = random.randint(1, 10)
                    edges.append((weight, u, v))
                # Edge to the South
                if y < self.height - 1:
                    v = (x, y + 1)
                    weight = random.randint(1, 10)
                    edges.append((weight, u, v))

        # 2. Sort edges by weight
        edges.sort() # Sorts by the first element (weight)

        # 3. Initialize Disjoint Set Union (DSU) data structure
        parent = {} # parent[node] -> representative of the set
        rank = {}   # rank[node] -> upper bound on tree height for optimization
        for y in range(self.height):
            for x in range(self.width):
                node = (x, y)
                parent[node] = node
                rank[node] = 0

        # 4. Iterate through sorted edges and add to MST if they connect different sets
        num_edges = 0
        total_cells = self.width * self.height
        for weight, u, v in edges:
            # If adding this edge connects two previously unconnected components
            if self._union_sets(parent, rank, u, v):
                # Add edge to maze (carve wall) and store weight
                x1, y1 = u; x2, y2 = v
                self.edge_weights[(u, v)] = weight
                self.edge_weights[(v, u)] = weight # Bidirectional

                if x1 == x2: # Vertical edge
                    if y1 < y2: self.grid[u].add('S'); self.grid[v].add('N')
                    else:       self.grid[u].add('N'); self.grid[v].add('S')
                else: # Horizontal edge
                    if x1 < x2: self.grid[u].add('E'); self.grid[v].add('W')
                    else:       self.grid[u].add('W'); self.grid[v].add('E')

                num_edges += 1
                # Optimization: Stop when we have enough edges for a spanning tree
                if num_edges >= total_cells - 1:
                    break

    def _add_loops(self, percentage):
        """Creates an "imperfect" maze by removing extra walls."""
        num_walls_to_remove = int(self.width * self.height * percentage / 100)
        for _ in range(num_walls_to_remove):
            x = random.randint(0, self.width - 1); y = random.randint(0, self.height - 1)
            possible_walls = []
            # Check potential walls to remove (that currently exist)
            if y > 0 and 'N' not in self.grid[(x, y)]: possible_walls.append(('N', 'S', x, y - 1))
            if y < self.height - 1 and 'S' not in self.grid[(x, y)]: possible_walls.append(('S', 'N', x, y + 1))
            if x > 0 and 'W' not in self.grid[(x, y)]: possible_walls.append(('W', 'E', x - 1, y))
            if x < self.width - 1 and 'E' not in self.grid[(x, y)]: possible_walls.append(('E', 'W', x + 1, y))
            if possible_walls:
                direction, opposite, nx, ny = random.choice(possible_walls)
                self.grid[(x, y)].add(direction); self.grid[(nx, ny)].add(opposite) # Add connection

    def _populate_node_weights(self, max_weight):
        """Assigns a "terrain cost" (node weight) to every cell."""
        for y in range(self.height):
            for x in range(self.width):
                self.node_weights[(x, y)] = random.randint(1, max_weight)

    def _populate_rewards(self, num_rewards):
        """Places 'num_rewards' at random locations."""
        for _ in range(num_rewards):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in self.rewards: self.rewards.add((x, y)); break

    def get_valid_moves(self, x, y):
        """Returns the set of open directions ('N', 'S', 'E', 'W') for a cell."""
        return self.grid.get((x,y), set())

    def get_neighbors(self, x, y):
        """Returns a list of (nx, ny) coordinates reachable from (x, y)."""
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
        """Gets the cost of a move, respecting maze's weight type."""
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
    def __init__(self, maze, algorithm="A*", knows_maze=False): # Re-added A* default
        self.maze = maze
        self.algorithm = algorithm

        # Online-only algorithms cannot use the 'knows_maze' = True mode.
        if self.algorithm in ["Monte Carlo", "Depth-First Search", "Greedy BFS", "BFS"]: # Online-only
            self.knows_maze = False
        else:
            self.knows_maze = knows_maze
        self.is_solver = self.knows_maze # is_solver is an alias for "is offline"

        # Basic Robot State
        self.start_pos = (random.randint(0, maze.width - 1), random.randint(0, maze.height - 1))
        while self.start_pos in self.maze.rewards: self.start_pos = (random.randint(0, maze.width-1), random.randint(0, maze.height-1))
        self.x, self.y = self.start_pos
        self.goal_pos = self._get_distant_pos()
        while self.goal_pos in self.maze.rewards or self.goal_pos == self.start_pos: self.goal_pos = self._get_distant_pos()

        self.path = [(self.x, self.y)] # Robot's movement history
        self.unvisited_rewards = set(self.maze.rewards)
        self.search_area = set() # All nodes visited by the search algorithm
        self.is_done = False

        # State for "Offline" (Solver)
        self.final_path = []

        # State for "Online" (Explorer)
        self.backtrack_stack = []
        self.current_target = None
        self.online_pq = []       # Priority queue for Dijkstra/A*/Greedy BFS explorers
        self.online_queue = collections.deque() # Queue for BFS explorer
        self.online_cost_so_far = {} # g(n) cost for Dijkstra/A* explorers
        self.online_came_from = {}   # Parent pointers for BFS/Greedy BFS explorers
        self.backtrack_target = None # For step-by-step backtracking visual
        self.walk_path = collections.deque() # Physical animation path between expansion nodes
        self.solution_cost_accum = 0
        self.stage_start = None

        # Metrics
        self.metrics = {
            'steps': 0,
            'unique_explored': 0,
            'nodes_expanded': 0,
            'path_length': 0,
            'total_cost': 0,
            'frontier_max': 0,
            'algorithm_steps': 0
        }

        # Initialize based on mode
        if self.is_solver: self.solve_maze_offline()
        else: self._setup_online_search()

    def _build_parent_chain(self, node):
        """Returns the chain of nodes following parent pointers up to the root.

        This is used by the physical animation system to construct a walk
        between the agent's current position and the next expansion node using
        the lowest common ancestor (LCA) of their parent chains. This keeps
        the animation faithful to discovered connectivity without changing the
        algorithm's semantics.
        """
        chain = []
        cur = node
        parents = self.online_came_from
        # Ensure we don't infinite-loop; cap by maze size
        limit = self.maze.width * self.maze.height + 2
        while cur is not None and cur not in chain and limit > 0:
            chain.append(cur)
            if cur not in parents: break
            cur = parents[cur]
            limit -= 1
        return chain

    def _plan_walk_to(self, target):
        """Plan a physical path along parent pointers from current position to target.

        Strategy:
          1) Build parent chains for current position and the target.
          2) Find the LCA where these chains meet.
          3) Walk up from current to LCA, then down from LCA to target.

        The generated waypoints are stored in self.walk_path and consumed by
        step(), advancing one cell per tick.
        """
        if target == (self.x, self.y):
            return
        a_chain = self._build_parent_chain((self.x, self.y))      # current -> root
        b_chain = self._build_parent_chain(target)                # target -> root
        if not a_chain or not b_chain:
            return
        b_set = set(b_chain)
        lca = None
        for n in a_chain:
            if n in b_set:
                lca = n
                break
        if lca is None:
            return
        i_lca_a = a_chain.index(lca)
        i_lca_b = b_chain.index(lca)
        path_up = a_chain[1:i_lca_a+1]               # step from current towards LCA (skip current)
        path_down = list(reversed(b_chain[:i_lca_b])) # from LCA to target (skip LCA)
        walk_seq = path_up + path_down
        self.walk_path.clear()
        for node in walk_seq:
            self.walk_path.append(node)

    def _record_move(self, prev_pos, new_pos, is_algorithm_move=False):
        """Record a single visual move and optionally attribute algorithmic cost.

        - steps always increments for animation moves (visual fidelity).
        - When is_algorithm_move=True, we also add edge cost to total_cost.
          This is used by algorithms that truly "walk" to explore (DFS, Monte Carlo)
          and during offline path following when appropriate.
        - unique_explored reflects the size of the discovered set (search_area).
        """
        if prev_pos != new_pos:
            self.metrics['steps'] += 1
            self.metrics['path_length'] = max(0, len(self.path) - 1)
            if is_algorithm_move:
                self.metrics['total_cost'] += self.maze.get_cost(prev_pos, new_pos)
        self.metrics['unique_explored'] = len(self.search_area)

    def _bump_frontier_metric(self):
        size = 0
        if self.algorithm == 'BFS': size = len(self.online_queue)
        elif self.algorithm in ['A*', 'Greedy BFS']: size = len(self.online_pq)
        elif self.algorithm in ['Depth-First Search', 'Monte Carlo']: size = len(self.backtrack_stack)
        if size > self.metrics['frontier_max']:
            self.metrics['frontier_max'] = size

    def _get_distant_pos(self):
        """Finds a suitable goal position far from the start."""
        while True:
            gx, gy = random.randint(0, self.maze.width-1), random.randint(0, self.maze.height-1)
            dist = abs(self.x - gx) + abs(self.y - gy)
            if dist > (self.maze.width + self.maze.height) / 2: return (gx, gy)

    def _heuristic(self, a, b):
        """Manhattan distance heuristic, used by A* and Greedy BFS."""
        (x1, y1) = a; (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)

    def _build_mst(self, method):
        """Builds an MST over the maze graph using Prim or Kruskal. Returns (tree_adj, visited_nodes)."""
        # Collect all nodes
        nodes = [(x, y) for y in range(self.maze.height) for x in range(self.maze.width)]
        # Build edge list (undirected)
        edges = []
        for (x, y) in nodes:
            for (nx, ny) in self.maze.get_neighbors(x, y):
                if (x, y) < (nx, ny):
                    cost = self.maze.get_cost((x, y), (nx, ny))
                    edges.append((cost, (x, y), (nx, ny)))
        tree_adj = collections.defaultdict(list)
        visited_nodes = set()
        if method == 'Prim Solver':
            start = self.start_pos
            visited = {start}; visited_nodes.add(start)
            pq = []
            for cost, u, v in edges:
                if u == start and v not in visited: heapq.heappush(pq, (cost, u, v))
                elif v == start and u not in visited: heapq.heappush(pq, (cost, v, u))
            while pq and len(visited) < len(nodes):
                cost, u, v = heapq.heappop(pq)
                if v in visited: continue
                visited.add(v); visited_nodes.add(v)
                tree_adj[u].append(v); tree_adj[v].append(u)
                # push edges from v
                for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                    nx, ny = v[0]+dx, v[1]+dy
                    if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height and (nx, ny) not in visited and (nx, ny) in self.maze.get_neighbors(v[0], v[1]):
                        c = self.maze.get_cost(v, (nx, ny))
                        heapq.heappush(pq, (c, v, (nx, ny)))
        else: # Kruskal Solver
            parent = {}; rank = {}
            def find(a):
                parent.setdefault(a, a)
                if parent[a] != a: parent[a] = find(parent[a])
                return parent[a]
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb: return False
                rank.setdefault(ra, 0); rank.setdefault(rb, 0)
                if rank[ra] < rank[rb]: parent[ra] = rb
                elif rank[rb] < rank[ra]: parent[rb] = ra
                else: parent[rb] = ra; rank[ra] += 1
                return True
            for n in nodes: parent[n] = n; rank[n] = 0
            edges.sort(key=lambda e: e[0])
            added = 0; need = len(nodes) - 1
            for cost, u, v in edges:
                if union(u, v):
                    tree_adj[u].append(v); tree_adj[v].append(u)
                    visited_nodes.add(u); visited_nodes.add(v)
                    added += 1
                    if added >= need: break
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
        pq = []
        if self.algorithm == 'BFS': heapq.heappush(pq, (0, start))
        elif self.algorithm == 'A*': heapq.heappush(pq, (self._heuristic(start, end), start)) # A* uses heuristic
        else: return [], {start}

        came_from = {start: None}; cost_so_far = {start: 0}; visited_nodes = set()
        while pq:
            # A* pops based on f-cost (g+h), others pop based on g-cost or length
            if self.algorithm == 'A*': _, current = heapq.heappop(pq)
            else: current_cost, current = heapq.heappop(pq)

            visited_nodes.add(current)
            if current == end: break

            for neighbor in self.maze.get_neighbors(current[0], current[1]):
                # Calculate cost to reach neighbor
                if self.algorithm == 'BFS': new_cost = cost_so_far[current] + 1
                else: new_cost = cost_so_far[current] + self.maze.get_cost(current, neighbor) # Dijkstra/A* use maze cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    # Calculate priority for the queue
                    if self.algorithm == 'BFS': priority = new_cost
                    else: priority = new_cost + self._heuristic(neighbor, end) # A* priority = g + h
                    heapq.heappush(pq, (priority, neighbor)); came_from[neighbor] = current
        # Reconstruct path
        path = []; current = end
        if end in came_from:
            while current != start: path.append(current); current = came_from[current]
            path.append(start); path.reverse()
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
            # Build MST and path along the tree
            tree_adj, visited_nodes = self._build_mst(self.algorithm)
            self.search_area.update(visited_nodes)

            def tree_path(a, b):
                # BFS in tree to get unique path
                parent = {a: None}
                dq = collections.deque([a])
                while dq:
                    u = dq.popleft()
                    if u == b: break
                    for v in tree_adj.get(u, []):
                        if v not in parent:
                            parent[v] = u; dq.append(v)
                if b not in parent: return []
                path = [b]
                cur = b
                while cur != a:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path

            current_loc = self.start_pos; rewards_to_visit = set(self.maze.rewards); full_path = []
            while rewards_to_visit:
                best_path, best_target, min_cost = [], None, float('inf')
                for reward in rewards_to_visit:
                    p = tree_path(current_loc, reward)
                    if p:
                        path_cost = sum(self.maze.get_cost(p[i], p[i+1]) for i in range(len(p)-1))
                        if path_cost < min_cost: min_cost, best_path, best_target = path_cost, p, reward
                if best_target: full_path.extend(best_path[1:]); current_loc = best_target; rewards_to_visit.remove(best_target)
                else: break
            p_goal = tree_path(current_loc, self.goal_pos)
            if p_goal: full_path.extend(p_goal[1:])
            self.final_path = full_path
            # Offline solver metrics
            self.metrics['algorithm_steps'] = len(self.search_area)
            # Cost along final_path
            cost_sum = 0
            cur = self.start_pos
            for nxt in self.final_path:
                cost_sum += self.maze.get_cost(cur, nxt)
                cur = nxt
            self.metrics['total_cost'] = cost_sum
        else:
            current_loc = self.start_pos; rewards_to_visit = set(self.maze.rewards); full_path = []
            while rewards_to_visit: # Path through rewards first
                best_path, best_target, min_cost = [], None, float('inf')
                for reward in rewards_to_visit:
                    path, visited = self._run_offline_search(current_loc, reward); self.search_area.update(visited)
                    if path:
                        # Use appropriate cost metric (length for BFS, actual cost otherwise)
                        path_cost = len(path) if self.algorithm == 'BFS' else sum(self.maze.get_cost(path[i], path[i+1]) for i in range(len(path)-1))
                        if path_cost < min_cost: min_cost, best_path, best_target = path_cost, path, reward
                if best_target: full_path.extend(best_path[1:]); current_loc = best_target; rewards_to_visit.remove(best_target)
                else: break # Unreachable reward
            # Path from last reward (or start) to goal
            path_to_goal, visited = self._run_offline_search(current_loc, self.goal_pos); self.search_area.update(visited)
            if path_to_goal: full_path.extend(path_to_goal[1:])
            self.final_path = full_path
            self.metrics['algorithm_steps'] = len(self.search_area)
            cost_sum = 0
            cur = self.start_pos
            for nxt in self.final_path:
                cost_sum += self.maze.get_cost(cur, nxt)
                cur = nxt
            self.metrics['total_cost'] = cost_sum

    def _step_follow_path(self):
        """Step function for an OFFLINE solver (animates pre-computed path)."""
        if self.final_path:
            prev = (self.x, self.y)
            next_pos = self.final_path.pop(0); self.x, self.y = next_pos
            self.path.append((self.x, self.y))
            self._record_move(prev, next_pos, is_algorithm_move=False)
            if (self.x, self.y) in self.unvisited_rewards: self.unvisited_rewards.remove((self.x, self.y))
        if (self.x, self.y) == self.goal_pos and not self.unvisited_rewards: self.is_done = True
    # --- END OFFLINE LOGIC ---

    # --- ONLINE (EXPLORER) LOGIC ---
    def _update_online_target(self, from_pos):
        """Find the next target (closest remaining reward or the goal).

        Online explorers pursue a moving objective: the nearest unvisited reward
        (by Manhattan distance heuristic) until rewards are exhausted, then the
        final goal. This target is re-evaluated when a reward is reached.
        """
        if not self.unvisited_rewards: self.current_target = self.goal_pos
        else:
            min_dist = float('inf'); best_reward = None
            for reward in self.unvisited_rewards:
                dist = self._heuristic(from_pos, reward) # Manhattan distance heuristic
                if dist < min_dist: min_dist, best_reward = dist, reward
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
        self._update_online_target((self.x, self.y))
        start_pos = (self.x, self.y)
        self.stage_start = start_pos
        self.online_cost_so_far = {start_pos: 0} # g-cost (Dijkstra/A*)
        self.online_came_from = {start_pos: None} # Parent pointers (BFS/Greedy BFS)
        self.online_pq = [] # Priority Queue (Dijkstra/A*/Greedy BFS)
        self.online_queue = collections.deque() # Queue (BFS)
        # Mark start as visited for algorithms that use search_area as already-seen visualization
        # For Greedy BFS, do NOT pre-mark start; search_area acts as the closed set there
        if self.algorithm != 'Greedy BFS':
            self.search_area.add(start_pos)

        # Prime the relevant data structure
        if self.algorithm == 'BFS': self.online_queue.append(start_pos)
        elif self.algorithm == 'A*':
            priority = self._heuristic(start_pos, self.current_target) # f = g+h, g=0 at start
            heapq.heappush(self.online_pq, (priority, 0, start_pos)) # (f_cost, g_cost, pos)
        elif self.algorithm == 'Greedy BFS':
            priority = self._heuristic(start_pos, self.current_target) # f = h
            heapq.heappush(self.online_pq, (priority, start_pos)) # (h_cost, pos)
        # DFS, Random Walk, Monte Carlo use backtrack_stack

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
        if not self.online_queue: return
        # Peek target; walk there physically before expanding
        current = self.online_queue[0]
        if (self.x, self.y) != current:
            self._plan_walk_to(current); return
        current = self.online_queue.popleft()
        prev = (self.x, self.y)
        self.x, self.y = current
        self.path.append(current)
        self.metrics['nodes_expanded'] += 1
        self.metrics['algorithm_steps'] += 1
        self._record_move(prev, current, is_algorithm_move=False)
        if current == self.goal_pos and not self.unvisited_rewards:
            # Compute cost along discovered path from start to goal using parents
            cost_sum = 0
            node = current
            while node is not None and node in self.online_came_from:
                parent = self.online_came_from[node]
                if parent is None: break
                cost_sum += self.maze.get_cost(parent, node)
                node = parent
            self.metrics['total_cost'] = cost_sum
            self.is_done = True; return
        for neighbor in self.maze.get_neighbors(current[0], current[1]):
            if neighbor not in self.online_came_from:
                self.online_came_from[neighbor] = current; self.search_area.add(neighbor)
                self.online_queue.append(neighbor)
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
        if not self.online_pq: return
        # Clean out stale entries at the top
        while self.online_pq and self.online_pq[0][2] in self.online_cost_so_far and self.online_pq[0][1] > self.online_cost_so_far[self.online_pq[0][2]]:
            heapq.heappop(self.online_pq)
        if not self.online_pq: return
        # Peek; physically walk to top before expanding
        peek = self.online_pq[0]
        current = peek[2]
        if (self.x, self.y) != current:
            self._plan_walk_to(current); return
        priority, cost, current = heapq.heappop(self.online_pq) # Pop based on f-cost (g+h)
        if current in self.online_cost_so_far and cost > self.online_cost_so_far[current]: return # Stale
        prev = (self.x, self.y)
        self.x, self.y = current; self.path.append(current)
        self.metrics['nodes_expanded'] += 1
        self.metrics['algorithm_steps'] += 1
        self._record_move(prev, current, is_algorithm_move=False)

        if current == self.current_target: # Reached reward or goal
            if current in self.unvisited_rewards: self.unvisited_rewards.remove(current)
            if not self.unvisited_rewards and current == self.goal_pos:
                self.solution_cost_accum += cost
                self.metrics['total_cost'] = self.solution_cost_accum
                self.is_done = True; return
            # Found reward, reset search for next target
            self._update_online_target(current)
            self.online_cost_so_far = {current: 0}; self.online_came_from = {current: None}; self.online_pq = []
            new_priority = self._heuristic(current, self.current_target) # f = g+h, g=0
            heapq.heappush(self.online_pq, (new_priority, 0, current))
            # Accumulate segment cost and reset stage start
            self.solution_cost_accum += cost
            self.stage_start = current
            self._bump_frontier_metric(); return # End step

        # Continue search
        for neighbor in self.maze.get_neighbors(current[0], current[1]):
            new_cost = cost + self.maze.get_cost(current, neighbor) # New g-cost
            if neighbor not in self.online_cost_so_far or new_cost < self.online_cost_so_far[neighbor]:
                self.online_cost_so_far[neighbor] = new_cost; self.search_area.add(neighbor)
                self.online_came_from[neighbor] = current
                new_priority = new_cost + self._heuristic(neighbor, self.current_target) # f = new_g + h
                heapq.heappush(self.online_pq, (new_priority, new_cost, neighbor)) # Add with f-cost priority
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
                self.metrics['total_cost'] = self.solution_cost_accum
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
        self.search_area.add((self.x, self.y))
        if (self.x, self.y) == self.current_target:
            if (self.x, self.y) in self.unvisited_rewards: self.unvisited_rewards.remove((self.x, self.y))
            if not self.unvisited_rewards and (self.x, self.y) == self.goal_pos: self.is_done = True; return
            self._update_online_target((self.x, self.y)); self.backtrack_stack.clear()

        unvisited_neighbors = []
        directions = ['N', 'S', 'W', 'E'];
        if randomize: random.shuffle(directions)
        for move in directions:
            if move in self.maze.get_valid_moves(self.x, self.y):
                nx, ny = self.x, self.y
                if move == 'N': ny -= 1; 
                elif move == 'S': ny += 1
                elif move == 'W': nx -= 1; 
                elif move == 'E': nx += 1
                if (nx, ny) not in self.search_area: unvisited_neighbors.append((nx, ny))

        if unvisited_neighbors:
            nx, ny = unvisited_neighbors[0]
            prev = (self.x, self.y)
            self.backtrack_stack.append((self.x, self.y)); self.x, self.y = nx, ny
            self.path.append((self.x, self.y))
            self.metrics['nodes_expanded'] += 1
            self.metrics['algorithm_steps'] += 1
            self._record_move(prev, (self.x, self.y), is_algorithm_move=True)
        elif self.backtrack_stack:
            self.backtrack_target = self.backtrack_stack.pop() # Initiate backtrack
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
        self.search_area.add((self.x, self.y))
        if (self.x, self.y) == self.current_target:
            if (self.x, self.y) in self.unvisited_rewards: self.unvisited_rewards.remove((self.x, self.y))
            if not self.unvisited_rewards and (self.x, self.y) == self.goal_pos: self.is_done = True; return
            self._update_online_target((self.x, self.y)); self.backtrack_stack.clear()

        neighbor_scores = []; valid_unvisited_neighbors = []
        for neighbor in self.maze.get_neighbors(self.x, self.y):
            if neighbor not in self.search_area: valid_unvisited_neighbors.append(neighbor)

        if valid_unvisited_neighbors:
            for neighbor in valid_unvisited_neighbors:
                best_rollout_dist = float('inf')
                for _ in range(MONTE_CARLO_ROLLOUTS):
                    sim_x, sim_y = neighbor
                    min_dist_rollout = self._heuristic((sim_x, sim_y), self.current_target)
                    for _ in range(MONTE_CARLO_DEPTH):
                        sim_neighbors = self.maze.get_neighbors(sim_x, sim_y)
                        if not sim_neighbors: break
                        sim_x, sim_y = random.choice(sim_neighbors)
                        dist = self._heuristic((sim_x, sim_y), self.current_target)
                        min_dist_rollout = min(dist, min_dist_rollout)
                    best_rollout_dist = min(best_rollout_dist, min_dist_rollout)
                heapq.heappush(neighbor_scores, (best_rollout_dist, neighbor))
            if neighbor_scores:
                _, (nx, ny) = heapq.heappop(neighbor_scores)
                prev = (self.x, self.y)
                self.backtrack_stack.append((self.x, self.y)); self.x, self.y = nx, ny
                self.path.append((self.x, self.y))
                self.metrics['nodes_expanded'] += 1
                self.metrics['algorithm_steps'] += 1
                self._record_move(prev, (self.x, self.y), is_algorithm_move=True)
                self._bump_frontier_metric(); return # End step

        if self.backtrack_stack:
            self.backtrack_target = self.backtrack_stack.pop() # Initiate backtrack
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
        if self.is_done: return

        # Handle planned physical walk moves first (one edge per tick)
        if self.walk_path:
            prev = (self.x, self.y)
            nxt = self.walk_path.popleft()
            self.x, self.y = nxt
            self.path.append((self.x, self.y))
            self._record_move(prev, nxt, is_algorithm_move=False)
            return

        # Handle Step-by-Step Backtracking first
        if self.backtrack_target:
            prev = (self.x, self.y)
            self.x, self.y = self.backtrack_target
            self.path.append((self.x, self.y))
            self._record_move(prev, (self.x, self.y))
            # Count algorithmic step for algorithms that truly walk back (DFS/Monte Carlo)
            if self.algorithm in ["Depth-First Search", "Monte Carlo"]:
                self.metrics['algorithm_steps'] += 1
            self.backtrack_target = None
            return # End step after backtrack move

        # If not backtracking, proceed with normal algorithm logic
        if self.is_solver:
            self._step_follow_path() # Offline solver just follows path
        else:
            # Online explorer takes one step of its specific algorithm
            if self.algorithm == "Random Walk": self._step_explore_dfs(randomize=True)
            elif self.algorithm == "Depth-First Search": self._step_explore_dfs(randomize=False)
            elif self.algorithm == "BFS": self._step_online_bfs()
            elif self.algorithm == "A*": self._step_online_a_star()
            elif self.algorithm == "Greedy BFS": self._step_online_greedy_bfs() # Added Greedy BFS
            elif self.algorithm == "Monte Carlo": self._step_monte_carlo()


class MazeApp:
    """Main application class for the Tkinter UI and simulation loop."""
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Algorithm Visualizer")
        self.root.configure(bg=BG_COLOR)

        # UI Variables
        self.mode = tk.StringVar(value="robot_vs_maze")
        self.speed = tk.IntVar(value=150)
        self.algorithm_var = tk.StringVar(value="A*") # Re-added A* as default
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
        ttk.Button(control_frame_top, text="New Simulation", command=self.start_new_simulation).pack(side=tk.LEFT, padx=5)
        self.knows_maze_check = ttk.Checkbutton(control_frame_top, text="Knows Full Maze?", variable=self.knows_maze_var, command=self.on_knows_maze_change); self.knows_maze_check.pack(side=tk.LEFT, padx=10)
        ttk.Label(control_frame_top, text="Algorithm:").pack(side=tk.LEFT, padx=(10, 5))
        # Updated algorithm list
        self.algo_combo = ttk.Combobox(control_frame_top, textvariable=self.algorithm_var, values=["A*", "Greedy BFS", "BFS", "Depth-First Search", "Monte Carlo"], state="readonly", width=18); self.algo_combo.pack(side=tk.LEFT, padx=5); self.algo_combo.bind("<<ComboboxSelected>>", self.on_algo_change)
        ttk.Label(control_frame_top, text="Mode:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(control_frame_top, text="Robot vs Maze", variable=self.mode, value="robot_vs_maze", command=self.start_new_simulation).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame_top, text="Robot vs Player", variable=self.mode, value="robot_vs_player", command=self.start_new_simulation).pack(side=tk.LEFT)
        ttk.Label(control_frame_top, text="Speed:").pack(side=tk.LEFT, padx=(10, 5)); ttk.Scale(control_frame_top, from_=1, to=200, orient=tk.HORIZONTAL, variable=self.speed).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Bottom Control Row
        ttk.Label(control_frame_bottom, text="Generator:").pack(side=tk.LEFT, padx=5)
        # Updated generator list
        self.gen_combo = ttk.Combobox(control_frame_bottom, textvariable=self.gen_method_var, values=["DFS", "BFS", "Greedy Frontier", "Random Prim's", "Weighted Prim's (MST)", "Kruskal's MST"], state="readonly", width=22); self.gen_combo.pack(side=tk.LEFT, padx=5); self.gen_combo.bind("<<ComboboxSelected>>", self.on_generator_change)
        ttk.Label(control_frame_bottom, text="Loop %:").pack(side=tk.LEFT, padx=(10, 5)); self.loop_spinbox = ttk.Spinbox(control_frame_bottom, from_=0, to=100, increment=5, textvariable=self.loop_percent_var, width=5, command=self.start_new_simulation); self.loop_spinbox.pack(side=tk.LEFT)
        ttk.Label(control_frame_bottom, text="Rewards:").pack(side=tk.LEFT, padx=(10, 5)); self.reward_spinbox = ttk.Spinbox(control_frame_bottom, from_=0, to=10, textvariable=self.num_rewards_var, width=5, command=self.start_new_simulation); self.reward_spinbox.pack(side=tk.LEFT)
        ttk.Label(control_frame_bottom, text="Max Weight:").pack(side=tk.LEFT, padx=(10, 5)); self.max_weight_spinbox = ttk.Spinbox(control_frame_bottom, from_=1, to=10, textvariable=self.max_weight_var, width=5, command=self.start_new_simulation); self.max_weight_spinbox.pack(side=tk.LEFT)
        # Metrics label (right side)
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(control_frame_bottom, textvariable=self.metrics_var).pack(side=tk.RIGHT)

        # Canvases
        canvas_size_w = CELL_SIZE * MAZE_WIDTH; canvas_size_h = CELL_SIZE * MAZE_HEIGHT
        self.left_canvas = tk.Canvas(maze_frame, width=canvas_size_w, height=canvas_size_h, bg=BG_COLOR, highlightthickness=0); self.left_canvas.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        self.right_canvas = tk.Canvas(maze_frame, width=canvas_size_w, height=canvas_size_h, bg=BG_COLOR, highlightthickness=0); self.right_canvas.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

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
        # Generator controls: Disable Max Weight and Loop % for MST generators
        is_mst_gen = self.gen_method_var.get() in ["Weighted Prim's (MST)", "Kruskal's MST"]
        self.max_weight_spinbox.configure(state='disabled' if is_mst_gen else 'normal')
        self.loop_spinbox.configure(state='disabled' if is_mst_gen else 'normal')

        # Algorithm controls: Disable "Knows Maze" for online-only algorithms
        online_only_algos = ["Monte Carlo", "Depth-First Search", "Greedy BFS", "BFS"]
        is_online_only = self.algorithm_var.get() in online_only_algos

        if is_online_only:
            self.knows_maze_var.set(False); self.knows_maze_check.configure(state='disabled')
        else: # A*, Dijkstra, BFS can be either
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
        """Resets maze, robots, and starts a new simulation."""
        self.is_running = False
        loop_percent = self.loop_percent_var.get()
        if self.gen_method_var.get() in ["Weighted Prim's (MST)", "Kruskal's MST"]: loop_percent = 0

        self.maze = Maze(MAZE_WIDTH, MAZE_HEIGHT, self.gen_method_var.get(), loop_percent, self.num_rewards_var.get(), self.max_weight_var.get())
        self.robot = Robot(self.maze, self.algorithm_var.get(), self.knows_maze_var.get())
        self.player = None
        if self.mode.get() == "robot_vs_player":
            self.player = Robot(self.maze, algorithm="Depth-First Search", knows_maze=False); self.player.color = PLAYER_COLOR
            self.player.goal_pos = self.robot.goal_pos; self.player.unvisited_rewards = set(self.maze.rewards)
            while True:
                self.player.start_pos = (random.randint(0,self.maze.width-1), random.randint(0,self.maze.height-1))
                if self.player.start_pos!=self.robot.start_pos and self.player.start_pos not in self.maze.rewards: break
            self.player.x,self.player.y = self.player.start_pos; self.player.path = [self.player.start_pos]
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
        """Main simulation loop tick."""
        if not self.is_running: return
        if self.mode.get() == "robot_vs_maze":
            self.robot.step(); self.draw_right_maze(); self._update_metrics_label()
            if self.robot.is_done: self.is_running = False; self.draw_winner_message("Robot Finished!")
        elif self.mode.get() == "robot_vs_player":
            self.robot.step(); self.draw_left_maze(); self._update_metrics_label(); self.check_winner()
        delay = 201 - self.speed.get(); self.root.after(delay, self.update_loop)

    def _update_metrics_label(self):
        m = self.robot.metrics
        text = f"Steps: {m['steps']}  |  Explored: {m['unique_explored']}  |  Expanded: {m['nodes_expanded']}  |  Path: {m['path_length']}  |  Cost: {m['total_cost']}  |  Frontier max: {m['frontier_max']}"
        self.metrics_var.set(text)

    def check_winner(self):
        if self.robot.is_done: self.is_running = False; self.draw_winner_message("Robot Wins!")
        elif self.player and self.player.is_done: self.is_running = False; self.draw_winner_message("Player Wins!")

    def move_player(self, direction):
        """Handle player movement via keypress."""
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
    def draw_left_maze(self): self._draw_maze_on_canvas(self.left_canvas, self.maze, False, self.robot)
    def draw_right_maze(self): agent = self.robot if self.mode.get()=="robot_vs_maze" else self.player; self._draw_maze_on_canvas(self.right_canvas, self.maze, False, agent)

    def _draw_maze_on_canvas(self, canvas, maze, show_full_maze, agent):
        """Main rendering function."""
        canvas.delete("all"); c_width=canvas.winfo_width(); c_height=canvas.winfo_height()
        if c_width<=1 or c_height<=1: canvas.after(50, lambda: self._draw_maze_on_canvas(canvas,maze,show_full_maze,agent)); return
        cell_w=c_width/self.maze.width; cell_h=c_height/self.maze.height; wall_width=max(2,int(min(cell_w,cell_h)/8))
        for y in range(maze.height):
            for x in range(maze.width):
                x1,y1,x2,y2 = x*cell_w, y*cell_h, (x+1)*cell_w, (y+1)*cell_h
                is_known = show_full_maze or agent.is_solver or (x,y) in agent.search_area
                if is_known:
                    if maze.weight_type=='node' and maze.max_weight>1: fill_color = self._get_weight_color(maze.node_weights.get((x,y),1), maze.max_weight)
                    else: fill_color = DEFAULT_CELL_COLOR
                    if (x,y) in agent.search_area: fill_color = SEARCH_AREA_COLOR
                    if agent.is_solver and (x,y) in agent.final_path: fill_color = FINAL_PATH_COLOR
                    elif not agent.is_solver and (x,y) in agent.path: fill_color = KNOWN_PATH_COLOR
                    canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="")
                    open_walls = maze.get_valid_moves(x, y)
                    if 'N' not in open_walls: canvas.create_line(x1, y1, x2, y1, fill=WALL_COLOR, width=wall_width)
                    if 'S' not in open_walls: canvas.create_line(x1, y2, x2, y2, fill=WALL_COLOR, width=wall_width)
                    if 'W' not in open_walls: canvas.create_line(x1, y1, x1, y2, fill=WALL_COLOR, width=wall_width)
                    if 'E' not in open_walls: canvas.create_line(x2, y1, x2, y2, fill=WALL_COLOR, width=wall_width)
                elif not show_full_maze: canvas.create_rectangle(x1, y1, x2, y2, fill=BG_COLOR, outline="")
        # Draw edge weights on full view for edge-weighted mazes
        if show_full_maze and maze.weight_type == 'edge':
            drawn = set()
            for (a, b), weight in maze.edge_weights.items():
                if (b, a) in drawn: continue
                (x, y), (nx, ny) = a, b
                # Only draw for adjacent cells
                if abs(x - nx) + abs(y - ny) != 1: continue
                cx = (x + nx) / 2.0
                cy = (y + ny) / 2.0
                px = cx * cell_w + cell_w / 2.0
                py = cy * cell_h + cell_h / 2.0
                canvas.create_text(px, py, text=str(weight), fill="white")
                drawn.add((a, b))
        for (rx,ry) in maze.rewards:
            if (rx,ry) in agent.unvisited_rewards: self._draw_marker(canvas,(rx,ry),REWARD_COLOR,shape='oval')
            elif (rx,ry) in agent.search_area or show_full_maze or agent.is_solver: self._draw_marker(canvas,(rx,ry),SEARCH_AREA_COLOR,shape='oval')
        self._draw_marker(canvas,self.robot.start_pos,START_COLOR); self._draw_marker(canvas,self.robot.goal_pos,GOAL_COLOR)
        if self.player: self._draw_marker(canvas, self.player.start_pos, START_COLOR)
        agent_color = ROBOT_COLOR if agent==self.robot else PLAYER_COLOR
        self._draw_marker(canvas,(agent.x,agent.y),agent_color,is_agent=True)

    def _draw_marker(self, canvas, pos, color, is_agent=False, shape='rect'):
        c_width=canvas.winfo_width(); c_height=canvas.winfo_height(); cell_w=c_width/self.maze.width; cell_h=c_height/self.maze.height
        x, y = pos; margin_frac = 0.2 if not is_agent else 0.25; mx, my = cell_w*margin_frac, cell_h*margin_frac
        x1, y1, x2, y2 = x*cell_w+mx, y*cell_h+my, (x+1)*cell_w-mx, (y+1)*cell_h-my
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
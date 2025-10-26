import tkinter as tk
from tkinter import ttk
import random, heapq, collections

# --- Configuration ---
CELL_SIZE = 20
MAZE_WIDTH = 25
MAZE_HEIGHT = 25

# --- Color Scheme (same as your base) ---
BG_COLOR = "#2c3e50"
WALL_COLOR = "#bdc3c7"
ROBOT_COLOR = "#f1c40f"
PLAYER_COLOR = "#3498db"
START_COLOR = "#1abc9c"
GOAL_COLOR = "#e74c3c"
PATH_COLOR = "#95a5a6"
KNOWN_PATH_COLOR = "#7f8c8d"

class Maze:
    """
    Generates a maze using weighted Prim’s algorithm (Minimum Spanning Tree).
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = collections.defaultdict(set)
        self.weights = {}
        self.generate()

    def generate(self):
        visited = set()
        start = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        visited.add(start)
        frontier = []

        def add_frontier_edges(cx, cy):
            for dx, dy, direction, opposite in [(0,-1,'N','S'),(0,1,'S','N'),(-1,0,'W','E'),(1,0,'E','W')]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    # weighted edges based on random + distance
                    dist = ((nx - start[0])**2 + (ny - start[1])**2)**0.5
                    w = random.randint(1, 10) + 0.5 * dist
                    self.weights[((cx, cy), (nx, ny))] = w
                    heapq.heappush(frontier, (w, (cx, cy), (nx, ny), direction, opposite))

        add_frontier_edges(*start)

        while frontier:
            w, cell, neighbor, direction, opposite = heapq.heappop(frontier)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            cx, cy = cell
            nx, ny = neighbor
            self.grid[(cx, cy)].add(direction)
            self.grid[(nx, ny)].add(opposite)
            add_frontier_edges(nx, ny)

    def get_valid_moves(self, x, y):
        return self.grid.get((x, y), set())

    def get_edge_weight(self, a, b):
        return self.weights.get((a, b)) or self.weights.get((b, a)) or 1


class Robot:
    """
    Robot that solves the maze using weighted Prim’s/Dijkstra’s algorithm.
    """
    def __init__(self, maze):
        self.maze = maze
        self.x = random.randint(0, maze.width - 1)
        self.y = random.randint(0, maze.height - 1)
        self.start_pos = (self.x, self.y)
        self.goal_pos = self._get_distant_pos()
        self.path = [(self.x, self.y)]
        self.known_maze = collections.defaultdict(set)
        self.frontier = [(0, self.start_pos)]
        self.cost_so_far = {self.start_pos: 0}
        self.parents = {}
        self.visited = set()
        self.solution = []
        self.solved = False

    def _get_distant_pos(self):
        while True:
            gx, gy = random.randint(0, self.maze.width - 1), random.randint(0, self.maze.height - 1)
            if abs(gx - self.x) + abs(gy - self.y) > (self.maze.width + self.maze.height)//2:
                return (gx, gy)

    def step(self):
        if self.solved:
            return
        if not self.frontier:
            return

        cost, current = heapq.heappop(self.frontier)
        if current in self.visited:
            return
        self.visited.add(current)
        self.x, self.y = current
        self.known_maze[current] = self.maze.get_valid_moves(*current)

        if current == self.goal_pos:
            self._reconstruct_path()
            self.solved = True
            return

        x, y = current
        for move in self.maze.get_valid_moves(x, y):
            nx, ny = x, y
            if move == 'N': ny -= 1
            elif move == 'S': ny += 1
            elif move == 'W': nx -= 1
            elif move == 'E': nx += 1
            edge_cost = self.maze.get_edge_weight((x, y), (nx, ny))
            new_cost = self.cost_so_far[(x, y)] + edge_cost
            if (nx, ny) not in self.cost_so_far or new_cost < self.cost_so_far[(nx, ny)]:
                self.cost_so_far[(nx, ny)] = new_cost
                self.parents[(nx, ny)] = (x, y)
                heapq.heappush(self.frontier, (new_cost, (nx, ny)))

    def _reconstruct_path(self):
        node = self.goal_pos
        while node != self.start_pos:
            self.solution.append(node)
            node = self.parents.get(node, self.start_pos)
        self.solution.append(self.start_pos)
        self.solution.reverse()
        self.path.extend(self.solution)


class MazeApp:
    """
    Same UI and colors as your base, with Prim’s generation and solver logic.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Prim's Algorithm Maze Visualizer")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        self.mode = tk.StringVar(value="robot_vs_maze")
        self.speed = tk.IntVar(value=50)
        self.is_running = False

        self._setup_ui()
        self.start_new_simulation()

    def _setup_ui(self):
        control_frame = tk.Frame(self.root, bg=BG_COLOR, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        maze_frame = tk.Frame(self.root, bg=BG_COLOR, padx=10, pady=10)
        maze_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Style().configure("TButton", padding=6, relief="flat", background="#34495e", foreground="white")
        ttk.Style().map("TButton", background=[('active', '#4a627a')])
        ttk.Style().configure("TRadiobutton", background=BG_COLOR, foreground="white")
        ttk.Style().configure("TScale", background=BG_COLOR)

        tk.Label(control_frame, text="Mode:", bg=BG_COLOR, fg="white").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control_frame, text="Robot vs Maze", variable=self.mode, value="robot_vs_maze", command=self.start_new_simulation).pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Robot vs Player", variable=self.mode, value="robot_vs_player", command=self.start_new_simulation).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="New Maze", command=self.start_new_simulation).pack(side=tk.LEFT, padx=20)
        tk.Label(control_frame, text="Speed:", bg=BG_COLOR, fg="white").pack(side=tk.LEFT, padx=5)
        ttk.Scale(control_frame, from_=1, to=200, orient=tk.HORIZONTAL, variable=self.speed).pack(side=tk.LEFT, fill=tk.X, expand=True)

        canvas_size = CELL_SIZE * MAZE_WIDTH
        self.left_canvas = tk.Canvas(maze_frame, width=canvas_size, height=canvas_size, bg=BG_COLOR, highlightthickness=0)
        self.left_canvas.pack(side=tk.LEFT, padx=10)
        self.right_canvas = tk.Canvas(maze_frame, width=canvas_size, height=canvas_size, bg=BG_COLOR, highlightthickness=0)
        self.right_canvas.pack(side=tk.RIGHT, padx=10)

    def start_new_simulation(self):
        self.is_running = False
        self.maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
        self.robot = Robot(self.maze)
        self.draw_all()
        self.is_running = True
        self.update_loop()

    def update_loop(self):
        if not self.is_running:
            return
        self.robot.step()
        self.draw_all()
        delay = 201 - self.speed.get()
        self.root.after(delay, self.update_loop)

    def draw_all(self):
        self._draw_maze_on_canvas(self.left_canvas, self.maze, show_full_maze=True, agent=self.robot)
        self._draw_maze_on_canvas(self.right_canvas, self.maze, show_full_maze=False, agent=self.robot)

    def _draw_maze_on_canvas(self, canvas, maze, show_full_maze, agent):
        canvas.delete("all")
        for y in range(maze.height):
            for x in range(maze.width):
                x1, y1 = x * CELL_SIZE, y * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE

                is_known = (x, y) in agent.known_maze or (x, y) in agent.path
                if show_full_maze or is_known:
                    path_color = BG_COLOR
                    if (x, y) in agent.path:
                        path_color = PATH_COLOR if show_full_maze else KNOWN_PATH_COLOR
                    canvas.create_rectangle(x1, y1, x2, y2, fill=path_color, outline="")
                    open_walls = maze.get_valid_moves(x, y)
                    if 'N' not in open_walls:
                        canvas.create_line(x1, y1, x2, y1, fill=WALL_COLOR, width=5)
                    if 'S' not in open_walls:
                        canvas.create_line(x1, y2, x2, y2, fill=WALL_COLOR, width=5)
                    if 'W' not in open_walls:
                        canvas.create_line(x1, y1, x1, y2, fill=WALL_COLOR, width=5)
                    if 'E' not in open_walls:
                        canvas.create_line(x2, y1, x2, y2, fill=WALL_COLOR, width=5)

        self._draw_marker(canvas, agent.start_pos, START_COLOR)
        self._draw_marker(canvas, agent.goal_pos, GOAL_COLOR)
        self._draw_marker(canvas, (agent.x, agent.y), ROBOT_COLOR, is_agent=True)

    def _draw_marker(self, canvas, pos, color, is_agent=False):
        x, y = pos
        margin = 2 if not is_agent else 3
        x1, y1 = x * CELL_SIZE + margin, y * CELL_SIZE + margin
        x2, y2 = (x + 1) * CELL_SIZE - margin, (y + 1) * CELL_SIZE - margin
        canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()

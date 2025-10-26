import tkinter as tk
from tkinter import ttk
import random
import collections

# --- Configuration ---
CELL_SIZE = 20
MAZE_WIDTH = 25
MAZE_HEIGHT = 25

# --- Color Scheme ---
BG_COLOR = "#2c3e50"
WALL_COLOR = "#bdc3c7" # Changed from #ecf0f1 for better contrast
ROBOT_COLOR = "#f1c40f"
PLAYER_COLOR = "#3498db"
START_COLOR = "#1abc9c"
GOAL_COLOR = "#e74c3c"
PATH_COLOR = "#95a5a6"
KNOWN_PATH_COLOR = "#7f8c8d"

class Maze:
    """
    Generates and stores the maze structure.
    Uses a randomized Depth-First Search algorithm.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Grid stores which walls are open for each cell
        # e.g., self.grid[(x,y)] = {'N', 'E'} means North and East walls are open
        self.grid = collections.defaultdict(set)
        self.generate()

    def generate(self):
        visited = set()
        stack = [(random.randint(0, self.width - 1), random.randint(0, self.height - 1))]
        visited.add(stack[0])

        while stack:
            current_x, current_y = stack[-1]
            neighbors = []

            # Check potential neighbors
            for dx, dy, direction, opposite in [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'),
                                               (-1, 0, 'W', 'E'), (1, 0, 'E', 'W')]:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, direction, opposite))
            
            if neighbors:
                nx, ny, direction, opposite = random.choice(neighbors)
                
                # Knock down walls between current cell and neighbor
                self.grid[(current_x, current_y)].add(direction)
                self.grid[(nx, ny)].add(opposite)

                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop() # Backtrack

    def get_valid_moves(self, x, y):
        return self.grid.get((x,y), set())

class Robot:
    """
    Represents the AI agent that navigates the maze.
    This class is the framework for your pathfinding algorithms.
    """
    def __init__(self, maze):
        self.maze = maze
        self.x = random.randint(0, maze.width - 1)
        self.y = random.randint(0, maze.height - 1)
        self.start_pos = (self.x, self.y)
        self.goal_pos = self._get_distant_pos()
        
        # --- Algorithm State ---
        # This is where you would store data for your algorithm
        self.path = [(self.x, self.y)]
        self.known_maze = collections.defaultdict(set) # Robot's memory of the maze
        self.known_maze[(self.x, self.y)] = self.maze.get_valid_moves(self.x, self.y)
        self.backtrack_stack = []

    def _get_distant_pos(self):
        """Find a suitable goal position far from the start."""
        while True:
            gx, gy = random.randint(0, self.maze.width - 1), random.randint(0, self.maze.height - 1)
            dist = abs(self.x - gx) + abs(self.y - gy)
            if dist > (self.maze.width + self.maze.height) / 2:
                return (gx, gy)

    def step(self):
        """
        *** ALGORITHM FRAMEWORK METHOD ***
        Implement your pathfinding algorithm here.
        This method is called on each "tick" of the simulation.
        It should decide the robot's next move based on its current state and knowledge.

        To implement your own algorithm (e.g., A*, BFS, DFS):
        1.  Modify the `__init__` method to set up any required data structures
            (e.g., a queue for BFS, a priority queue for A*).
        2.  Replace the logic below with your algorithm's logic for choosing the next
            cell to visit.
        3.  The method must update `self.x`, `self.y`, `self.path`, and `self.known_maze`.
        
        Current Placeholder Logic: A simple random walk with backtracking.
        """
        if (self.x, self.y) == self.goal_pos:
            return # Reached goal

        valid_moves = self.maze.get_valid_moves(self.x, self.y)
        self.known_maze[(self.x, self.y)] = valid_moves # Update knowledge

        # Find unvisited neighbors based on robot's path history
        unvisited_neighbors = []
        for move in valid_moves:
            nx, ny = self.x, self.y
            if move == 'N': ny -= 1
            elif move == 'S': ny += 1
            elif move == 'W': nx -= 1
            elif move == 'E': nx += 1

            if (nx, ny) not in self.path:
                unvisited_neighbors.append((nx, ny, move))
        
        if unvisited_neighbors:
            # Choose a random unvisited neighbor to move to
            nx, ny, move = random.choice(unvisited_neighbors)
            self.backtrack_stack.append((self.x, self.y)) # Store current pos for backtracking
            self.x, self.y = nx, ny
            self.path.append((self.x, self.y))
        elif self.backtrack_stack:
            # No unvisited neighbors, so backtrack
            self.x, self.y = self.backtrack_stack.pop()
            self.path.append((self.x, self.y))

class MazeApp:
    """
    The main application class that handles the UI and simulation loop.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Algorithm Visualizer")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        self.mode = tk.StringVar(value="robot_vs_maze")
        self.speed = tk.IntVar(value=50)
        self.is_running = False

        self._setup_ui()
        self.start_new_simulation()

    def _setup_ui(self):
        # --- Main Frames ---
        control_frame = tk.Frame(self.root, bg=BG_COLOR, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        maze_frame = tk.Frame(self.root, bg=BG_COLOR, padx=10, pady=10)
        maze_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Controls ---
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

        # --- Canvases ---
        canvas_size = CELL_SIZE * MAZE_WIDTH
        self.left_canvas = tk.Canvas(maze_frame, width=canvas_size, height=canvas_size, bg=BG_COLOR, highlightthickness=0)
        self.left_canvas.pack(side=tk.LEFT, padx=10)

        self.right_canvas = tk.Canvas(maze_frame, width=canvas_size, height=canvas_size, bg=BG_COLOR, highlightthickness=0)
        self.right_canvas.pack(side=tk.RIGHT, padx=10)
        
        # --- Player Controls ---
        self.root.bind("<KeyPress-Up>", lambda e: self.move_player('N'))
        self.root.bind("<KeyPress-Down>", lambda e: self.move_player('S'))
        self.root.bind("<KeyPress-Left>", lambda e: self.move_player('W'))
        self.root.bind("<KeyPress-Right>", lambda e: self.move_player('E'))

    def start_new_simulation(self):
        self.is_running = False
        
        self.maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
        self.robot = Robot(self.maze)
        
        if self.mode.get() == "robot_vs_player":
            self.player = Robot(self.maze) # Treat player like a robot for position/goal data
            self.player.color = PLAYER_COLOR

        self.draw_all()
        self.is_running = True
        self.update_loop()

    def update_loop(self):
        if not self.is_running:
            return

        if self.mode.get() == "robot_vs_maze":
            self.robot.step()
            self.draw_right_maze() # Only redraw the robot's progress
        elif self.mode.get() == "robot_vs_player":
            self.robot.step()
            self.draw_left_maze() # Redraw robot progress on left
            self.check_winner()

        # Schedule next update
        delay = 201 - self.speed.get()
        self.root.after(delay, self.update_loop)

    def check_winner(self):
        if (self.robot.x, self.robot.y) == self.robot.goal_pos:
            self.is_running = False
            self.draw_winner_message("Robot Wins!")
        elif (self.player.x, self.player.y) == self.player.goal_pos:
            self.is_running = False
            self.draw_winner_message("Player Wins!")

    def move_player(self, direction):
        if self.mode.get() == "robot_vs_player" and self.is_running:
            if direction in self.maze.get_valid_moves(self.player.x, self.player.y):
                if direction == 'N': self.player.y -= 1
                elif direction == 'S': self.player.y += 1
                elif direction == 'W': self.player.x -= 1
                elif direction == 'E': self.player.x += 1
                self.player.path.append((self.player.x, self.player.y))
                # Add knowledge of the new cell so it can be drawn
                self.player.known_maze[(self.player.x, self.player.y)] = self.maze.get_valid_moves(self.player.x, self.player.y)
                self.draw_right_maze() # Redraw player progress
                self.check_winner()

    def draw_all(self):
        if self.mode.get() == "robot_vs_maze":
            self._draw_maze_on_canvas(self.left_canvas, self.maze, show_full_maze=True, agent=self.robot)
            self._draw_maze_on_canvas(self.right_canvas, self.maze, show_full_maze=False, agent=self.robot)
        else: # robot_vs_player
            self.draw_left_maze()
            self.draw_right_maze()

    def draw_left_maze(self):
         self._draw_maze_on_canvas(self.left_canvas, self.maze, show_full_maze=False, agent=self.robot)

    def draw_right_maze(self):
        if self.mode.get() == "robot_vs_maze":
             self._draw_maze_on_canvas(self.right_canvas, self.maze, show_full_maze=False, agent=self.robot)
        else:
             self._draw_maze_on_canvas(self.right_canvas, self.maze, show_full_maze=False, agent=self.player)


    def _draw_maze_on_canvas(self, canvas, maze, show_full_maze, agent):
        canvas.delete("all")
        for y in range(maze.height):
            for x in range(maze.width):
                x1, y1 = x * CELL_SIZE, y * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE

                # For the robot's view, only draw what it knows
                is_known = (x, y) in agent.known_maze or (x,y) in agent.path

                if show_full_maze or is_known:
                    # Draw path/visited cells
                    path_color = BG_COLOR
                    if (x, y) in agent.path:
                        path_color = PATH_COLOR if show_full_maze else KNOWN_PATH_COLOR
                    canvas.create_rectangle(x1, y1, x2, y2, fill=path_color, outline="")

                    # Draw walls
                    open_walls = maze.get_valid_moves(x, y)
                    canvas.lineWidth = 5
                    if 'N' not in open_walls:
                        canvas.create_line(x1, y1, x2, y1, fill=WALL_COLOR, width = 5)
                    if 'S' not in open_walls:
                        canvas.create_line(x1, y2, x2, y2, fill=WALL_COLOR, width = 5)
                    if 'W' not in open_walls:
                        canvas.create_line(x1, y1, x1, y2, fill=WALL_COLOR, width = 5)
                    if 'E' not in open_walls:
                        canvas.create_line(x2, y1, x2, y2, fill=WALL_COLOR, width = 5)

        # Draw Start and Goal
        self._draw_marker(canvas, agent.start_pos, START_COLOR)
        self._draw_marker(canvas, agent.goal_pos, GOAL_COLOR)

        # Draw Agent
        agent_color = ROBOT_COLOR if agent == self.robot else PLAYER_COLOR
        self._draw_marker(canvas, (agent.x, agent.y), agent_color, is_agent=True)

    def _draw_marker(self, canvas, pos, color, is_agent=False):
        x, y = pos
        margin = 2 if not is_agent else 3
        x1, y1 = x * CELL_SIZE + margin, y * CELL_SIZE + margin
        x2, y2 = (x + 1) * CELL_SIZE - margin, (y + 1) * CELL_SIZE - margin
        canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def draw_winner_message(self, message):
        # Winner message should appear on the main "action" canvas, which is the right one.
        canvas = self.right_canvas
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w > 0 and h > 0: # Ensure canvas has been rendered
            canvas.create_rectangle(w/2 - 100, h/2 - 30, w/2 + 100, h/2 + 30, fill=BG_COLOR, outline="white")
            canvas.create_text(w/2, h/2, text=message, fill="white", font=("Helvetica", 16, "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()


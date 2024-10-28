import random
import heapq

def heuristic(a, b):
    """Calculate the Manhattan distance heuristic between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid, start, goal):
    """Perform A* search algorithm to find the shortest path."""
    width = len(grid[0])
    height = len(grid)

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            total_path = [current]
            while current in came_from:
                current = came_from[current]
                total_path.append(current)
            total_path.reverse()
            return total_path

        x, y = current
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                if grid[neighbor[0]][neighbor[1]] != 1:  # Not a wall
                    tentative_g_score = g_score[current] + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score_neighbor = tentative_g_score + heuristic(neighbor, goal)
                        f_score[neighbor] = f_score_neighbor
                        heapq.heappush(open_set, (f_score_neighbor, neighbor))

    return None

def generate_grid(width, height):
    """Generate a grid map with walls around edges, start in corner, target in center, and square wall around center with two wide exits."""
    grid = [[0 for _ in range(width)] for _ in range(height)]

    # Add walls around the edges
    for i in range(height):
        grid[i][0] = 1  # Left wall
        grid[i][width - 1] = 1  # Right wall
    for j in range(width):
        grid[0][j] = 1  # Top wall
        grid[height - 1][j] = 1  # Bottom wall

    # Place start position in a corner
    corners = [(1, 1), (1, width - 2), (height - 2, 1), (height - 2, width - 2)]
    start = random.choice(corners)
    grid[start[0]][start[1]] = 0  # Ensure start position is open

    # Place target in center
    center_i = height // 2
    center_j = width // 2
    goal = (center_i, center_j)
    grid[center_i][center_j] = 0  # Ensure goal position is open

    # Create square wall around the center target
    wall_size = 7  # Size of the square wall (should be odd to center around target)
    half_wall = wall_size // 2
    top = center_i - half_wall
    bottom = center_i + half_wall
    left = center_j - half_wall
    right = center_j + half_wall

    # Build walls around the square
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if i == top or i == bottom or j == left or j == right:
                grid[i][j] = 1

    # Create two wide exits/entries in the square wall
    exit_width = 3  # Width of the exits
    # Exits on the left and right walls
    for offset in range(-exit_width // 2, exit_width // 2 + 1):
        # Left exit
        exit_i = center_i + offset
        exit_j = left
        grid[exit_i][exit_j] = 0
        # Right exit
        exit_i = center_i + offset
        exit_j = right
        grid[exit_i][exit_j] = 0

    return grid, start, goal

def print_grid(grid, start=None, goal=None, path=None):
    """Print the grid with start, goal, and path."""
    height = len(grid)
    width = len(grid[0])
    for i in range(height):
        row = ''
        for j in range(width):
            if (i, j) == start:
                row += '@ '
            elif (i, j) == goal:
                row += 'X '
            elif path and (i, j) in path:
                row += '+ '
            elif grid[i][j] == 1:
                row += '1 '
            else:
                row += '. '
        print(row)

def visualize_path(grid, start, goal, path):
    """Visualize the path on the grid."""
    print_grid(grid, start=start, goal=goal, path=path)

def main():
    width = 50
    height = 50
    grid, start, goal = generate_grid(width, height)

    print("Initial grid:")
    print_grid(grid, start=start, goal=goal)
    print(f"\nFinding path from {start} to {goal}...\n")

    path = a_star_search(grid, start, goal)

    if path:
        print("Path found:")
        visualize_path(grid, start, goal, path)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()

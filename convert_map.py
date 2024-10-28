import heapq
from PIL import Image, ImageOps
import numpy as np

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

def image_to_grid(image_path):
    """Convert an image to a grid matrix where dark regions are walls."""
    # Open the image and convert to grayscale
    image = Image.open(image_path).convert('L')

    # Resize the image to a manageable size if necessary
    image = image.resize((50, 50))  # Adjust the size as needed

    # Invert the image so that walls are dark regions
    image = ImageOps.invert(image)

    # Convert image to numpy array
    img_array = np.array(image)

    # Normalize the pixel values
    img_array = img_array / 255.0

    # Threshold the image to create walls
    threshold = 0.5  # Adjust the threshold as needed
    grid = (img_array > threshold).astype(int)

    return grid

def find_start_and_goal(grid):
    """Find start and goal positions in the grid."""
    height, width = grid.shape

    # Start position at the top-left corner that is not a wall
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 0:
                start = (i, j)
                break
        else:
            continue
        break

    # Goal position at the bottom-right corner that is not a wall
    for i in reversed(range(height)):
        for j in reversed(range(width)):
            if grid[i][j] == 0:
                goal = (i, j)
                break
        else:
            continue
        break

    return start, goal

def print_grid(grid, start=None, goal=None, path=None):
    """Print the grid with start, goal, and path."""
    height, width = grid.shape
    for i in range(height):
        row = ''
        for j in range(width):
            if (i, j) == start:
                row += '@ '
            elif (i, j) == goal:
                row += 'X '
            elif path and (i, j) in path:
                row += '. '
            elif grid[i][j] == 1:
                row += '1 '
            else:
                row += '0 '
        print(row)

def visualize_path(grid, start, goal, path):
    """Visualize the path on the grid."""
    print_grid(grid, start=start, goal=goal, path=path)

def main():
    image_path = 'left_1.jpg'  # Replace with your image of robot arena you want to create grid

    grid = image_to_grid(image_path)

    start, goal = find_start_and_goal(grid)

    print("Grid generated from image:")
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

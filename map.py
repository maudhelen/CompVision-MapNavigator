import keyboard

map_grid = [
    [" ", " ", " ", " ", " "],
    [" ", "X", "X", "X", " "],
    [" ", " ", " ", " ", " "],
    [" ", "X", "X", "X", " "],
    [" ", " ", " ", " ", " "]
]

# Initialize the starting position
y, x = 2, 2  # Example starting position

def move_up(y, x):
    if y == 0:
        print("Already at the top! Cannot move up")
        return y, x
    elif map_grid[y-1][x] == 'X':
        print("Obstacle detected! Cannot move up.")
        return y, x
    else:
        return y - 1, x

def move_down(y, x):
    if y == len(map_grid) - 1:
        print("Already at the bottom! Cannot move down")
        return y, x
    elif map_grid[y+1][x] == 'X':
        print("Obstacle detected! Cannot move down")
        return y, x
    else:
        return y + 1, x

def move_left(y, x):
    if x == 0:
        print("Already at the leftmost! Cannot move left")
        return y, x
    elif map_grid[y][x-1] == 'X':
        print("Obstacle detected! Cannot move down")
        return y, x
    else:
        return y, x - 1

def move_right(y, x):
    if x == len(map_grid[0]) - 1:
        print("Already at the rightmost! Cannot move right")
        return y, x
    elif map_grid[y][x+1] == 'X':
        print("Obstacle detected! Cannot move down")
        return y, x
    else:
        return y, x + 1
    
def print_map(pos):
    for i in range(len(map_grid)):
        for j in range(len(map_grid[i])):
            if (i, j) == pos:
                print("O", end=" ")
            else:
                print(map_grid[i][j], end=" ")
        print()

def update_position(new_y, new_x):
    global y, x
    y, x = new_y, new_x
    print(f"New position: ({y}, {x})")
    print_map((y, x))
    
def event_navigate(event):
    global y, x
    if event.name == "up":
        new_y, new_x = move_up(y, x)
    elif event.name == "down":
        new_y, new_x = move_down(y, x)
    elif event.name == "left":
        new_y, new_x = move_left(y, x)
    elif event.name == "right":
        new_y, new_x = move_right(y, x)
    else:
        print("Invalid key")
        return
    update_position(new_y, new_x)

# Register the keyboard hooks
keyboard.on_press_key("up", event_navigate)
keyboard.on_press_key("down", event_navigate)
keyboard.on_press_key("left", event_navigate)
keyboard.on_press_key("right", event_navigate)

print("Use arrow keys to navigate the map. Press ESC to exit.")
print_map((y, x))

# Keep the program running and listening for key events
keyboard.wait('space')

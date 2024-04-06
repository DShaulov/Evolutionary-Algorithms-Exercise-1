import math

def get_coords(filename):
    """
    Reads the file with the city locations and returns them in a list
    """
    coords = []
    with open(filename, 'r') as file:
        for line in file:
            coords.append([int(x) for x in line.split()])
    return coords

def get_path(filename):
    integer_list = []
    # Open the file in read mode
    with open(filename, 'r') as file:
        # Read lines into a list, stripping newline characters and converting to integers
        integer_list = [int(line.strip()) for line in file]
    return integer_list

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def total_distance(route, city_coords):
    """
    Calculates the sum of distance of cities in the order they appear in the route.
    Returns 1/fitness so that the smaller the total distance the better
    """
    total_distance = 0
    for i in range(len(route)):
        city_index = route[i]
        next_city_index = route[(i + 1) % len(route)]
        city = city_coords[city_index - 1]
        next_city = city_coords[next_city_index - 1]
        distance = euclidean_distance(city[0], city[1], next_city[0], next_city[1])
        total_distance += distance
    return total_distance

if __name__ == "__main__":
    coords = get_coords('tsp.txt')
    route = get_path('317005403.txt')
    print(total_distance(route, coords))
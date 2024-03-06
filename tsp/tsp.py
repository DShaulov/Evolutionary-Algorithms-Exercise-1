import random
import time
import math
# import matplotlib.pyplot as plt

def init_population(population_size):
    """
    Creates the initial population of cities.
    """
    population = []
    indices = [x for x in range(1, 49)]
    for i in range(population_size):
        route = indices.copy()
        random.shuffle(route)
        population.append(route)
    return population

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_coords(filename):
    """
    Reads the file with the city locations and returns them in a list
    """
    coords = []
    with open(filename, 'r') as file:
        for line in file:
            coords.append([int(x) for x in line.split()])
    print(coords)
    return coords


def fitness(route, city_coords):
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
    fitness = 1 / total_distance
    return fitness

def selection(population, fitness_scores, elitism_size, s_type):
    """
    Performs selection based on the specified type
    """
    selection_types = {
        "roulette": roulette_selection,
        "elitism": elitism_selection
    }
    return selection_types[s_type](population, fitness_scores, elitism_size)

def roulette_selection(population, fitness_scores, elitism_size):
    """
    Uses roulette wheel selection to choose a route
    """
    fitness_scores_sum = sum(fitness_scores)
    random_value = random.uniform(0, fitness_scores_sum)
    iter_sum = 0
    for i in range(len(fitness_scores)):
        iter_sum += fitness_scores[i]
        if random_value <= iter_sum:
            return population[i]
def elitism_selection(population, fitness_scores, elite_size):
    """
    Uses elitism selection according to elite_size
    """
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    top_indices = sorted_indices[:elite_size]
    chosen_index = random.choice(top_indices)
    return population[chosen_index]


def crossover(route_A, route_B):
    """
    Performs crossover - takes a subsection of parent A and adds cities that are not present in the subsection in order from parent B.
    """
    first_crossover_point = random.choice(range(0, 48))
    second_crossover_point = random.choice(range(0, 48))
    while first_crossover_point == second_crossover_point:
        second_crossover_point = random.choice(range(0, 48))
    if second_crossover_point < first_crossover_point:
        temp = first_crossover_point
        first_crossover_point = second_crossover_point
        second_crossover_point = temp
    first_child = [0] * 48
    second_child = [0] * 48
    first_child[first_crossover_point:second_crossover_point] = route_A[first_crossover_point:second_crossover_point]
    second_child[first_crossover_point:second_crossover_point] = route_B[first_crossover_point:second_crossover_point]
    complete_child(first_child, route_B, second_crossover_point)
    complete_child(second_child, route_A, second_crossover_point)
    return [first_child, second_child]

def complete_child(child, parent, second_crossover_point):
    index = second_crossover_point
    for city in parent:
        if city not in child:
            if index >= len(parent):
                index = 0
            child[index] = city
            index += 1

def mutation(route):
    """
    Chooses a city at random and places it a random index, shifting all cities one index to the right
    """
    random_index = random.choice(range(1, 48))
    city_at_index = route.pop(random_index)
    new_index = random.choice(range(1, 48))
    route.insert(new_index, city_at_index)
    return

if __name__ == "__main__":
    start_time = time.time()
    # Hyper paramters
    num_generations = 10000
    population_size = 250
    elitism_size = 10
    mutation_rate = 0.1
    s_type = "elitism"

    # Main loop
    city_coords = get_coords("tsp.txt")
    population = init_population(population_size)
    best_gen_routes = []
    best_route = []
    best_route_generation = 0
    smallest_distance = float('inf')
    elapsed_generations = 0
    for i in range(num_generations):
        # Evaluate fitness
        fitness_scores = [fitness(route, city_coords) for route in population]
        max_fitness = max(fitness_scores)
        best_gen_routes.append(1 / max_fitness)
        max_fitness_index = fitness_scores.index(max(fitness_scores))
        total_distance = 1 / max_fitness
        print("Generation " + str(i) + ": shortest route distance " + str(total_distance))
        if smallest_distance > total_distance:
            smallest_distance = total_distance
            best_route = population[max_fitness_index]
            best_route_generation = i
        # Create the next generation
        new_population = []
        while len(new_population) < population_size:
            parent_A = selection(population, fitness_scores, elitism_size, s_type)
            parent_B = selection(population, fitness_scores, elitism_size, s_type)
            children = crossover(parent_A, parent_B)
            if random.random() <= mutation_rate:
                mutation(children[0])
                mutation(children[1])
            new_population += children
        population = new_population
        elapsed_generations += 1

    print("DONE")
    print("Best route: " + str(best_route))
    print("Generation " + str(best_route_generation))
    print("Total distance: " + str(1/fitness(best_route, city_coords)))
    end_time = time.time()
    print("Total runtime: " + str(end_time - start_time))

    # plt.figure(figsize=(10, 6))
    # plt.plot(best_gen_routes, label='Shortest Distance')
    # plt.title('Evolution of the Best Route Distance over Generations')
    # plt.xlabel('Generation')
    # plt.ylabel('Distance')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    file_name = 'test.txt'
    with open(file_name, 'w') as file:
        for city in best_route:
            file.write(f"{city}\n")
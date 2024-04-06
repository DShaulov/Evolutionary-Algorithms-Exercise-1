import random
import time

def init_population(population_size):
    """
    Creates the initial population of chromosomes.
    Chromosomes are represented as a list of length 8, where chromosome[i] is the row of the queen in the i'th column.
    """
    population = []
    indices = [1,2,3,4,5,6,7,8]
    for i in range(population_size):
        new_board = indices.copy()
        random.shuffle(new_board)
        population.append(new_board)
    return population


def fitness(board):
    """
    Calculates the fitness of chromosome.
    There are 28 different pairs of queens, a perfect solution has a score of 28 - no queen threatens another.
    """
    threats = 0
    for i in range(8):
        for j in range(i+1, 8):
            horizontal_threat = board[i] == board[j]
            diagonal_threat = board[i] - board[j] == j - i or board[j] - board[i] == j - i
            if horizontal_threat or diagonal_threat:
                threats += 1
    return 28 - threats

def selection(population, fitness_scores):
    """
    Uses roulette wheel selection to choose a parent board
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

def crossover(crossover_type, parent_A, parent_B, mutation_rate, mutation_repeat):
    """
    Performs crossover according to the specified type.
    """
    crossover_mapping = {
        'single_point': single_point_crossover,
        'two_point': two_point_crossover,
        'uniform': uniform_crossover
    }
    children = crossover_mapping.get(crossover_type)(parent_A, parent_B)
    if random.random() <= mutation_rate:
        for child in children:
            mutation(child, mutation_repeat)
    return children

def single_point_crossover(parent_A, parent_B):
    """
    Performs single point crossover and returns two children
    """
    indices = list(range(1, 8))
    crossover_index = random.choice(indices)
    first_child = parent_A[:crossover_index] + parent_B[crossover_index:]
    second_child = parent_B[:crossover_index] + parent_A[crossover_index:]
    return [first_child, second_child]

def two_point_crossover(parent_A, parent_B):
    """
    Performs two-point crossover and returns two children
    """
    indices = random.sample(range(1, 8), 2)
    indices.sort()
    crossover_index1, crossover_index2 = indices
    first_child = parent_A[:crossover_index1] + parent_B[crossover_index1:crossover_index2] + parent_A[crossover_index2:]
    second_child = parent_B[:crossover_index1] + parent_A[crossover_index1:crossover_index2] + parent_B[crossover_index2:]
    return [first_child, second_child]

def uniform_crossover(parent_A, parent_B):
    """
    Performs uniform crossover and returns two children.
    """
    first_child = []
    second_child = []

    for gene_A, gene_B in zip(parent_A, parent_B):
        if random.random() < 0.5:
            first_child.append(gene_A)
            second_child.append(gene_B)
        else:
            first_child.append(gene_B)
            second_child.append(gene_A)
    
    return [first_child, second_child]

def mutation(board, mutation_repeat):
    """
    Chooses two queens at random and swaps their places on the board a specified number of times.
    """
    for i in range(mutation_repeat):
        first_index = random.choice(range(len(board)))
        second_index = random.choice(range(len(board)))
        while first_index == second_index:
            second_index = random.choice(range(len(board)))
        temp = board[first_index]
        board[first_index] = board[second_index]
        board[second_index] = temp
    return


def eight_queens_bruteforce(repeat):
    """
    Tries to solve the 8-queen puzzle using brute force
    """
    start_time = time.time()
    solutions = dict()
    indices = [1,2,3,4,5,6,7,8]
    for i in range(repeat):
        new_board = indices.copy()
        random.shuffle(new_board)
        if fitness(new_board) == 28:
            board_as_tuple = tuple(new_board)
            if board_as_tuple not in solutions:
                print(new_board)
                solutions[board_as_tuple] = new_board
                break
    end_time = time.time()
    print("Found " + str(len(solutions)) + " unique solutions")
    print(end_time - start_time)
    return

if __name__ == "__main__":
    # eight_queens_bruteforce(1000000)
    start_time = time.time()
    # Hyper paramters
    num_generations = 10000
    population_size = 500
    crossover_type = "single_point"
    mutation_rate = 0.05
    mutation_repeat = 1

    # Main loop
    population = init_population(population_size)
    solutions = dict()
    elapsed_generations = 0
    for i in range(num_generations):
        # Evaluate fitness
        fitness_scores = [fitness(board) for board in population]
        max_fitness_index = fitness_scores.index(max(fitness_scores))
        # Check if a solution has been found
        solution_found = False
        for j in range(len(fitness_scores)):
            if fitness_scores[j] == 28:
                solution_found = True
                print(str(population[j]) + ", Generation: " + str(i))
                break
        if solution_found:
            break
        # Create the next generation
        new_population = []
        while len(new_population) < population_size:
            parent_A = selection(population, fitness_scores)
            parent_B = selection(population, fitness_scores)
            children = crossover(crossover_type, parent_A, parent_B, mutation_rate, mutation_repeat)
            new_population += children
        population = new_population
        elapsed_generations += 1

    end_time = time.time()
    print("Total runtime: " + str(end_time - start_time))
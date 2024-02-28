import math
import random
import matplotlib as plt


def init_population(population_size):
    """
    Creates the initial population of chromosomes.
    Chromosomes are represented as a list of length 8, where chromosome[i] is the row of the queen in the i'th column.
    """
    population = []
    indices = [1,2,3,4,5,6,7,8]
    for i in range(len(population_size)):
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
    return

def two_point_crossover(parent_A, parent_B):
    return

def uniform_crossover(parent_A, parent_B):
    return

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


def eight_queens_bruteforce():
    """
    Tries to solve the 8-queen puzzle using brute force
    """
    return

if __name__ == "__main__":
    # Hyper paramters
    num_generations = 30
    population_size = 100
    crossover_type = "single_point"
    mutation_rate = 0.1
    mutation_repeat = 1

    # Main loop
    population = init_population(population_size)
    max_fitness_values = []
    for i in range(len(num_generations)):
        # Evaluate fitness
        fitness_scores = [fitness(board) for board in population]
        max_fitness_index = fitness_scores.index(max(fitness_scores))
        max_fitness_values.append(fitness_scores[max_fitness_index])
        # Check if a solution has been found
        if fitness_scores[max_fitness_index] == 28:
            print("Generation Number " + i)
            print(population[max_fitness_index])
            break
        
        # Create the next generation
        new_population = []
        while len(new_population) < population_size:
            parent_A = selection(population, fitness_scores)
            parent_B = selection(population, fitness_scores)
            children = crossover(crossover_type, parent_A, parent_B, mutation_rate, mutation_repeat)
            new_population += children
        population = new_population

    # Delete later
    generations = [i for i in range(1, num_generations+1)]
    # Creating the plot
    plt.plot(generations, max_fitness_values, marker='o')  # Plots the data points and connects them with a line
    plt.title('Best Performing Chromosome over Generations')  # Title of the plot
    plt.xlabel('Generation')  # Label for the x-axis
    plt.ylabel('Best Score')  # Label for the y-axis
    plt.grid(True)  # Adds a grid for easier reading
    plt.show()  # Displays the plot
import nfl_strategy as nfl
import time
import random
from copy import copy, deepcopy

# Set partitioning divisions
max_yards_to_score = 80
max_yards_to_down = 10

# Set global bounds
y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4 = 0, 0, 0, 0, 0, 0, 0, 0
bounds = [y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4]
policy_individual = []
grid_size = 0
playbook_size = 0

# Depricated visit counts for debugging partition
visits = {}
visits['x_0, y_0'] = 0
visits['x_0, y_1'] = 0
visits['x_0, y_2'] = 0
visits['x_0, y_3'] = 0
visits['x_0, y_4'] = 0

visits['x_1, y_0'] = 0
visits['x_1, y_1'] = 0
visits['x_1, y_2'] = 0
visits['x_1, y_3'] = 0
visits['x_1, y_4'] = 0

visits['x_2, y_0'] = 0
visits['x_2, y_1'] = 0
visits['x_2, y_2'] = 0
visits['x_2, y_3'] = 0
visits['x_2, y_4'] = 0

visits['x_3, y_0'] = 0
visits['x_3, y_1'] = 0
visits['x_3, y_2'] = 0
visits['x_3, y_3'] = 0
visits['x_3, y_4'] = 0

visits['x_4, y_0'] = 0
visits['x_4, y_1'] = 0
visits['x_4, y_2'] = 0
visits['x_4, y_3'] = 0
visits['x_4, y_4'] = 0
bad = 0

# Genetic Algorithm
def ga_learn(model: nfl.NFLStrategy, time_limit, fitness_sims, grid=3):
    """ 
    Standard genetic algorithm
    Pre: a nfl strategy model, a time limit, a number of times to simulate fitness of individuals, and grid size
    Post: return a policy function of approximate state -> action mapping
    """
    
    # Global variables

    global grid_size
    global policy_individual
    global playbook_size
    global bounds
    global y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4

    # Hyperparameters
    probability_crossover = 0.8
    probability_mutation = 0.5
    population_size = 100
    elite_cull_count = int(population_size/10)
    grid_size = grid
    playbook_size = model.offensive_playbook_size()

    # Set the partitions
    if grid_size == 3:
        y_1 = (max_yards_to_score * .03125) 
        y_2 = (max_yards_to_score * .05) 

        x_1 = (max_yards_to_down * .24)
        x_2 = (max_yards_to_down * .4)
    
    elif grid_size == 4:
        y_1 = (max_yards_to_score * .03125) 
        y_2 = (max_yards_to_score * .037) 
        y_3 = (max_yards_to_score * .045) 

        x_1 = (max_yards_to_down * .22)
        x_2 = (max_yards_to_down * .3)
        x_3 = (max_yards_to_down * .4)

    elif grid_size == 5:
        y_1 = (max_yards_to_score * .03) 
        y_2 = (max_yards_to_score * .0325) 
        y_3 = (max_yards_to_score * .037) 
        y_4 = (max_yards_to_score * .045) 

        x_1 = (max_yards_to_down * .23)
        x_2 = (max_yards_to_down * .25)
        x_3 = (max_yards_to_down * .3)
        x_4 = (max_yards_to_down * .4)
    
    # initalize global bounds
    bounds = y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4

    # Bookkeeping
    generational_stats = []
    
    initial_population = initialize_population(population_size, grid_size, playbook_size)
    max_fitness, max_fitness_individual, generational_stats, population, final_fitness, final_population = \
        genetic_algorithm(model, initial_population, time_limit, probability_crossover, probability_mutation, generational_stats, fitness_sims, elite_cull_count)
    
    policy_individual = max_fitness_individual
    #print(f"Max Fitness: {max_fitness} Gene: {max_fitness_gene}")
    #print(f"Final Fitnesses: {final_fitness} Avg: {sum(final_fitness)/len(final_fitness)}")
    #print_visits(visits)

    return policy
 
# Initialize the population
def initialize_population(population_size, grid_size, playbook_size):
    population = []
    for i in range(0, population_size):
        individual = []
        for j in range(0, grid_size * grid_size):
            individual.append(random.choice(list(range(0, playbook_size))))
        population.append(individual)
    
    return population
        

def genetic_algorithm(model, population, time_left, probability_crossover, probability_mutation, generational_stats, fitness_sims, elite_cull_count):
    gen_count = 0
    while time_left > 0.2:
        # Get the time before this iteration begins
        start_time = time.time()
        fitness = []
        for individual in population:
            trial = get_fitness(model, individual, fitness_sims)
            fitness.append(trial)
        
        # Book Keeping
        final_fitness = fitness.copy()
        final_population = population.copy()
        max_fitness = max(fitness)
        min_fitness = min(fitness)
        avg_fitness = sum(fitness)/len(fitness)
        max_fitness_individual = population[fitness.index(max_fitness)].copy()
        this_generation_stats = (max_fitness, min_fitness, avg_fitness, gen_count)
        generational_stats.append(this_generation_stats)

        # Take our population of individual and get the top n% for later
        elites, elite_indexes = select_max(population, fitness, elite_cull_count)

        # Select the two for crossover and mutation (Two Random Rank)
        parent1, parent2 = select(population, fitness)

        # Crossover the selected  with a random single point crossover 
        crossed1, crossed2 = crossover(parent1, parent2, probability_crossover)

        # Mutate the entire population by potentially increasing and decreasing (wrap around) 2 genes
        population = mutate_population(population, probability_mutation)

        # Create new population by culling bottom 20% and replacing with the elite ones and crossed ones
        cull_random(population, fitness, elite_cull_count, elites, crossed1, crossed2)

        # Remove episode time from the allotted time
        end_time = time.time()
        time_left -= (end_time-start_time)
        gen_count += 1

    return max_fitness, max_fitness_individual, generational_stats, population, final_fitness, final_population

# Get fitness from simulating 
def get_fitness(model, individual, fitness_sims):
    global bounds, visits
    wins = 0
    play_count = 0
    # For each fitness simulation
    for i in range(fitness_sims):
        pos = model.initial_position()
        # Policy code for GA
        state = get_approximate_state(pos, bounds)
        index = (state[0] * grid_size) + state[1]
        action = individual[index]
        
        # Play out the game
        while not model.game_over(pos):
            play_count += 1
            pos, _ = model.result(pos, action)
            state = get_approximate_state(pos, bounds)
            index = (state[0] * grid_size) + state[1]
            action = individual[index]
        if model.win(pos):
            wins += 1
    # Return average # of wins
    return wins / fitness_sims

# Perform singlepoint crossover based on crossover probability
def crossover(parent1, parent2, probability_crossover):

    # If we hit the probability, then crossover on the parents
    random_val = random.random()
    if random_val < probability_crossover:

        # pick a random division for the crossover
        swap = 0
        cross_point = random.randrange(1, (grid_size*grid_size))

        # Perform Crossover
        crossover1 = []
        crossover2 = []
        for i in range(0, len(parent1)):
            if i % cross_point == 0 or swap == 1:
                crossover1.append(parent1[i])
                crossover2.append(parent2[i])
                swap = 1
            else:
                crossover1.append(parent2[i])
                crossover2.append(parent1[i])
        
        # Return the Crossover
        return crossover1, crossover2

    else:
        return parent1, parent2

# Mutate the entire population by adding 1 to a random action
# and subtracting one from another
def mutate_population(population, probability_mutation):
    global playbook_size
    for individual in population:
        random_val = random.random()

        # If we hit the probabilty, mutate
        if random_val < probability_mutation:

            # Get random indices to mutate
            rand_add = random.randrange(len(individual))
            rand_sub = random.randrange(len(individual))

            # Only do if it doesn't break bounds, 
            # otherwise, wrap around
            if individual[rand_add] < playbook_size-1:
                individual[rand_add] += 1
            elif individual[rand_add] == playbook_size-1:
                individual[rand_add] = 0
            
            if individual[rand_sub] > 0:
                individual[rand_sub] -= 1
            elif individual[rand_add] == 0:
                individual[rand_add] = playbook_size-1

    return population
       
# Select and return two via roulette, with probability of choice
# weighted on their fitness in the old generation
def select(old_gen, fitness):
    old_gen_temp = old_gen.copy()
    fitness_temp = fitness.copy()

    # Get the first individual
    # Normalize the weights
    min_value = min(fitness_temp)
    fitness_temp = list(map(lambda x: (x + abs(min_value) + 1), fitness_temp))

    individual1 = random.choices(old_gen_temp, weights=fitness_temp, k=1)
    individual1 = individual1[0]

    # Get the index from first individual and  remove the fitness of it from fitness, 
    # and the individual from oldgen to get the second
    individual1_index = old_gen_temp.index(individual1)
    del(old_gen_temp[individual1_index])
    del(fitness_temp[individual1_index])

    # Get the second individual
    individual2 = random.choices(old_gen_temp, weights=fitness_temp, k=1)
    individual2 = individual2[0] 

    return individual1, individual2

# Select and return n best individuals in the old generation
def select_max(population, fitness, n):
    fitness_temp = fitness.copy()

    elites = []
    elite_indexes = []
    for i in range(0, n):
        # Get the index of the max fitness
        elite_fitness = max(fitness_temp)
        elite_index = fitness.index(elite_fitness) 
        elite_temp_index = fitness_temp.index(elite_fitness) 
        elite_individual = population[elite_index].copy()

        # Add to our list of elites and elite indices
        elites.append(elite_individual)
        elite_indexes.append(elite_index)

        # Remove from temp to keep getting next max
        del(fitness_temp[elite_temp_index])

    return elites, elite_indexes

# Cull random n in population, this produces the next generation.
def cull_random(population, fitness, n, elites, crossed1, crossed2):

    # Inverse the fitness so we can weigh bad ones higher, +1 for smoothing of weights
    inverse = fitness.copy()
    max_inverse = max(inverse)
    inverse = list(map(lambda x: max_inverse - x + 1, inverse))

    # Cull two random ones weighted towards the bad ones.
    for individual in range(0,n+2):
        culled = random.choices(population, weights=inverse, k = 1)
        culled = culled[0]
        culled_index = population.index(culled)
        del(inverse[culled_index])
        del(population[culled_index])
    
    # reintroduce crossovers / elites
    population.extend(elites)
    population.extend([crossed1, crossed2])


# Policy function
def policy(pos):
    """ 
    Policy for choosing and action from our learned model
    Pre: a position in the game, a globally initialized policy individual from GA
    Post: action to take
    """
    global policy_individual
    global grid_size
    global playbook_size
    global bounds
    pos = get_approximate_state(pos, bounds)

    index = pos[0] * grid_size + pos[1]

    return policy_individual[index]

# Approximate state function
def get_approximate_state(position, bounds):
    """ 
    Conversion from position to approximate state in a nxn grid
    Pre: a position in the game, and bounds
    Post: approximation of state 
    """
    global grid_size
    y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4= bounds
    if grid_size == 3: 
        # Unwrap the bounds and position
        yards_to_score, downs_left, distance_to_down, time_left = position

        # Set the y box
        if time_left == 0:
            y = 0
        elif yards_to_score/time_left > y_2:
            y = 2
        elif yards_to_score/time_left > y_1:
            y = 1
        elif yards_to_score/time_left >= 0:
            y = 0

        # Set the x box
        if distance_to_down/downs_left > x_2:
            x = 2
        elif distance_to_down/downs_left > x_1:
            x = 1
        elif distance_to_down/downs_left >= 0:
            x = 0
    elif grid_size == 4: 
        # Unwrap the bounds and position
        yards_to_score, downs_left, distance_to_down, time_left = position

        # Set the y box
        if time_left == 0:
            y = 0
        elif yards_to_score/time_left > y_3:
            y = 3
        elif yards_to_score/time_left > y_2:
            y = 2
        elif yards_to_score/time_left > y_1:
            y = 1
        elif yards_to_score/time_left >= 0:
            y = 0

        # Set the x box
        if distance_to_down/downs_left > x_3:
            x = 3
        elif distance_to_down/downs_left > x_2:
            x = 2
        elif distance_to_down/downs_left > x_1:
            x = 1
        elif distance_to_down/downs_left >= 0:
            x = 0
    elif grid_size == 5: 
        # Unwrap the bounds and position
        yards_to_score, downs_left, distance_to_down, time_left = position

        # Set the y box
        if time_left == 0:
            y = 0
        elif yards_to_score/time_left > y_4:
            y = 4
        elif yards_to_score/time_left > y_3:
            y = 3
        elif yards_to_score/time_left > y_2:
            y = 2
        elif yards_to_score/time_left > y_1:
            y = 1
        elif yards_to_score/time_left >= 0:
            y = 0

        # Set the x box
        if distance_to_down/downs_left > x_4:
            x = 4
        elif distance_to_down/downs_left > x_3:
            x = 3
        elif distance_to_down/downs_left > x_2:
            x = 2
        elif distance_to_down/downs_left > x_1:
            x = 1
        elif distance_to_down/downs_left >= 0:
            x = 0
    return (y, x)

# Standard argmax function
def argmax(list):
    """ 
    Get the max argument of the a list
    Pre: list of comparable values
    Post: return the index of the max value 
    """
    max_item = max(list)
    return list.index(max_item)

""" ------------------------OLD FUNCTIONS FOR DEBUGGING----------------------------------------"""
def print_visits(visits):
    global bounds
    sum_x0 = visits['x_0, y_0'] + visits['x_0, y_1'] + visits['x_0, y_2'] + visits['x_0, y_3'] + visits['x_0, y_4']
    sum_x1 = visits['x_1, y_0'] + visits['x_1, y_1'] + visits['x_1, y_2'] + visits['x_1, y_3'] + visits['x_1, y_4']
    sum_x2 = visits['x_2, y_0'] + visits['x_2, y_1'] + visits['x_2, y_2'] + visits['x_2, y_3'] + visits['x_2, y_4']
    sum_x3 = visits['x_3, y_0'] + visits['x_3, y_1'] + visits['x_3, y_2'] + visits['x_3, y_3'] + visits['x_3, y_4']
    sum_x4 = visits['x_4, y_0'] + visits['x_4, y_1'] + visits['x_4, y_2'] + visits['x_4, y_3'] + visits['x_4, y_4']

    sum_y0 = visits['x_0, y_0'] + visits['x_1, y_0'] + visits['x_2, y_0'] + visits['x_3, y_0'] + visits['x_4, y_0']
    sum_y1 = visits['x_0, y_1'] + visits['x_1, y_1'] + visits['x_2, y_1'] + visits['x_3, y_1'] + visits['x_4, y_1']
    sum_y2 = visits['x_0, y_2'] + visits['x_1, y_2'] + visits['x_2, y_2'] + visits['x_3, y_2'] + visits['x_4, y_2']
    sum_y3 = visits['x_0, y_3'] + visits['x_1, y_3'] + visits['x_2, y_3'] + visits['x_3, y_3'] + visits['x_4, y_3']
    sum_y4 = visits['x_0, y_4'] + visits['x_1, y_4'] + visits['x_2, y_4'] + visits['x_3, y_4'] + visits['x_4, y_4']

    print(f"x_4 > {bounds[7]} x_3 > {bounds[6]} x_2 > {bounds[5]} x_1 > {bounds[4]} y_4 > {bounds[3]} y_3 > {bounds[2]} y_2 > {bounds[1]} y_1 > {bounds[0]}")
    print(f"Total x = {sum_x0+sum_x1+sum_x2+sum_x3+sum_x4} division should be {(sum_x0+sum_x1+sum_x2+sum_x3+sum_x4)/5}")
    print(f"Sum x0: {sum_x0}")
    print(f"Sum x1: {sum_x1}")
    print(f"Sum x2: {sum_x2}")
    print(f"Sum x3: {sum_x3}")
    print(f"Sum x4: {sum_x4}")
    print("\n")
    print(f"Total y = {sum_y0+sum_y1+sum_y2+sum_y3+sum_y4} division should be {(sum_y0+sum_y1+sum_y2+sum_y3+sum_y4)/5}")
    print(f"Sum y0: {sum_y0}")
    print(f"Sum y1: {sum_y1}")
    print(f"Sum y2: {sum_y2}")
    print(f"Sum y3: {sum_y3}")
    print(f"Sum y4: {sum_y4}")

def increment_visits(y,x):
    global visits
    if x == 0 and y == 0:
        visits['x_0, y_0'] += 1
    elif x == 0 and y == 1:
        visits['x_0, y_1'] += 1
    elif x == 0 and y == 2:
        visits['x_0, y_2'] += 1
    elif x == 0 and y == 3:
        visits['x_0, y_3'] += 1
    elif x == 0 and y == 4:
        visits['x_0, y_4'] += 1

    elif x == 1 and y == 0:
        visits['x_1, y_0'] += 1
    elif x == 1 and y == 1:
        visits['x_1, y_1'] += 1
    elif x == 1 and y == 2:
        visits['x_1, y_2'] += 1
    elif x == 1 and y == 3:
        visits['x_1, y_3'] += 1
    elif x == 1 and y == 4:
        visits['x_1, y_4'] += 1
    
    elif x == 2 and y == 0:
        visits['x_2, y_0'] += 1
    elif x == 2 and y == 1:
        visits['x_2, y_1'] += 1
    elif x == 2 and y == 2:
        visits['x_2, y_2'] += 1
    elif x == 2 and y == 3:
        visits['x_2, y_3'] += 1
    elif x == 2 and y == 4:
        visits['x_2, y_4'] += 1

    elif x == 3 and y == 0:
        visits['x_3, y_0'] += 1
    elif x == 3 and y == 1:
        visits['x_3, y_1'] += 1
    elif x == 3 and y == 2:
        visits['x_3, y_2'] += 1
    elif x == 3 and y == 3:
        visits['x_3, y_3'] += 1
    elif x == 3 and y == 4:
        visits['x_3, y_4'] += 1

    elif x == 4 and y == 0:
        visits['x_4, y_0'] += 1
    elif x == 4 and y == 1:
        visits['x_4, y_1'] += 1
    elif x == 4 and y == 2:
        visits['x_4, y_2'] += 1
    elif x == 4 and y == 3:
        visits['x_4, y_3'] += 1
    elif x == 4 and y == 4:
        visits['x_4, y_4'] += 1
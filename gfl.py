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
policy_gene = []
grid_size = 0
playbook_size = 0

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

# Q learning function
def ga_learn(model: nfl.NFLStrategy, time_limit, fitness_sims):
    """ 
    Standard genetic algorithm
    Pre: 
    Post:
    """
    
    # Global variables

    global grid_size
    global policy_gene
    global playbook_size
    global bounds
    global y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4

    # Hyperparameters
    probability_crossover = 0.8
    probability_mutation = 0.5
    population_size = 100
    elite_cull_count = int(population_size/10)
    grid_size = 3
    playbook_size = model.offensive_playbook_size()

    # q learning does .48
    if grid_size == 3:
        y_1 = (max_yards_to_score * .03125) 
        y_2 = (max_yards_to_score * .05) 

        x_1 = (max_yards_to_down * .24)
        x_2 = (max_yards_to_down * .4)
    
    # Q learning does .46
    elif grid_size == 4:
        y_1 = (max_yards_to_score * .03125) 
        y_2 = (max_yards_to_score * .037) 
        y_3 = (max_yards_to_score * .045) 

        x_1 = (max_yards_to_down * .22)
        x_2 = (max_yards_to_down * .3)
        x_3 = (max_yards_to_down * .4)

    # Q learning does .46
    elif grid_size == 5:
        y_1 = (max_yards_to_score * .03) 
        y_2 = (max_yards_to_score * .0325) 
        y_3 = (max_yards_to_score * .037) 
        y_4 = (max_yards_to_score * .045) 

        x_1 = (max_yards_to_down * .23)
        x_2 = (max_yards_to_down * .25)
        x_3 = (max_yards_to_down * .3)
        x_4 = (max_yards_to_down * .4)
    
    bounds = y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4

    # Bookkeeping
    generational_stats = []
    
    initial_population = initialize_population(population_size, grid_size, playbook_size)
    max_fitness, max_fitness_gene, generational_stats, population, final_fitness, final_population = \
        genetic_algorithm(model, initial_population, time_limit, probability_crossover, probability_mutation, generational_stats, fitness_sims, elite_cull_count)
    
    policy_gene = max_fitness_gene
    print(f"Max Fitness: {max_fitness} Gene: {max_fitness_gene}")
    print(f"Final Fitnesses: {final_fitness} Avg: {sum(final_fitness)/len(final_fitness)}")
    print_visits(visits)

    return policy
 
# Initialize the population
def initialize_population(population_size, grid_size, playbook_size):
    population = []
    for i in range(0, population_size):
        gene = []
        for j in range(0, grid_size * grid_size):
            gene.append(random.choice(list(range(0,playbook_size))))
        population.append(gene)
    
    return population
        

def genetic_algorithm(model, population, time_left, probability_crossover, probability_mutation, generational_stats, fitness_sims, elite_cull_count):
    gen_count = 0
    while time_left > 0.2:
        # Get the time before this iteration begins
        start_time = time.time()
        fitness = []
        for gene in population:
            trial = get_fitness(model, gene, policy, fitness_sims)
            fitness.append(trial)
        
        # Book Keeping
        final_fitness = fitness.copy()
        final_population = population.copy()
        max_fitness = max(fitness)
        min_fitness = min(fitness)
        avg_fitness = sum(fitness)/len(fitness)
        max_fitness_gene = population[fitness.index(max_fitness)].copy()
        this_generation_stats = (max_fitness, min_fitness, avg_fitness, gen_count)
        generational_stats.append(this_generation_stats)


        print(f"Gen Count: {gen_count} Max: {max_fitness} Min: {min_fitness} Avg: {avg_fitness}")
        print(max_fitness_gene)

        # Take our population of genes and get the top 20% for later
        elite_genes, elite_indexes = select_max(population, fitness, elite_cull_count)
        #print(f"Elite1 {elite1_index, fitness[elite1_index]} Elite2 {elite2_index, fitness[elite2_index]}")

        # Select the two for crossover and mutation (Two Random Rank)
        parent1, parent2 = select(population, fitness)

        # Crossover the selected  with a 2 point crossover (split into thirds, alternate between parents.)
        crossed1, crossed2 = crossover(parent1, parent2, probability_crossover)

        # Mutate the entire population by potentially increasing (wrap around) digits 2 and 3 of every single state. (chance for random reset of 0
        # as well)
        population = mutate_population(population, probability_mutation)

        # Create new population by culling bottom 20% and replacing with the elite ones and crossed ones
        cull_random(population, fitness, elite_cull_count, elite_genes, crossed1, crossed2)

        # Remove episode time from the allotted time
        end_time = time.time()
        time_left -= (end_time-start_time)
        gen_count += 1

    return max_fitness, max_fitness_gene, generational_stats, population, final_fitness, final_population

# Get fitness from simulating against a model
def get_fitness(model, gene, policy, fitness_sims):
    global bounds, visits
    wins = 0
    play_count = 0
    for i in range(fitness_sims):
        pos = model.initial_position()
        #print(bounds)
        state = get_approximate_state(pos, bounds)
        index = (state[0] * grid_size) + state[1]
        action = gene[index]
        while not model.game_over(pos):
            play_count += 1
            pos, _ = model.result(pos, action)
            state = get_approximate_state(pos, bounds)
            increment_visits(*state)
            index = (state[0] * grid_size) + state[1]
            action = gene[index]
        if model.win(pos):
            wins += 1
    return wins / fitness_sims

# Perform singlepoint crossover based on crossover probability, with points being at the separation of the state action brackets
def crossover(parent1, parent2, probability_crossover):

    # If we hit the probability, then crossover on the parents
    random_val = random.random()
    if random_val < probability_crossover:

        # pick a random division for the crossover
        swap = 0
        cross_point = random.randrange(1, (grid_size*grid_size))
        #print(cross_point)
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

# Mutate the entire population by adding 1 to a random move location and subtracting one from another
def mutate_population(population, probability_mutation):
    global playbook_size
    for gene in population:
        random_val = random.random()

        # If we hit the probabilty, mutate
        if random_val < probability_mutation:

            #Get random index to add to and subtract from
            rand_add = random.randrange(len(gene))
            rand_sub = random.randrange(len(gene))

            # Only do if it doesn't break, otherwise, wrap
            if gene[rand_add] < playbook_size-1:
                gene[rand_add] += 1
            elif gene[rand_add] == playbook_size-1:
                gene[rand_add] = 0
            
            if gene[rand_sub] > 0:
                gene[rand_sub] -= 1
            elif gene[rand_add] == 0:
                gene[rand_add] = playbook_size-1

               
    return population
       
# Select and return two via roulette, with probability of choice weighted on their fitness in the old generation
def select(old_gen, fitness):
    old_gen_temp = old_gen.copy()
    fitness_temp = fitness.copy()

    # Get the first gene
    # Normalize the weights
    min_value = min(fitness_temp)
    fitness_temp = list(map(lambda x: (x + abs(min_value) + 1), fitness_temp))

    gene1 = random.choices(old_gen_temp, weights=fitness_temp, k=1)
    gene1 = gene1[0]

    # Get the index from first gene and  remove the fitness of it from fitness, and the gene from oldgen to get the second
    gene1_index = old_gen_temp.index(gene1)
    del(old_gen_temp[gene1_index])
    del(fitness_temp[gene1_index])

    # Get the second
    #print(f"Fitness {fitness}")
    gene2 = random.choices(old_gen_temp, weights=fitness_temp, k=1)
    gene2 = gene2[0] 

    return gene1, gene2

# Select and return two best indexes in the old generation for elitism later. Remember their indexes
def select_max(population, fitness, n):
    fitness_temp = fitness.copy()
    population_temp = population.copy()

    elites = []
    elite_indexes = []
    for i in range(0, n):
        # Get the first index
        elite_fitness = max(fitness_temp)
        elite_index = fitness.index(elite_fitness) 
        elite_temp_index = fitness_temp.index(elite_fitness) 
        elite_gene = population[elite_index].copy()

        elites.append(elite_gene)
        elite_indexes.append(elite_index)

        # Remove from temp to keep getting next max
        del(fitness_temp[elite_temp_index])

    return elites, elite_indexes

# Cull random 2 in population (keeping the elite values) and then add in the mutated. This produces the next generation.
def cull_random(population, fitness, n, elite_genes, crossed1, crossed2):

    # Inverse the fitness so we can weigh bad ones higher, +1 for smoothing of weights
    inverse = fitness.copy()
    max_inverse = max(inverse)
    inverse = list(map(lambda x: max_inverse - x + 1, inverse))

    # Cull two random ones weighted towards the bad ones.
    for gene in range(0,n+2):
        culled = random.choices(population, weights=inverse, k = 1)
        culled = culled[0]
        culled_index = population.index(culled)
        del(inverse[culled_index])
        del(population[culled_index])
    
    population.extend(elite_genes)
    population.extend([crossed1, crossed2])


# Policy function
def policy(pos):
    """ 
    Policy for choosing and action from our learned model
    Pre: a position in the game, an initialized q_table, and bounds
    Post: action to take
    """
    global policy_gene
    global grid_size
    global playbook_size
    global bounds
    pos = get_approximate_state(pos, bounds)

    index = pos[0] * grid_size + pos[1]

    return policy_gene[index]

# Approximate stat function
def get_approximate_state(position, bounds):
    """ 
    Conversion from position to approximate state in a 3x3 grid
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
    #for (key, value) in visits.items():
    #    print(f"{key}: {value}")

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
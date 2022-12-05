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
    global y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4

    # Hyperparameters
    probability_crossover = 0.5
    probability_mutation = 0.7
    population_size = 100 
    grid_size = 5
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

    # Bookkeeping
    generational_stats = []
    
    initial_population = initialize_population(population_size, grid_size, playbook_size)
    max_fitness, max_fitness_gene, generational_stats, population, final_fitness, final_population = \
        genetic_algorithm(model, initial_population, time_limit, probability_crossover, probability_mutation, generational_stats, fitness_sims)
    
    policy_gene = max_fitness_gene
    print(f"Max Fitness: {max_fitness} Gene: {max_fitness_gene}")
    print(f"Final Fitnesses: {final_fitness} Avg: {sum(final_fitness)/len(final_fitness)}")

    return policy
 
# Initialize the population
def initialize_population(population_size, grid_size, playbook_size):
    population = []
    for i in range(0, population_size):
        gene = []
        for j in range(0, playbook_size * grid_size * grid_size):
            gene.append(j)
        random.shuffle(gene)
        population.append(gene)
    
    return population
        

def genetic_algorithm(model, population, time_left, probability_crossover, probability_mutation, generational_stats, fitness_sims):
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

        # Take our population of ants and get the top 20% for later
        elite1_index, elite2_index = select_max(fitness)
        #print(f"Elite1 {elite1_index, fitness[elite1_index]} Elite2 {elite2_index, fitness[elite2_index]}")

        # Select the two for crossover and mutation (Two Random Rank)
        parent1, parent2 = select(population, fitness)

        # Crossover the selected  with a 2 point crossover (split into thirds, alternate between parents.)
        crossed1, crossed2 = crossover(parent1, parent2, probability_crossover)

        # Grab the information for the elites
        elite1_gene = population[elite1_index].copy()
        elite2_gene = population[elite2_index].copy()

        # Create new population by culling random 2 and replace with the crossed over ones, keep the elite ones in as they were
        cull1_index, cull2_index = cull_random(population, fitness)
        population[cull1_index] = crossed1
        population[cull2_index] = crossed2

        # Mutate the entire population by potentially increasing (wrap around) digits 2 and 3 of every single state. (chance for random reset of 0
        # as well)
        population = mutate_population(population, probability_mutation)
       
        # Put back in the the elites
        population[elite1_index] = elite1_gene
        population[elite2_index] = elite2_gene

        # Remove episode time from the allotted time
        end_time = time.time()
        time_left -= (end_time-start_time)
        gen_count += 1

    return max_fitness, max_fitness_gene, generational_stats, population, final_fitness, final_population

# Get fitness from simulating against a model
def get_fitness(model, gene, policy, fitness_sims):
    global policy_gene
    policy_gene = gene

    return model.simulate(policy, fitness_sims)

# Perform singlepoint crossover based on crossover probability, with points being at the separation of the state action brackets
def crossover(parent1, parent2, probability_crossover):

    # If we hit the probability, then crossover on the parents
    random_val = random.random()
    if random_val < probability_crossover:

        # pick a random division for the crossover
        divider = random.randrange(1, (grid_size*grid_size) - 1)

        swap = 0
        cross_point = (divider * playbook_size * grid_size) - 1
        print(cross_point)
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
    for gene in population:
        random_val = random.random()

        # If we hit the probabilty, mutate
        if random_val < probability_mutation:

            #Get random index to add to and subtract from
            rand_add = random.randrange(len(gene))
            rand_sub = random.randrange(len(gene))

            gene[rand_add] += 1
            gene[rand_sub] -= 1

               
    return population
       
# Select and return two via roulette, with probability of choice weighted on their fitness in the old generation
def select(old_gen, fitness):
    old_gen1 = old_gen.copy()
    fitness1 = fitness.copy()

    # Get the first gene
    # Normalize the weights
    min_value = min(fitness1)
    fitness2 = list(map(lambda x: (x + abs(min_value) + 1), fitness1))

    gene1 = random.choices(old_gen1, weights=fitness2, k=1)
    gene1 = gene1[0]

    # Get the index from first gene and  remove the fitness of it from fitness, and the gene from oldgen to get the second
    gene1_index = old_gen1.index(gene1)
    del(old_gen1[gene1_index])
    del(fitness2[gene1_index])

    # Get the second
    #print(f"Fitness {fitness}")
    gene2 = random.choices(old_gen1, weights=fitness2, k=1)
    gene2 = gene2[0] 

    return gene1, gene2

# Select and return two best indexes in the old generation for elitism later. Remember their indexes
def select_max(fitness):
    fitness1 = fitness.copy()

    # Get the first index
    max1 = (max(fitness1))
    index1 = fitness.index(max1) 

    # Remove first by setting it to 0 and get second best index
    del(fitness1[index1])
    max2 = (max(fitness1))
    index2 = fitness.index(max2) 

    return index1, index2

# Cull random 2 in population (keeping the elite values) and then add in the mutated. This produces the next generation.
def cull_random(population, fitness):

    # Inverse the fitness so we can weigh bad ones higher, +1 for smoothing of weights
    inverse = fitness.copy()
    population1 = population.copy()
    max_inverse = max(inverse)
    inverse = list(map(lambda x: max_inverse - x + 1, inverse))

    # Cull two random ones weighted towards the bad ones.
    to_cull1 = random.choices(population1, weights=inverse, k = 1)
    to_cull1 = to_cull1[0]
    to_cull1_index = population.index(to_cull1)
    del(inverse[to_cull1_index])
    del(population1[to_cull1_index])

    to_cull2 = random.choices(population1, weights=inverse, k = 1)
    to_cull2 = to_cull2[0]
    to_cull2_index = population.index(to_cull2)

    # Return Indexes to replace
    return to_cull1_index, to_cull2_index

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


    start_slice = pos[0] * grid_size + pos[1] * playbook_size
    stop_slice = start_slice + playbook_size
    state = slice(start_slice, stop_slice) 

    possible_actions = policy_gene[state]


    return argmax(possible_actions)

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

import nfl_strategy as nfl
import time
import random
import genetic


# Set partitioning divisions
max_yards_to_score = 80
max_yards_to_down = 10

y_1 = (max_yards_to_score * .03125) 
y_2 = (max_yards_to_score * .05) 

x_1 = (max_yards_to_down * .24)
x_2 = (max_yards_to_down * .4)

# Set global bounds
bounds = [y_1, y_2, x_1, x_2]

visits = {}
visits['x_0, y_0'] = 0
visits['x_0, y_1'] = 0
visits['x_0, y_2'] = 0
visits['x_1, y_0'] = 0
visits['x_1, y_1'] = 0
visits['x_1, y_2'] = 0
visits['x_2, y_0'] = 0
visits['x_2, y_1'] = 0
visits['x_2, y_2'] = 0
bad = 0

state_actions = []
grid_size = 0
playbook_size = 0

# Q learning function
def ga_learn(model: nfl.NFLStrategy, time_limit):
    """ 
    Standard genetic algorithm
    Pre: 
    Post:
    """
    
    # Global variables
    global visits
    global bad
    global grid_size
    global state_actions
    global playbook_size

    # Hyperparameters
    probability_crossover = 0.2
    probability_mutation = 0.5
    initial_population_size = 4
    grid_size = 3
    playbook_size = model.offensive_playbook_size()

    # Bookkeeping
    generational_stats = []
    
    # Keep track of time limit
    time_left = time_limit

    initial_population = genetic.initialize_population(initial_population_size, grid_size, playbook_size)
    max_fitness, max_fitness_gene, max_fitness_trial, population, final_fitness, final_population, winners = \
        genetic.genetic_algorithm(model, initial_population, time_limit, probability_crossover, probability_mutation, generational_stats)
    
    while time_left > 0.2:

        # Set a minimum bound for epsilon
        if epsilon < 0.1:
            epsilon = 0.1

        # Get the time before this episode begins
        start_time = time.time()

        # Get the initial position
        current_position = model.initial_position()

        # While this episode is ongoing
        while(not model.game_over(current_position)):

            # Convert the position to the approximate state
            state = get_approximate_state(current_position, bounds)
            increment_visits(*state)

            # Epsilon greedy choice of an action
            if random.uniform(0, 1) < epsilon:
                action = random.choice(list(range(0, model.offensive_playbook_size()))) # Explore
            else:
                action = argmax(q_table[state]) # Exploit learned values
            
            # Get the new position / reward / approximate state from taking our e-greedy action
            new_position, reward_shaping = model.result(current_position, action)
            new_state = get_approximate_state(new_position, bounds)
            reward = get_reward(model, new_position, reward_shaping)
            
            old_value = q_table[state][action] # Keep track of old value for calculations
            
            # Adujust alpha if visists % 10 = 0 for that state action pair
            sa_visits[state][action] = sa_visits[state][action] + 1
            if sa_visits[state][action] % 100 == 0:
                learning_rates[state][action] *= learning_rate_decay
            
            # Set the learning rate for whatever SA we chose
            learning_rate = learning_rates[state][action]

            # If we are terminal, expected reward from next state is 0.
            # Else, its max of the next states actions
            if model.game_over(new_position):
                next_max = 0 
            else:
                next_max = max(q_table[new_state])

            # Set our new q value and update table
            new_value = (1-learning_rate) * old_value + learning_rate * (reward + discount * next_max)
            q_table[state][action] = new_value

            # update current position
            current_position = new_position

        # Remove episode time from the allotted time
        end_time = time.time()
        time_left -= (end_time-start_time)
    print_visits(visits)

    return policy

def print_visits(visits):
    global bounds
    sum_x0 = visits['x_0, y_0'] + visits['x_0, y_1'] + visits['x_0, y_2']
    sum_x1 = visits['x_1, y_0'] + visits['x_1, y_1'] + visits['x_1, y_2']
    sum_x2 = visits['x_2, y_0'] + visits['x_2, y_1'] + visits['x_2, y_2']
    sum_y0 = visits['x_0, y_0'] + visits['x_1, y_0'] + visits['x_2, y_0']
    sum_y1 = visits['x_0, y_1'] + visits['x_1, y_1'] + visits['x_2, y_1']
    sum_y2 = visits['x_0, y_2'] + visits['x_1, y_2'] + visits['x_2, y_2']
    print(f"x_2 > {bounds[3]} x_1 > {bounds[2]} y_2 > {bounds[1]} y_1 > {bounds[0]}")
    print(f"Total x = {sum_x0+sum_x1+sum_x2} division should be {(sum_x0+sum_x1+sum_x2)/3}")
    print(f"Sum x0: {sum_x0}")
    print(f"Sum x1: {sum_x1}")
    print(f"Sum x2: {sum_x2}")
    print("\n")
    print(f"Total y = {sum_y0+sum_y1+sum_y2} division should be {(sum_y0+sum_y1+sum_y2)/3}")
    print(f"Sum y0: {sum_y0}")
    print(f"Sum y1: {sum_y1}")
    print(f"Sum y2: {sum_y2}")
    for (key, value) in visits.items():
        print(f"{key}: {value}")

def increment_visits(y,x):
    global visits
    if x == 0 and y == 0:
        visits['x_0, y_0'] += 1
    elif x == 0 and y == 1:
        visits['x_0, y_1'] += 1
    elif x == 0 and y == 2:
        visits['x_0, y_2'] += 1
    elif x == 1 and y == 0:
        visits['x_1, y_0'] += 1
    elif x == 1 and y == 1:
        visits['x_1, y_1'] += 1
    elif x == 1 and y == 2:
        visits['x_1, y_2'] += 1
    elif x == 2 and y == 0:
        visits['x_2, y_0'] += 1
    elif x == 2 and y == 1:
        visits['x_2, y_1'] += 1
    elif x == 2 and y == 2:
        visits['x_2, y_2'] += 1
 
# Policy function
def policy(pos):
    """ 
    Policy for choosing and action from our learned model
    Pre: a position in the game, an initialized q_table, and bounds
    Post: action to take
    """
    global state_actions
    global grid_size
    global playbook_size
    pos = get_approximate_state(pos, bounds)

    state = slice(pos[0]*grid_size + pos[1], playbook_size) 
    possible_actions = state_actions[state]
    return argmax(possible_actions)

# Approximate stat function
def get_approximate_state(position, bounds):
    """ 
    Conversion from position to approximate state in a 3x3 grid
    Pre: a position in the game, and bounds
    Post: approximation of state 
    """
    # Unwrap the bounds and position
    y_1, y_2, x_1, x_2 = bounds
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

# Reward function + basic reward shaping
def get_reward(model, position, reward_shaping):
    """ 
    Get the reward using simple reward shaping
    Pre: a model, position, and reward_shaping values
    Post: reward for the current position and result of taking that action
    """
    # Unwrap reward shaping portion of result
    yards_gained, elapsed_time, game_losing = reward_shaping

    # Initialize reward to be proportional to moving forwards efficiently
    reward = yards_gained / elapsed_time

    # Discourage risky behavior
    if game_losing:
        reward -= 10

    # Encourage winning :)
    if model.win(position):
        reward += 300
    
    return reward





import nfl_strategy as nfl
import time
import random

# Initialize global variables
q_table = {} 
learning_rates = {} 
sa_visits = {} 

# Set partitioning divisions
max_yards_to_score = 80
max_yards_to_down = 10

# Set global bounds
y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4 = 0, 0, 0, 0, 0, 0, 0, 0
bounds = [y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4]
grid_size = 0

# For visit counts, can ignore, used in debugging
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
def q_learn(model: nfl.NFLStrategy, time_limit, grid):
    """ 
    Standard approximate q learning function
    Pre: a model of nfl game
    Post: a table of approximate state action pairs with expected rewards
    """
    
    # Global variables
    global q_table
    global learning_rates
    global sa_visits
    global bounds
    global y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4
    global grid_size

    # Hyperparameters
    learning_rate = 0.2
    learning_rate_decay = 0.5
    original_epsilon = 0.3
    discount = 0.99
    grid_size = grid
    
    # If 3x3
    if grid_size == 3:
        y_1 = (max_yards_to_score * .03125) 
        y_2 = (max_yards_to_score * .05) 

        x_1 = (max_yards_to_down * .24)
        x_2 = (max_yards_to_down * .4)
    
    # If 4x4
    elif grid_size == 4:
        y_1 = (max_yards_to_score * .03125) 
        y_2 = (max_yards_to_score * .037) 
        y_3 = (max_yards_to_score * .045) 

        x_1 = (max_yards_to_down * .22)
        x_2 = (max_yards_to_down * .3)
        x_3 = (max_yards_to_down * .4)

    # Q learning does .46
    # If 5x5
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
    # Initialize the q table to 0 Cell y,x, state action pair visits to 0, and
    # the learning rates for each division to the given learning rate
    # [0] Cell 00
    # [1] Cell 01
    # [2] Cell 02
    # [3] Cell 10
    # [4] Cell 11
    # [5] Cell 12
    # [6] Cell 20
    # [7] Cell 21
    # [8] Cell 22
    for i in range (0, grid_size):
        for j in range (0, grid_size):
            actions = []
            alphas = []
            sa_vis = []
            for play in range(0, model.offensive_playbook_size()):
                actions.append(0) 
                alphas.append(learning_rate)
                sa_vis.append(0) 
            q_table[(i,j)] = actions
            learning_rates[(i,j)] = alphas
            sa_visits[(i,j)] = sa_vis
    
    # Keep track of time limit
    time_left = time_limit

    # While there is still time left to train
    while time_left > 0.2:

        # Decrease epsilon as we learn more, to slow the rate of exploration
        epsilon = original_epsilon * (time_left / time_limit) 
        
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
            #increment_visits(*state)

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
    
    #print_visits(visits) # Old used for partitions

    return policy

# Policy function
def policy(pos):
    """ 
    Policy for choosing and action from our learned model
    Pre: a position in the game, an initialized q_table, and bounds
    Post: action to take
    """
    global q_table
    global bounds
    pos = get_approximate_state(pos, bounds)

    return argmax(q_table[pos])

# Approximate stat function
def get_approximate_state(position, bounds):
    """ 
    Conversion from position to approximate state in a 3x3 grid
    Pre: a position in the game, and bounds
    Post: approximation of state 
    """
    global grid_size
    y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4= bounds

    # Unwrap the bounds and position
    yards_to_score, downs_left, distance_to_down, time_left = position
    if grid_size == 3: 

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
    
    # if 5x5
    elif grid_size == 4: 

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
    
    # If 5x5
    elif grid_size == 5: 
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


""" ------------------------OLD FUNCTIONS FOR DEBUGGING----------------------------------------"""
# Old printing function for debugging.
def print_visits(visits):
    global bounds
    sum_x0 = visits['x_0, y_0'] + visits['x_0, y_1'] + visits['x_0, y_2'] + visits['x_0, y_3']
    sum_x1 = visits['x_1, y_0'] + visits['x_1, y_1'] + visits['x_1, y_2'] + visits['x_1, y_3']
    sum_x2 = visits['x_2, y_0'] + visits['x_2, y_1'] + visits['x_2, y_2'] + visits['x_2, y_3']
    sum_x3 = visits['x_3, y_0'] + visits['x_3, y_1'] + visits['x_3, y_2'] + visits['x_3, y_3']
    sum_y0 = visits['x_0, y_0'] + visits['x_1, y_0'] + visits['x_2, y_0'] + visits['x_3, y_0']
    sum_y1 = visits['x_0, y_1'] + visits['x_1, y_1'] + visits['x_2, y_1'] + visits['x_3, y_1']
    sum_y2 = visits['x_0, y_2'] + visits['x_1, y_2'] + visits['x_2, y_2'] + visits['x_3, y_2']
    sum_y3 = visits['x_0, y_3'] + visits['x_1, y_3'] + visits['x_2, y_3'] + visits['x_3, y_3']

    print(f"x_3 > {bounds[5]} x_2 > {bounds[4]} x_1 > {bounds[3]} y_3 > {bounds[2]} y_2 > {bounds[1]} y_1 > {bounds[0]}")
    print(f"Total x = {sum_x0+sum_x1+sum_x2+sum_x3} division should be {(sum_x0+sum_x1+sum_x2+sum_x3)/4}")
    print(f"Sum x0: {sum_x0}")
    print(f"Sum x1: {sum_x1}")
    print(f"Sum x2: {sum_x2}")
    print(f"Sum x3: {sum_x3}")
    print("\n")
    print(f"Total y = {sum_y0+sum_y1+sum_y2+sum_y3} division should be {(sum_y0+sum_y1+sum_y2+sum_y3)/4}")
    print(f"Sum y0: {sum_y0}")
    print(f"Sum y1: {sum_y1}")
    print(f"Sum y2: {sum_y2}")
    print(f"Sum y3: {sum_y3}")
    #for (key, value) in visits.items():
    #    print(f"{key}: {value}")

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







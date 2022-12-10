# I am using a genetic algorithm to play NFL strategy. 
# This test script runs the same as the QFL script,
# however, there is an additional parameter before the
# number of evaluation games. This parameter is the number
# of simulation games to test each member of the population.

# Hyperparams Default:
# Grid Size: 3x3
# Crossover Prob: 0.8
# Mutation Prob: 0.5
# Elite/Cull: 10%
# Pop Size:: 100
GFL:
	echo "#!/bin/bash" > GFL
	echo "pypy3 test_gfl.py \"\$$@\"" >> GFL
	chmod u+x Mcts

# This test script will run it over each model,
# grid size, and for a certain number of tests
# to average results and get std dev.
# args: time_limit, evaluation_games, ga_fitness_games, aggregate_tests
GFL_RESULTS:
	echo "#!/bin/bash" > GFL_RESULTS
	echo "pypy3 gfl_results.py \"\$$@\"" >> GFL_RESULTS
	chmod u+x Mcts

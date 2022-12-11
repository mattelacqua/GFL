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

# Example: 
# ./GFL 0 9 1000 250000
# 0.500824
#
GFL:
	echo "#!/bin/bash" > GFL
	echo "pypy3 test_gfl.py \"\$$@\"" >> GFL
	chmod u+x GFL

# This test script will run it over each model,
# grid size, and for a certain number of tests
# to average results and get std dev.
# args: time_limit, evaluation_games, ga_fitness_games, aggregate_tests
# Example: (results will vary from low evaluation time, but this is just
# example of running the aggregate results.)
# Model No: 0 Grid Size: 3 Iterations: 10: Seconds: 1.0 Playout games: 1000 
# Average: 0.43679999999999997, Min: 0.304 Max: 0.51 StdDev: 0.06359909153788633
#
# Model No: 0 Grid Size: 4 Iterations: 10: Seconds: 1.0 Playout games: 1000
# Average: 0.4191999999999999, Min: 0.363 Max: 0.473 StdDev: 0.03549272225870162
#
# Model No: 0 Grid Size: 5 Iterations: 10: Seconds: 1.0 Playout games: 1000
# Average: 0.39540000000000003, Min: 0.327 Max: 0.483 StdDev: 0.05121024203114928
#
# Model No: 1 Grid Size: 3 Iterations: 10: Seconds: 1.0 Playout games: 1000
# Average: 0.3837, Min: 0.207 Max: 0.48 StdDev: 0.0879015231823532
#
# Model No: 1 Grid Size: 4 Iterations: 10: Seconds: 1.0 Playout games: 1000
# Average: 0.32180000000000003, Min: 0.224 Max: 0.452 StdDev: 0.06930416052926885
#
# Model No: 1 Grid Size: 5 Iterations: 10: Seconds: 1.0 Playout games: 1000
# Average: 0.2771, Min: 0.186 Max: 0.345 StdDev: 0.05328424824571621
GFL_RESULTS:
	echo "#!/bin/bash" > GFL_RESULTS
	echo "pypy3 gfl_results.py \"\$$@\"" >> GFL_RESULTS
	chmod u+x GFL_RESULTS

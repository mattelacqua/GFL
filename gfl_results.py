import time
import sys
import random
import nfl_strategy as nfl

import gfl
import statistics

# defenses are 4-3 Blast Man-to-man, 4-3 Sam-Will Blitz, Prevent
# outcomes are (yards, time, turnover)

_slant_37 = [[(-1, 4, False), (20, 3, False), (10, 2, False), (7, 2, False), (4, 2, False)],
           [(1, 3, False), (10, 2, False), (5, 2, False), (1, 4, False), (2, 2, False)],
           [(8, 2, False), (20, 3, False), (4, 2, False), (2, 2, False), (6, 2, False)]]

_y_cross_80 = [[(0, 1, True), (-7, 4, False), (0, 1, False), (0, 1, True), (24, 2, False)],
              [(0, 1, False), (0, 1, True), (-6, 4, False), (18, 2, False), (13, 2, False)],
              [(19, 4, False), (0, 1, True), (22, 2, False), (47, 3, False), (0, 1, False)]]

_z_corner_82 = [[(28, 3, False), (-9, 4, False), (47, 3, False), (0, 1, False), (0, 1, False)],
               [(30, 3, False), (0, 1, True), (8, 2, False), (-8, 4, False), (0, 1, False)],
               [(38, 3, False), (-9, 4, False), (0, 1, False), (0, 1, True), (0, 1, False)]]

game_parameters = [
    ([_slant_37, _y_cross_80, _z_corner_82], [0.20, 0.05, 0.25, 0.1, 0.4]),
    ([_y_cross_80, _z_corner_82, _slant_37], [0.05, 0.20, 0.1, 0.25, 0.4])
    ]


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("USAGE: {python3 | pypy3}", sys.argv[0], "learning-time num-games num-test num-ga-test")
        sys.exit(1)

    try:
        limit = float(sys.argv[1])
        n = int(sys.argv[2])
        iterations = int(sys.argv[3])
        fit_n = int(sys.argv[4])
    except:
        print("USAGE: {python3 | pypy3}", sys.argv[0], "learning-time num-games num-test num-ga-test")
        sys.exit(1)


    for model_no in range(0,2):
        for grid_size in range(3, 6):
            values = []
            for x in range(0, iterations):
                model = nfl.NFLStrategy(*game_parameters[model_no])
                start = time.time();
                policy = gfl.ga_learn(model, limit, fit_n, grid_size);
                t = time.time() - start
                if t > limit:
                    print("WARNING: Genetic Algorithm ran for", t, "seconds; allowed", limit)
                values.append(model.simulate(policy, n))
                #print(f"Iteration {x}: {values[x]} Running Average:{sum(values)/len(values)}")
            #print(values)
            print(f"Model No: {model_no} Grid Size: {grid_size} Iterations: {iterations}: Seconds: {limit} Playout games: {n}")
            print(f"Average: {sum(values)/len(values)}, Min: {min(values)} Max: {max(values)} StdDev: {statistics.stdev(values)}\n")
    

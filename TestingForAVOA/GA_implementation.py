import time
import pandas as pd

from mealpy.swarm_based import AVOA
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.tuner import Tuner
import numpy as np
import random


def F1(X):
    output = sum(np.square(X))
    return output


def F2(X):
    output = np.sum(np.abs(X)) + np.prod(np.abs(X))
    return output


def F3(X):
    output = sum(np.square(X[j]) for i in range(len(X)) for j in range(i + 1))
    return output


def F4(X):
    output = max(np.abs(X))
    return output


def F5(X):
    output = sum(100 * np.square(X[i + 1] - np.square(X[i])) + np.square(X[i] - 1) for i in range(len(X) - 1))
    return output


def F6(X):
    output = sum(np.square(abs(i + 0.5)) for i in X)
    return output


def F7(X):
    output = sum((i + 1) * pow(X[i], 4) for i in range(len(X))) + random.random()
    return output


def F8(X):
    output = sum(-(X * np.sin(np.sqrt(np.abs(X)))))
    return output


def F9(X):
    output = 10 * len(X) + sum(pow(X[i], len(X)) - 10 * np.cos(2 * np.pi * X[i]) for i in range(len(X)))
    return output


def F10(X):
    output = -20 * np.exp(-0.2 * np.sqrt(1 / len(X) * sum(np.square(X)))) - np.exp(
        1 / len(X) * sum(np.cos(2 * np.pi * X[i]) for i in range(len(X)))) + 20 + np.exp(1)
    return output


def F11(X):
    output = 1 / 4000 * sum(np.square(X)) - np.prod(
        [np.cos(a / (b + 1) ** (1 / 2)) for a, b in zip(X, list(range(len(X))))]) + 1
    return output


def F12(X):
    a = 10
    k = 100
    m = 4
    pt1 = 0
    for i in X:
        if i > a:
            pt1 += k * pow((i - 1), m)
        elif -a <= i <= a:
            pt1 += 0
        else:
            pt1 += k * pow((-i - 1), m)
    pt2 = np.pi / len(X) * (10 * pow(np.sin(np.pi * (1 + 1 / 4 * (X[0] + 1))), 2) + sum(
        pow((1 + 1 / 4 * (X[i] + 1) - 1), 2) * (1 + 10 * pow(np.sin(np.pi * (1 + 1 / 4 * (X[i + 1] + 1))), 2)) for i in
        range(len(X) - 1)) + pow((1 + 1 / 4 * (X[len(X) - 1])), 2))
    output = pt1 + pt2
    return output


def F13(X):
    a = 5
    k = 100
    m = 4
    pt1 = 0
    for i in X:
        if i > a:
            pt1 += k * pow((i - 1), m)
        elif -a <= i <= a:
            pt1 += 0
        else:
            pt1 += k * pow((-i - 1), m)
    pt2 = 0.1 * (pow(np.sin(3 * np.pi * (X[0])), 2) + sum(
        pow(X[i] - 1, 2) * (1 + pow(np.sin(3 * np.pi * X[i] + 1), 2)) for i in range(len(X))) + pow(X[len(X) - 1],
                                                                                                    2) * (
                         1 + pow(np.sin(2 * np.pi * X[len(X) - 1]), 2)))
    return pt1 + pt2


# epoch = 1000
# pop = 50
# pc = 0.85
# pm = 0.15

# paras = {
#     "epoch": [1000],
#     "pop_size": [50],
#     "pc": [0.85, 0.75, 0.65],
#     "pm": [0.15, 0.25, 0.05],
# }

paras = {
    "epoch": [1000],
    "pop_size": [50],
    "p1": [0.4, 0.6, 0.8],
    "p2": [0.4, 0.6, 0.8],
    "p3": [0.4, 0.6, 0.8],
    "alpha": [0.6, 0.8, 1.0],
    "gama": [1.0, 2.5, 3.0]
}

if __name__ == "__main__":

    replication = 10
    dim = 100
    result_list = []
    runtime_list_out = []

    for i in range(0,1):
        best_pos_array = []
        best_fit_array = []
        runtime_array = []
        if i == 0:
            problem = {
                "lb": [-100, ] * dim,
                "ub": [100, ] * dim,
                "minmax": "min",
                "fit_func": F1,
                "name": "benchmark 1",
            }
        elif i == 1:
            problem = {
                "lb": [-10, ] * dim,
                "ub": [10, ] * dim,
                "minmax": "min",
                "fit_func": F2,
                "name": "benchmark 2",
            }
        elif i == 2:
            problem = {
                "lb": [-100, ] * dim,
                "ub": [100, ] * dim,
                "minmax": "min",
                "fit_func": F3,
                "name": "benchmark 3",
            }
        elif i == 3:
            problem = {
                "lb": [-100, ] * dim,
                "ub": [100, ] * dim,
                "minmax": "min",
                "fit_func": F4,
                "name": "benchmark 4",
            }
        elif i == 4:
            problem = {
                "lb": [-30, ] * dim,
                "ub": [30, ] * dim,
                "minmax": "min",
                "fit_func": F5,
                "name": "benchmark 5",
            }
        elif i == 5:
            problem = {
                "lb": [-100, ] * dim,
                "ub": [100, ] * dim,
                "minmax": "min",
                "fit_func": F6,
                "name": "benchmark 6",
            }
        elif i == 6:
            problem = {
                "lb": [-128, ] * dim,
                "ub": [128, ] * dim,
                "minmax": "min",
                "fit_func": F7,
                "name": "benchmark 7",
            }
        elif i == 7:
            problem = {
                "lb": [-500, ] * dim,
                "ub": [500, ] * dim,
                "minmax": "min",
                "fit_func": F8,
                "name": "benchmark 8",
            }
        elif i == 8:
            problem = {
                "lb": [-5.12, ] * dim,
                "ub": [5.12, ] * dim,
                "minmax": "min",
                "fit_func": F9,
                "name": "benchmark 9",
            }
        elif i == 9:
            problem = {
                "lb": [-32, ] * dim,
                "ub": [32, ] * dim,
                "minmax": "min",
                "fit_func": F10,
                "name": "benchmark 10",
            }
        elif i == 10:
            problem = {
                "lb": [-600, ] * dim,
                "ub": [600, ] * dim,
                "minmax": "min",
                "fit_func": F11,
                "name": "benchmark 11",
            }
        elif i == 11:
            problem = {
                "lb": [-50, ] * dim,
                "ub": [50, ] * dim,
                "minmax": "min",
                "fit_func": F12,
                "name": "benchmark 12",
            }
        elif i == 12:
            problem = {
                "lb": [-50, ] * dim,
                "ub": [50, ] * dim,
                "minmax": "min",
                "fit_func": F13,
                "name": "benchmark 13",
            }

        time_start = time.time()
        # model = BaseGA()
        model = AVOA.OriginalAVOA()
        tuner = Tuner(model, paras)
        tuner.execute(problem=problem, n_trials=10, mode="parallel", n_workers=4)
        time_end = time.time()
        runtime = time_end - time_start
        best_position, best_fitness = model.solve(problem)
        best_pos_array.append(best_position)
        best_fit_array.append(best_fitness)
        runtime_array.append(runtime)
        # print(f"Best Fitness: {best_fitness}; Runtime: {runtime}")
        print(tuner.best_score)
        print(tuner.best_params)
        print(tuner.best_algorithm)
        print(tuner.best_algorithm.get_name())

    # for j in range(replication):
    #     time_start = time.time()
    #     model = BaseGA(epoch, pop, pc, pm, selection="roulette", crossover="multi_points")
    #     time_end = time.time()
    #     runtime = time_end - time_start
    #     best_position, best_fitness = model.solve(problem)
    #     best_pos_array.append(best_position)
    #     best_fit_array.append(best_fitness)
    #     runtime_array.append(runtime)
    #     print(f"Best Fitness: {best_fitness}; Runtime: {runtime}")
    # print(f'Benchmark: F {i + 1}\n')
    # print(f"Solution Final Output: \n\tnumber of replications={replication}\n\t"
    #       f"Best of Gbests={np.min(best_fit_array)}\n\tWorst of Gbests={np.max(best_fit_array)}\n\taverage of Gbests={np.average(best_fit_array)}\n\tSTD of Gbests={np.std(best_fit_array)}")
    # print(
    #     f"Runtime Final Output: \n\tnumber of replications={replication}\n\taverage of Runtime={np.average(runtime_array)}\n\t"
    #     f"Best of Runtime={np.min(runtime_array)}\n\tWorst of Runtime={np.max(runtime_array)}\n\tSTD of Runtime={np.std(runtime_array)}")
    #
    # temp_list_result = []
    # temp_list_runtime = []
    # str_func = 'F' + str(i + 1)
    # temp_list_result.append(str_func)
    # mean = np.mean(best_fit_array)
    # standard_deviation = np.std(best_fit_array)
    # distance_from_mean = abs(best_fit_array - mean)
    # max_deviations = 3
    # not_outlier = distance_from_mean < max_deviations * standard_deviation
    # no_outliers = np.array(best_fit_array)[not_outlier]
    # no_outliers_list = no_outliers.tolist()
    #
    # temp_list_result.append(mean)
    # temp_list_result.append(np.mean(no_outliers_list))
    # temp_list_result.append(np.median(best_fit_array))
    # temp_list_result.append(np.min(best_fit_array))
    # temp_list_result.append(np.max(best_fit_array))
    # temp_list_result.append(standard_deviation)
    #
    # temp_list_runtime.append(str_func)
    # temp_list_runtime.append(np.mean(runtime_array))
    # temp_list_runtime.append(np.median(runtime_array))
    # temp_list_runtime.append(np.min(runtime_array))
    # temp_list_runtime.append(np.max(runtime_array))
    # temp_list_runtime.append(np.std(runtime_array))
    #
    # result_list.append(temp_list_result)
    # runtime_list_out.append(temp_list_runtime)

# print(np.array(result_list, dtype='object'))
# df_result = pd.DataFrame(np.array(result_list, dtype='object'),
#                          columns=['Benchmark', 'Mean', 'Mean_no_outlier', 'Median', 'Best', 'Worst', 'STD'])
# df_runtime = pd.DataFrame(np.array(runtime_list_out, dtype='object'),
#                           columns=['Benchmark', 'Mean', 'Median', 'Best', 'Worst', 'STD'])
#
# import os
#
# os.makedirs('Data', exist_ok=True)
# df_result.to_csv('Data/GA_result_output_100.csv')
# df_runtime.to_csv('Data/GA_runtime_output_100.csv')

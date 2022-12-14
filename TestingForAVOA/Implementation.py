from mealpy.swarm_based import AVOA
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.human_based.TLO import BaseTLO
from mealpy.utils.visualize import *
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


problem = {
    "lb": [-100, ]*30,
    "ub": [100, ]*30,
    "minmax": "min",
    "fit_func": F1,
    "name": "benchmark 1",
}

epoch = 1000
pop = 50
p1 = 0.6
p2 = 0.4
p3 = 0.6
alpha = 0.8
gama = 2.5

model = AVOA.OriginalAVOA(epoch, pop, p1, p2, p3, alpha, gama)
best_position, best_fitness = model.solve(problem)
print(f"Solution: {best_position}, Fitness: {best_fitness}")


export_convergence_chart(model.history.list_global_best_fit, title='Global Best Fitness')            # Draw global best fitness found so far in previous generations
export_convergence_chart(model.history.list_current_best_fit, title='Local Best Fitness')             # Draw current best fitness in each previous generation
export_convergence_chart(model.history.list_epoch_time, title='Runtime chart', y_label="Second")        # Draw runtime for each generation

export_explore_exploit_chart([model.history.list_exploration, model.history.list_exploitation])  # Draw exploration and exploitation chart

export_diversity_chart([model.history.list_diversity], list_legends=['GA'])        # Draw diversity measurement chart

global_obj_list = np.array([agent[1][1] for agent in model.history.list_global_best])     # 2D array / matrix 2D
global_obj_list = [global_obj_list[:,idx] for idx in range(0, len(global_obj_list[0]))]     # Make each obj_list as a element in array for drawing
export_objectives_chart(global_obj_list, title='Global Objectives Chart')

current_obj_list = np.array([agent[1][1] for agent in model.history.list_current_best])  # 2D array / matrix 2D
current_obj_list = [current_obj_list[:, idx] for idx in range(0, len(current_obj_list[0]))]  # Make each obj_list as a element in array for drawing
export_objectives_chart(current_obj_list, title='Local Objectives Chart')

# epoch = 1000
# pop_size = 50
# c1 = 2.05
# c2 = 2.05
# w_min = 0.4
# w_max = 0.9
# model = OriginalPSO(epoch, pop_size, c1, c2, w_min, w_max)
# best_position, best_fitness = model.solve(problem)
# print(f"Solution: {best_position}, Fitness: {best_fitness}")
#
# epoch = 1000
# pop_size = 50
# model = BaseTLO(epoch, pop_size)
# best_position, best_fitness = model.solve(problem)
# print(f"Solution: {best_position}, Fitness: {best_fitness}")

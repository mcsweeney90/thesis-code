#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set weights for Cholesky DAGs and save.
"""

import dill, pathlib
from timeit import default_timer as timer
import src 
import numpy as np

# =============================================================================
# Testing.
# =============================================================================

# mu = 100
# v = 0.5
# # data = np.random.gamma(1/v**2, mu*v**2, size=1000)
# data = [gamma(alpha=1/v**2, beta=mu*v**2) for _ in range(1000)]
# emp_mu = sum(data)/len(data)
# emp_sd = np.std(data)
# print(emp_mu, emp_sd)

# costs = np.random.gamma(100, [5 for i in range(1, 21)])
# print(costs)

# a = np.array([1, 2, 3, 4, 5])
# # b = np.array([1, 1, 1, 1, 1])
# # c = np.array([2, 2, 2, 2, 2])

# x = np.random.choice(a, p=a)
# print(x)



# =============================================================================
# Cholesky.
# =============================================================================

# dag_top_path = '../graphs/cholesky/topologies/'
# nt = 20
# with open('{}/{}.dill'.format(dag_top_path, nt), 'rb') as file:
#     top = dill.load(file)
# G = src.DAG(top)

# q, vmach, vnoise, ccr = 8, 0.1, 0.5, 1.0
# start = timer()
# G.set_cholesky_weights(nprocessors=q, vmach=vmach, vnoise=vnoise, ccr=ccr)
# elapsed = timer() - start
# print("Time to set costs: {}".format(elapsed))

# # # actual_ccr = G.ccr()
# # # print("Actual CCR: {}".format(actual_ccr))

# # # print(G.comm_costs)

# start = timer()
# mlb = G.makespan_lower_bound()
# elapsed = timer() - start
# print("\nMakespan lower bound: {}".format(mlb))
# print("Time to calculate lower bound: {}".format(elapsed))

# start = timer()
# mst = G.minimal_serial_time()
# elapsed = timer() - start
# print("\nMinimal serial time: {}".format(mst))
# print("Time to calculate MST: {}".format(elapsed))

# start = timer()
# hmkspan = src.heft(G, avg_type="HM")
# elapsed = timer() - start
# print("\nHEFT makespan: {}".format(hmkspan))
# print("SLR: {}".format(hmkspan/mlb))
# print("Speedup: {}".format(mst/hmkspan))
# print("Time for HEFT: {}".format(elapsed))

# start = timer()
# L, path_counts, criticalities = G.monte_carlo(realizations=1000, pmf="HM")
# elapsed = timer() - start
# print("\nTime for MC: {}".format(elapsed))

# # Criticalities as task priorities.
# start = timer()
# cmkspan = src.priority_scheduling(G,priorities=criticalities)
# elapsed = timer() - start
# print("\nCriticality prios makespan: {}".format(cmkspan))
# print("SLR: {}".format(cmkspan/mlb))
# print("Speedup: {}".format(mst/cmkspan))
# print("Time taken: {}".format(elapsed))

# # Mean values as task priorities.
# start = timer()
# means = {t: np.mean(L[t]) for t in G.top_sort}
# mkspan = src.priority_scheduling(G,priorities=means)
# elapsed = timer() - start
# print("\nMean prios makespan: {}".format(mkspan))
# print("SLR: {}".format(mkspan/mlb))
# print("Speedup: {}".format(mst/mkspan))
# print("Time taken: {}".format(elapsed))

# # UCB values as task priorities.
# start = timer()
# plus = {t: np.mean(L[t]) + np.std(L[t]) for t in G.top_sort}
# mkspan = src.priority_scheduling(G,priorities=plus)
# elapsed = timer() - start
# print("\nUCB-plus prios makespan: {}".format(mkspan))
# print("SLR: {}".format(mkspan/mlb))
# print("Speedup: {}".format(mst/mkspan))
# print("Time taken: {}".format(elapsed))

# # UCB minus.
# start = timer()
# minus = {t: np.mean(L[t]) - np.std(L[t]) for t in G.top_sort}
# mkspan = src.priority_scheduling(G, priorities=minus)
# elapsed = timer() - start
# print("\nUCB-minus prios makespan: {}".format(mkspan))
# print("SLR: {}".format(mkspan/mlb))
# print("Speedup: {}".format(mst/mkspan))
# print("Time taken: {}".format(elapsed))





# =============================================================================
# STG.
# =============================================================================

size = 100
dag_path = '../graphs/STG/{}'.format(size)

dname = 23
# Load the DAG topology.
with open('{}/{}.dill'.format(dag_path, dname), 'rb') as file:
    T = dill.load(file)
# Convert to DAG object.
G = src.DAG(T)

q = 8
# CNB.
rtask = 0.9
rmach = 0.9
mu = 10
v = 0.9
params = (rtask, rmach, mu, v)
G.set_random_weights(nprocessors=q, comp_method="CNB", comp_params=params, vband=0.2, muccr=1.0, vccr=0.2)


ccrs = []
for t in G.top_sort:
    comp_avg = G.task_average(t)
    children = list(G.graph.successors(t))
    if not children:
        continue
    s = children[0]
    edge_average = G.edge_average(parent=t, child=s)
    ccrs.append(edge_average/comp_avg)
    print(t, comp_avg, edge_average)

print("\n")
print(sum(ccrs)/len(ccrs))
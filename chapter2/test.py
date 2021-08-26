#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script.
"""

import dill, src
import pandas as pd
from timeit import default_timer as timer
from statistics import harmonic_mean

dag_path = '../graphs/cholesky/optimizing/'
ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

r = 32
ngpus = [1, 4]
nbs = [128, 1024]


# =============================================================================
# Autopsy.
# =============================================================================

nb, nt, s = 1024, 5, 4

with open('{}/nb{}/{}.dill'.format(dag_path, nb, nt), 'rb') as file:
    G = dill.load(file) 
    
d = G.edge_average(parent="T1", child="G1", r=r, s=s)
print(d)

# TRSM.
c, g = G.graph.nodes["T1"]['weight']["c"], G.graph.nodes["T1"]['weight']["g"]
# est_ccr = G.edge_average(parent="P1", child="T1", r=r, s=s) / src.average(c, g, r, s, avg_type="M")
est_ccr = d / src.average(c, g, r, s, avg_type="M")
print("Est. CCR (TRSM) : {}".format(est_ccr))

# GEMM.
c, g = G.graph.nodes["G1"]['weight']["c"], G.graph.nodes["G1"]['weight']["g"]
# est_ccr = G.edge_average(parent="T1", child="G1", r=r, s=s) / src.average(c, g, r, s, avg_type="M")
est_ccr = d / src.average(c, g, r, s, avg_type="M")
print("Est. CCR (GEMM) : {}".format(est_ccr))

# SYRK.
c, g = G.graph.nodes["S1"]['weight']["c"], G.graph.nodes["S1"]['weight']["g"]
# est_ccr = G.edge_average(parent="T1", child="S1", r=r, s=s) / src.average(c, g, r, s, avg_type="M")
est_ccr = d / src.average(c, g, r, s, avg_type="M")
print("Est. CCR (SYRK) : {}".format(est_ccr))

# POTRF.
c, g = G.graph.nodes["P2"]['weight']["c"], G.graph.nodes["P2"]['weight']["g"]
# est_ccr = G.edge_average(parent="S1", child="P2", r=r, s=s) / src.average(c, g, r, s, avg_type="M")
est_ccr = d / src.average(c, g, r, s, avg_type="M")
print("Est. CCR (POTRF) : {}".format(est_ccr))

# U = G.get_upward_ranks(r, s, avg_type="M") 
# heft, S = src.priority_scheduling(G, r, s, priorities=U, sel_policy="EFT", return_schedule=True)

# alpha = {}
# for worker, load in S.items():
#     for block in load:
#         alpha[block[0]] = "c" if worker < r else "g"        
# # print(alpha)            
# edge_delta = {"cc" : 0.0, "cg" : 1.0, "gc" : 1.0, "gg" : (s - 1)/s}
# ranks = {}
# backward_traversal = list(reversed(G.top_sort))
# for t in backward_traversal:
#     a = alpha[t]
#     ranks[t] = G.graph.nodes[t]['weight'][a]
#     try:
#         ranks[t] += max(edge_delta[alpha[t] + alpha[c]]*G.graph[t][c]['weight'] + ranks[c] for c in G.graph.successors(t))
#     except ValueError:
#         pass            
# aut, S1 = src.priority_scheduling(G, r, s, priorities=ranks, sel_policy="AMT", assignment=alpha, return_schedule=True)

# beta= {}
# for worker, load in S1.items():
#     for block in load:
#         beta[block[0]] = "c" if worker < r else "g" 
        
# # # Check they're the same...
# for t in G.top_sort:
#     if alpha[t] != beta[t]:
#         print("DIFFERENT, TASK {}".format(t))

# =============================================================================
#     
# =============================================================================

# s, ccr = 1, 10.0
# size = 100
# dag_path = '../graphs/STG/{}'.format(size)
# with open('{}/32.dill'.format(dag_path), 'rb') as file:
#     T = dill.load(file)
# G = src.DAG(T)
# G.set_random_weights(r=r, s=s, ccr=ccr)  

# mst = G.minimal_serial_time()

# U = G.get_upward_ranks(r, s, avg_type="M") 
# heft, S = src.priority_scheduling(G, r, s, priorities=U, sel_policy="EFT", return_schedule=True)

# gcp, S1 = src.priority_scheduling(G, r, s, priorities=U, sel_policy="GCP", return_schedule=True)

# print(mst, heft, gcp)

# src.summarize_schedule(S, r, s)
# src.summarize_schedule(S1, r, s)
# print(S1)
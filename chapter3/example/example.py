#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example graph used throughout.
"""

import dill
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt 

import sys
sys.path.append('../') 
from src import DAG, priority_scheduling, heft, cpop

# =============================================================================
# Global variables, functions, etc.
# =============================================================================

avgs = ["M", "MD", "B", "SB", "W", "SW", "HM", "SHM", "GM", "SGM", "R", "D", "NC", "SD"]

def get_random_graph(ntasks, single_source=True, single_sink=True):
    """TODO."""    

    G1 = nx.gnp_random_graph(ntasks, 0.1, directed=True)
    G = nx.DiGraph([(u,v) for (u,v) in G1.edges() if u<v])

    if single_source:
        sources = set(nd for nd in G if not list(G.predecessors(nd)))
        if len(sources) > 1:
            source = 0
            for nd in sources:
                if nd > 0:
                    G.add_edge(source, nd)
    if single_sink:
        sinks = set(nd for nd in G if not list(G.successors(nd)))
        if len(sinks) > 1:
            sink = ntasks - 1
            for nd in sinks:
                if nd < ntasks - 1:
                    G.add_edge(nd, sink) 
    return DAG(G)

G = nx.DiGraph()
info = {1:[2, 3, 4], 2:[5], 3:[5], 4:[5], 5:[6, 7, 8], 6:[9], 7:[9], 8:[9]}
for n, kids in info.items():
    for c in kids:
        G.add_edge(n, c)   

# =============================================================================
# Find weights that give the desired behaviour.
# =============================================================================

# D = DAG(G)
# # Set weights randomly until we get desired behaviour..
# for _ in range(1000):
#     D.set_example_weights(nprocessors=2)
#     # Compute averaged longest paths.
#     mkspans = {"MST":D.minimal_serial_time(), "MLB":D.makespan_lower_bound()}
#     # Averages.
#     for avg in avgs:
#         mkspan = heft(D, avg_type=avg)
#         mkspans[avg] = mkspan
#     if len(list(mkspans.values())) < 4:
#         continue
    
#     # Optimistic/pessimistic.
#     opt = D.optimistic_critical_path()
#     opt_ranks = {t : min(opt[t].values()) for t in D.top_sort}
#     mkspan = priority_scheduling(D, priorities=opt_ranks)
#     mkspans["OPT"] = mkspan
#     pes = D.optimistic_critical_path(pessimistic=True)
#     pes_ranks = {t : max(opt[t].values()) for t in D.top_sort}
#     mkspan = priority_scheduling(D, priorities=pes_ranks)
#     mkspans["PESS"] = mkspan
    
#     # Monte Carlo.
#     for pmf in ["M", "HM"]: 
#         L, path_counts, criticalities = D.monte_carlo(realizations=1000, pmf=pmf)
#         # Task criticalities.
#         mkspan = priority_scheduling(D, priorities=criticalities)
#         mkspans["{}-CR".format(pmf)] = mkspan
#         # Mean values.
#         means = {t: np.mean(L[t]) for t in D.top_sort}
#         mkspan = priority_scheduling(D, priorities=means)
#         mkspans["{}-MEAN".format(pmf)] = mkspan            
#         means10 = {t: np.mean(L[t][:10]) for t in D.top_sort}
#         mkspan = priority_scheduling(D, priorities=means10)
#         mkspans["{}10-MEAN".format(pmf)] = mkspan
#         # UCB.
#         ucbs = {t: np.mean(L[t]) + np.std(L[t]) for t in D.top_sort}
#         mkspan = priority_scheduling(D, priorities=ucbs)
#         mkspans["{}-UCB".format(pmf)] = mkspan            
#         ucbs10 = {t: np.mean(L[t][:10]) + np.std(L[t][:10]) for t in D.top_sort}
#         mkspan = priority_scheduling(D, priorities=ucbs10)
#         mkspans["{}10-UCB".format(pmf)] = mkspan 
    
#     if (mkspans["M"] > mkspans["M-MEAN"]) and (mkspans["HM"] > mkspans["HM-MEAN"]):
#         # Save the DAG and makespans
#         with open('example.dill', 'wb') as handle:
#             dill.dump(D, handle)
#         with open('mkspans.dill', 'wb') as handle:
#             dill.dump(mkspans, handle)
#         break      
        
    
# =============================================================================
# Get the saved graph.
# =============================================================================

with open('example.dill', 'rb') as file:
    D = dill.load(file)
with open('mkspans.dill', 'rb') as file:
    mkspans = dill.load(file)
# print(mkspans)
    
# =============================================================================
# Draw labelled graph.
# =============================================================================

# Draw graph the labelled graph.
# plt.figure(figsize=(14,14))

# pos = graphviz_layout(G, prog='dot')   
# # Draw the topology. 
# nx.draw_networkx_nodes(G, pos, node_color='#348ABD', node_size=2000, alpha=0.5)
# nx.draw_networkx_edges(G, pos, width=1.0)

# # Draw the node labels.
# nx.draw_networkx_labels(G, pos, font_size=18, font_weight='bold')

# # Draw the node weights.
# node_weights = {t : list(D.graph.nodes[t]['weight']) for t in D.top_sort}
# alt_pos = {}
# for p in pos:
#     if p in [1, 3, 5, 7, 9]:
#         alt_pos[p] = (pos[p][0] + 12, pos[p][1] )
#     else:
#         alt_pos[p] = (pos[p][0], pos[p][1] + 15)            
# nx.draw_networkx_labels(G, alt_pos, node_weights, font_size=18, font_weight='bold', font_color='#E24A33')

# # # Draw the edge weights.
# edge_weights = {}
# for t, s in D.graph.edges:
#     edge_weights[(t, s)] = list(D.graph[t][s]['weight'].values())[0]
# nx.draw_networkx_edge_labels(G, pos, font_size=18, edge_labels=edge_weights, font_weight='bold', font_color='#E24A33')

# plt.axis("off") 
# plt.savefig('example.png', bbox_inches='tight') 

# =============================================================================
# Get the critical paths.
# =============================================================================

# critical_paths = {}
# for avg in avgs:
#     U = D.get_upward_ranks(avg_type=avg)
#     W = D.get_downward_ranks(avg_type=avg)
#     ranks = {t : U[t] + W[t] for t in D.top_sort}   
                
#     # Identify a single critical path (unless all_critical_tasks) - randomly if there are multiple...
#     cp_length = ranks[D.top_sort[0]] # Single entry/exit task.   
#     ct = D.top_sort[0]
#     critical_path = [ct]
#     while True:
#         children = list(D.graph.successors(ct))
#         if not children:
#             break
#         for child in children:
#             if abs(ranks[child] - cp_length) < 1e-6:
#                 critical_path.append(child)
#                 ct = child
#                 break 
#     tpath = tuple(critical_path)
#     try:
#         critical_paths[tpath].append(avg)
#     except KeyError:
#         critical_paths[tpath] = [avg]

# print(critical_paths)

# for k, v in critical_paths.items():
#     plt.figure(figsize=(14,14))
#     pos = graphviz_layout(G, prog='dot')   
#     # Draw the topology. 
#     nx.draw_networkx_nodes(G, pos, node_color='#348ABD', node_size=2000, alpha=0.5)
#     nx.draw_networkx_edges(G, pos, width=1.0)    
#     crit_edges = []
#     for i, t in enumerate(k[1:]):
#         crit_edges.append((k[i], t))   
#     nx.draw_networkx_edges(G, pos, edgelist=crit_edges, edge_color='#E24A33', alpha=0.5, width=20.0)
#     # Draw the node labels.
#     nx.draw_networkx_labels(G, pos, font_size=18, font_weight='bold')
#     plt.axis("off") 
#     plt.savefig('{}.png'.format(v[0]), bbox_inches='tight') 

# =============================================================================
# Upward ranks.
# =============================================================================

# for avg in avgs:
#     U = D.get_upward_ranks(avg_type=avg)
#     print(avg, U)

# =============================================================================
# Get the priority list.
# =============================================================================

# priority_lists = {}
# for avg in avgs:
#     U = D.get_upward_ranks(avg_type=avg)
#     ready_tasks = []
#     prio_list = tuple(sorted(U, key=U.get, reverse=True))
#     try:
#         priority_lists[prio_list].append(avg)
#     except KeyError:
#         priority_lists[prio_list] = [avg]

# print(priority_lists)
# print(mkspans)

# =============================================================================
# Optimistic and pessimistic. 
# =============================================================================

# opt, path = D.optimistic_critical_path(return_path=True)
# opt_ranks = {t : min(opt[t].values()) for t in D.top_sort}
# print(opt)
# print(path)

# pess, path = D.optimistic_critical_path(return_path=True, pessimistic=True)
# pess_ranks = {t : max(opt[t].values()) for t in D.top_sort}
# print(pess)
# print(path)

# =============================================================================
# Monte Carlo.
# =============================================================================

# priorities = {}
# pmf_path_counts = {}
# for pmf in ["A", "H"]: 
#     for r in [10, 1000]:
#         L, path_counts, criticalities = D.monte_carlo(realizations=r, pmf=pmf)
#         priorities["{}-C-{}".format(pmf, r)] = criticalities
#         pmf_path_counts[(pmf, r)] = path_counts
#         # Mean values.
#         means = {t: np.mean(L[t]) for t in D.top_sort}
#         priorities["{}-M-{}".format(pmf, r)] = means   
#         # UCB.
#         ucbs = {t: np.mean(L[t]) + np.std(L[t]) for t in D.top_sort}
#         priorities["{}-U-{}".format(pmf, r)] = ucbs   

# mc_mkspans = {}
# for k, v in priorities.items():
#     mkspan = priority_scheduling(D, priorities=v)
#     mc_mkspans[k] = mkspan    
# print(mc_mkspans)

# print(pmf_path_counts)

# =============================================================================
# Save schedule (for Chapter 4).
# =============================================================================

# plt.style.use('ggplot')
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 10
# plt.rcParams['font.weight'] = 'bold' 

# U = D.get_upward_ranks(avg_type="SW")
# mkspan, pi = priority_scheduling(D, priorities=U, return_schedule=True)
# loads = {"P{}".format(k) : [(t[1], t[2] - t[1]) for t in v] for k, v in pi.items()}

# fig, ax = plt.subplots(dpi=400)
# for i, w in enumerate(["P0", "P1"]):
#     ax.broken_barh(loads[w], (5*i, 5), facecolors='#348ABD', alpha=0.5, edgecolor='w', joinstyle='bevel')    
        
# ax.set_xlim(0, mkspan)
# ax.set_xlabel('TIME')
# ax.set_ylabel('PROCESSOR')
# ax.set_ylim(0, 10)
# ax.set_yticks(list(2.5 + 5*i for i in range(2)))
# ax.set_yticklabels(["P1", "P2"]) 

# # # Add task IDs.
# for i, w in enumerate(pi):    
#     for t in pi[w]:
#         if t[0] in [6, 7]:
#             ax.annotate(t[0], (t[1] + 0.2 * (t[2] - t[1]), 2.5 + 5*i), color='w')
#         else:    
#             ax.annotate(t[0], (t[1] + 0.4 * (t[2] - t[1]), 2.5 + 5*i), color='w')
    
# ax.annotate('Makespan = {}'.format(int(mkspan)), (mkspan, 5),
#     xytext=(0.85, 0.9), textcoords='axes fraction',
#     bbox=dict(boxstyle="round", fc="w"),
#     arrowprops=dict(facecolor='black', shrink=0.05),
#     fontsize=14,
#     horizontalalignment='right', verticalalignment='top')

# # # Tend to play around with these...
# ax.xaxis.grid(False)
# ax.yaxis.grid(False)

# plt.savefig('opt_schedule.png', bbox_inches='tight') 
# plt.close(fig) 

# =============================================================================
# Disjunctive graph (for Chapter 4).
# =============================================================================

J = nx.DiGraph()
info = {1:[2, 3, 4], 2:[4, 5], 3:[5, 8], 4:[5], 5:[6, 7, 8], 6:[7, 9], 7:[9], 8:[9]}
for n, kids in info.items():
    for c in kids:
        J.add_edge(n, c)  
proper_edges = list(G.edges())
dis_edges = [(2, 4), (3, 8), (6, 7)]  
node_weights = {1:2, 2:7, 3:3, 4:6, 5:5, 6:1, 7:1, 8:3, 9:4}
edge_weights = {(1, 3):1, (3, 5):1, (5, 8):8, (6, 9):4, (7, 9):9}

# Draw graph the labelled graph.
plt.figure(figsize=(14,14))
pos = graphviz_layout(J, prog='dot')   
# Draw the topology. 
nx.draw_networkx_nodes(J, pos, node_color='#348ABD', node_size=2000, alpha=0.5)
nx.draw_networkx_edges(J, pos, edgelist=proper_edges, width=1.0)
nx.draw_networkx_edges(J, pos, edgelist=dis_edges, width=3.0, style='--')

# Draw the node labels.
nx.draw_networkx_labels(J, pos, font_size=18, font_weight='bold')

#Draw the node weights.
alt_pos = {}
for p in pos:
    if p in [1, 4, 6, 9]:
        alt_pos[p] = (pos[p][0] + 6, pos[p][1])
    elif p in [2, 3, 7]:
        alt_pos[p] = (pos[p][0], pos[p][1] + 21)
    elif p in [5]:
        alt_pos[p] = (pos[p][0] - 6, pos[p][1])
    else:
        alt_pos[p] = (pos[p][0], pos[p][1] - 25)            
nx.draw_networkx_labels(J, alt_pos, node_weights, font_size=18, font_weight='bold', font_color='#E24A33')

# # Draw the edge weights.    
nx.draw_networkx_edge_labels(J, pos, font_size=18, edge_labels=edge_weights, font_weight='bold', font_color='#E24A33')

# Calculate and draw the longest path.
for p, c in J.edges():
    J[p][c]['weight'] = node_weights[p]
    if c == 9:
        J[p][c]['weight'] += node_weights[c]
    try:
        J[p][c]['weight'] += edge_weights[(p, c)]
    except KeyError:
        pass
lp = nx.algorithms.dag.dag_longest_path(J)
crit_edges = []
for i, t in enumerate(lp[1:]):
    crit_edges.append((lp[i], t)) 
nx.draw_networkx_edges(G, pos, edgelist=crit_edges, edge_color='#E24A33', alpha=0.5, width=20.0)

plt.axis("off") 
plt.savefig('disjunctive.png', bbox_inches='tight') 


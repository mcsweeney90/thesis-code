#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Not used at the moment but may change mind.
"""

import dill, pathlib, os
from statistics import variance
import sys
sys.path.append('../') 
from src import DAG, heft

####################################################################################################

size = 1000
dag_path = '../../graphs/STG/{}'.format(size)
ccr = 1.0
r = 32
s = 1
delta = lambda source, dest: 0.0 if (source == dest) or (source < r and dest < r) else 1.0

####################################################################################################

for dname in os.listdir(dag_path):     
    print(dname)
    
    # Load the DAG topology.
    with open('{}/{}'.format(dag_path, dname), 'rb') as file:
        T = dill.load(file)
    # Convert to DAG object.
    G = DAG(T)
    
    G.set_random_weights(r=r, s=s, ccr=ccr)  
    
    # Compute HEFT schedule.
    _, pi = heft(G, r, s, return_schedule=True)
    
    # Convert pi to a schedule graph.
    where_scheduled = {}
    for w, load in pi.items():
        for t in list(c[0] for c in load):
            where_scheduled[t] = w 
    # Construct graph topology.
    S = G.graph.__class__()
    S.add_nodes_from(G.graph)
    S.add_edges_from(G.graph.edges)
    # Set the weights.
    for t in G.top_sort:
        proc_type = "c" if where_scheduled[t] < r else "g"
        S.nodes[t]['weight'] = G.graph.nodes[t]['weight'][proc_type]             
        
        for c in G.graph.successors(t):
            if (where_scheduled[c] == where_scheduled[t]) or (where_scheduled[t] < r and where_scheduled[c] < r):
                S[t][c]['weight'] = 0.0
            else:       
                S[t][c]['weight'] = delta(where_scheduled[t], where_scheduled[c])*G.graph[t][c]['weight']
            
        # Add disjunctive edge if necessary.
        idx = list(r[0] for r in pi[where_scheduled[t]]).index(t)
        if idx > 0:
            d = pi[where_scheduled[t]][idx - 1][0]
            if not S.has_edge(d, t):
                S.add_edge(d, t)
                S[d][t]['weight'] = 0.0    
    
    # Save the graph.
    dag_save_path = '../../chapter4/stg/'
    pathlib.Path(dag_save_path).mkdir(parents=True, exist_ok=True)
    with open('{}/{}.dill'.format(dag_save_path, dname[:-5]), 'wb') as handle:
        dill.dump(S, handle)



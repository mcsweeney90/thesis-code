#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create topologies for Cholesky DAGs.
"""

import dill, pathlib
import networkx as nx
from timeit import default_timer as timer

def cholesky(n_tiles):
    """
    Create a networkx digraph representing Cholesky factorization. 
    From the earliest days of my PhD so it's a bit code soup, will optimize some day!
    """
    
    last = {} # Useful for keeping track of things...   
    counts = {"P" : 1, "T" : 1, "S" : 1, "G" : 1}
    G = nx.DiGraph()    
    for k in range(n_tiles): # Grow the DAG column by column.
        n1 = "P{}".format(counts["P"])
        counts["P"] += 1
        if k > 0:
            G.add_edge(last[(k, k)], n1)  
        last[(k, k)] = n1      
        for i in range(k + 1, n_tiles):
            n2 = "T{}".format(counts["T"])
            counts["T"] += 1
            G.add_edge(n1, n2)
            try:
                G.add_edge(last[(i, k)], n2)
            except KeyError:
                pass
            last[(i, k)] = n2     
        for i in range(k + 1, n_tiles): 
            n3 = "S{}".format(counts["S"])
            counts["S"] += 1
            try:
                G.add_edge(last[(i, i)], n3)
            except KeyError:
                pass
            last[(i, i)] = n3            
            try:
                G.add_edge(last[(i, k)], n3)
            except KeyError:
                pass                
            for j in range(k + 1, i):       
                n4 = "G{}".format(counts["G"])
                counts["G"] += 1
                try:
                    G.add_edge(last[(i, j)], n4)
                except KeyError:
                    pass
                last[(i, j)] = n4                
                try:
                    G.add_edge(last[(i, k)], n4)
                except KeyError:
                    pass                
                try:
                    G.add_edge(last[(j, k)], n4)
                except KeyError:
                    pass                  
    return G

# =============================================================================
# Create and save topologies.
# =============================================================================
          
chol_dag_dest = 'topologies'
pathlib.Path(chol_dag_dest).mkdir(parents=True, exist_ok=True)
for nt in range(5, 51, 5):
    start = timer()
    C = cholesky(nt)
    with open('{}/{}.dill'.format(chol_dag_dest, nt), 'wb') as handle:
        dill.dump(C, handle)
    elapsed = timer() - start
    print("\n{}x{} ({} tasks) took {} seconds.".format(nt, nt, len(C), elapsed))
    



            


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare CLT-based heuristics for Cholesky graphs.
"""

import dill
import pandas as pd
from itertools import product
from timeit import default_timer as timer

# TODO: the following apparently unnecessary import statement may be needed to load the DAGs - suspect due to dill saving by reference.
import sys
sys.path.append("../")
from src import RV, StochDAG

data = []
ntasks = list(range(5, 51, 5))
for N, nb, s in product(ntasks, [128, 1024], [1, 4]): 
    chol_load_path = '../chol_graphs/nb{}s{}'.format(nb, s)
    with open('{}/{}.dill'.format(chol_load_path, N), 'rb') as file:
        R = dill.load(file)
    G = StochDAG(R)
    
    graph_data = {"n" : G.size, "N" : N, "nb" : nb, "s" : s}
    
    start = timer()
    cpm = G.CPM()
    elapsed = timer() - start
    graph_data["CPM"] = cpm
    graph_data["CPM TIME"] = elapsed
        
    # Sculli.
    start = timer()
    SL = G.sculli()
    elapsed = timer() - start
    mu, var = SL.mu, SL.var
    graph_data["SCULLI MU"] = mu
    graph_data["SCULLI VAR"] = var
    graph_data["SCULLI TIME"] = elapsed   
    
    # Sculli (reverse)
    start = timer()
    SL = G.sculli(reverse=True)
    elapsed = timer() - start
    mu, var = SL.mu, SL.var
    graph_data["SCULLI-R MU"] = mu
    graph_data["SCULLI-R VAR"] = var
    graph_data["SCULLI-R TIME"] = elapsed      
    
    # CorLCA.
    start = timer()
    CL = G.corLCA()
    elapsed = timer() - start
    mu, var = CL.mu, CL.var
    graph_data["CORLCA MU"] = mu
    graph_data["CORLCA VAR"] = var
    graph_data["CORLCA TIME"] = elapsed
    
    # CorLCA (reverse)
    start = timer()
    CL = G.corLCA(reverse=True)
    elapsed = timer() - start
    mu, var = CL.mu, CL.var
    graph_data["CORLCA-R MU"] = mu
    graph_data["CORLCA-R VAR"] = var
    graph_data["CORLCA-R TIME"] = elapsed
    
    # Kamburowski.
    start = timer()
    lm, um, ls, us = G.kamburowski()
    elapsed = timer() - start
    graph_data["K LOWER MU"] = lm
    graph_data["K UPPER MU"] = um
    graph_data["K LOWER VAR"] = ls
    graph_data["K UPPER VAR"] = us
    graph_data["K TIME"] = elapsed    
    
    # Save the data.
    data.append(graph_data)

# Save the dataframe.
df = pd.DataFrame(data)  
df.to_csv('clt.csv', encoding='utf-8', index=False)
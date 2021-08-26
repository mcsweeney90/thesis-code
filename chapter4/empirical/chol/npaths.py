#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Number of paths observed in MC. 
(Should have done originally but decided to add later...)
TODO: check still works since changes to how Cholesky graphs are saved.
"""

import dill
import pandas as pd
from itertools import product

# import sys
# sys.path.append("../../")
# from src import StochDAG

data = []
ntasks = list(range(5, 51, 5))
for nt, nb, s in product(ntasks, [128, 1024], [1, 4]): 
    chol_load_path = '../../chol_graphs/nb{}s{}'.format(nb, s)
    with open('{}/{}.dill'.format(chol_load_path, nt), 'rb') as file:
        G = dill.load(file)
    
    graph_data = {"n" : G.size, "nt" : nt, "nb" : nb, "s" : s}
    
    runs = 100000  
    for distro in ["normal", "gamma", "uniform"]:   
        
        # Generate the MC data.
        paths, _ = G.monte_carlo_paths(samples=runs, dist=distro)
        graph_data["{}".format(distro)] = len(paths) 
    
    # Save the data.
    data.append(graph_data)
    
# Save the dataframe.
df = pd.DataFrame(data)  
df.to_csv('npaths.csv', encoding='utf-8', index=False)
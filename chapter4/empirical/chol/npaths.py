#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Number of distinct paths observed to be critical when generating empirical longest path distribution. 
NOTE: to avoid overwriting the data 'npaths.csv' that was used in thesis, have changed the name of save destination to 'new_npaths.csv'. 
"""

import dill
import pandas as pd
from itertools import product

data = []
ntiles = list(range(5, 51, 5))
for N, nb, s in product(ntiles, [128, 1024], [1, 4]): 
    chol_load_path = '../../chol_graphs/nb{}s{}'.format(nb, s)
    with open('{}/{}.dill'.format(chol_load_path, N), 'rb') as file:
        G = dill.load(file)
    
    graph_data = {"n" : G.size, "N" : N, "nb" : nb, "s" : s}
    
    runs = 100000  
    for distro in ["normal", "gamma", "uniform"]:   
        
        # Generate the MC data.
        paths, _ = G.monte_carlo_paths(samples=runs, dist=distro)
        graph_data["{}".format(distro)] = len(paths) 
    
    # Save the data.
    data.append(graph_data)
    
# Save the dataframe.
df = pd.DataFrame(data)  
df.to_csv('new_npaths.csv', encoding='utf-8', index=False)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate and save empirical distributions.
NOTE: commented out by default to prevent overwriting data used in thesis. If want to rerun, suggest changing save location on line 26.
"""

import dill, pathlib
from itertools import product

# ntiles = list(range(5, 51, 5))
# for N, nb, s in product(ntiles, [128, 1024], [1, 4]): 
#     chol_load_path = '../../chol_graphs/nb{}s{}'.format(nb, s)
#     with open('{}/{}.dill'.format(chol_load_path, N), 'rb') as file:
#         G = dill.load(file)
    
#     runs = 100000     
#     for distro in ["normal", "gamma", "uniform"]:   
#         save_path = "nb{}s{}/data/{}/".format(nb, s, distro)     
#         pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        
#         # Generate the MC data.
#         D = G.monte_carlo(samples=runs, dist=distro)
        
#         # Save the data. 
#         with open('{}/{}.dill'.format(save_path, N), 'wb') as handle:
#             dill.dump(D, handle)
        
        
            


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate empirical distributions.
"""

import dill, pathlib
from itertools import product

import sys
sys.path.append("../../")
from src import StochDAG

ntasks = list(range(5, 51, 5))
for nt, nb, s in product(ntasks, [128, 1024], [1, 4]): 
    chol_load_path = '../../chol_graphs/nb{}s{}'.format(nb, s)
    with open('{}/{}.dill'.format(chol_load_path, nt), 'rb') as file:
        R = dill.load(file)
    G = StochDAG(R)
    
    runs = 100000     
    for distro in ["normal", "gamma", "uniform"]:   
        save_path = "nb{}s{}/data/{}/".format(nb, s, distro)     
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Generate the MC data.
        D = G.monte_carlo(samples=runs, dist=distro)
        
        # Save the data.
        with open('{}/{}.dill'.format(save_path, nt), 'wb') as handle:
            dill.dump(D, handle)
        
        
            


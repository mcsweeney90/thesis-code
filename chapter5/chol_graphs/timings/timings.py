#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save timing data as a dict for ease of access.
"""

import dill
import numpy as np

tile_sizes = [32, 64, 128, 256, 512, 1024]
kernels = {"G" : "gemm", "P" : "potrf", "S" : "syrk", "T" : "trsm"}

cpu_type = "skylake"
cpu_raw = {k : np.genfromtxt('{}/d{}.csv'.format(cpu_type, kernels[k]), delimiter=',', skip_header=1) for k in kernels}
gpu_type = "V100"
gpu_raw = {k : np.genfromtxt('{}/d{}.csv'.format(gpu_type, kernels[k]), delimiter=',', skip_header=1) for k in kernels}

# Save kernel timings.
timings = {nb : {kernel : {} for kernel in kernels} for nb in tile_sizes}
runs, idx = 1000, 1
for nb in tile_sizes:
    for kernel in kernels:
        timings[nb][kernel]["c"] = list(y[2] for y in cpu_raw[kernel][idx : idx + runs])  
        timings[nb][kernel]["g"] = list(y[2] for y in gpu_raw[kernel][idx : idx + runs]) 
        timings[nb][kernel]["d"] = list(y[3] - y[2] for y in gpu_raw[kernel][idx : idx + runs]) # TODO.
    idx += (runs + 1)
with open('timings.dill', 'wb') as handle:
    dill.dump(timings, handle)

# Graphs

Graph topologies used throughout. 

## Navigation

`cholesky` : Cholesky factorization DAGs.

`STG` : Graphs based on the Standard Task Graph (STG) set. Graphs are converted from original format into `networkx` DiGraphs, then saved using `dill`.
 
## Comments

Cholesky factorization topologies are used to create Cholesky task/schedule graphs (i.e., with weights) in `chapter2`, `chapter3` and `chapter4` which are then saved locally in that directory for future use. Although they are sometimes fairly large, this proved much faster. For the STG graphs, there are typically so many of them needed for experiments that saving them just isn't possible so scripts use the topologies, create the task/schedule graphs, and then apply whichever algorithms are being evaluated. 

## References

[1] T. Tobita and H. Kasahara. A standard task graph set for fair evaluation of multiprocessor scheduling algorithms. Journal of Scheduling, vol.5, no. 5, pp. 379-394, 2002.

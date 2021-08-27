# Thesis

This is the central repository for the (Python) code used to generate results in my PhD thesis, "Scheduling with Precedence Constraints in Heterogeneous Parallel Computing".

## Navigation

`chapter2` : Optimizing scheduling heuristics for accelerated architectures.

`chapter3` : The critical path in heterogeneous scheduling.
 
`chapter4` : Predicting schedule length under uncertainty.

`chapter5` : Stochastic scheduling.

`graphs` : Graph topologies - `networkx` DiGraphs saved using `dill` - which are used by code in all of the above chapter folders. 

`requirements.txt` : Python packages required to run all code.

## Getting started

This has only been tested for Python **>= 3.8** so performance for older versions cannot be guaranteed. 

To install all relevant Python packages:
```
pip install -r requirements.txt
```

## Troubleshooting

The most common problems I encountered were:

1. Issues with `dill`. 


## License

This project is licensed under the GPL-3.0 License - see [LICENSE](LICENSE) for details.

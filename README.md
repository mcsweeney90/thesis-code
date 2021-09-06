# Thesis code

This is the central repository for the (Python) code used to generate results in my PhD thesis, "Scheduling with Precedence Constraints in Heterogeneous Parallel Computing".

## Navigation

`chapter2` : Optimizing scheduling heuristics for accelerated architectures.

`chapter3` : The critical path in heterogeneous scheduling.
 
`chapter4` : Predicting schedule length under uncertainty.

`chapter5` : Stochastic scheduling.

`graphs` : Graph topologies which are used by code in all of the above chapter folders. 

`requirements.txt` : Python packages required to run all code.

## Getting started

I strongly recommend using a version of Python **>= 3.8** since some of the code in `chapter4` and `chapter5` uses functions from the `statistics` module that were only introduced in **3.8**.
Older Python3 versions might be adequate for `chapter2` and `chapter3` but Python2 definitely will not work. 

To install all relevant Python packages:
```
pip install -r requirements.txt
```

## License

This project is licensed under the GPL-3.0 License - see [LICENSE](LICENSE) for details.

# Thesis

This is the central repository for code used in my PhD thesis, "Scheduling with Precedence Constraints in Heterogeneous Parallel Computing".

## chapter2, chapter3, chapter4, chapter5

These folders contain the source code used to generate results presented and discussed in the corresponding thesis chapters. All code is written in Python. Details on specific packages required and other installation information are provided within.

## Graphs

Contains graph topologies, saved and loaded using the Python `dill` package, which are used by code in all of the above chapter folders.

## Requirements

This has only been tested for Python **>= 3.6** so performance for older versions cannot be guaranteed. In particular, I imagine there may be issues since it is often implicitly assumed that `dict` data types (and therefore also `defaultdicts` from the `collections` package) are ordered by insertion time. 

To install all relevant Python packages:
```
pip install -r requirements.txt
```

## License

This project is licensed under the GPL-3.0 License - see [LICENSE](LICENSE) for details.

# Distribution `dtype` for Pandas

A small proof-of-concept about using a custom Pandas dtype to store probability distributions.

...


## Getting started

Install the package from source:

```
pip install git+https://github.com/jojolebarjos/distribution-pandas-dtype.git
```

Additionally, you may want to install `pyarrow`, to support serialization as Parquet files:

```
pip install pyarrow
```

At import, the extension dtypes are registered into the pandas ecosystem.
Behind the scenes, the data is stored as a structured NumPy array, which is designed to store C-style structures.
The multi-values nature of these objects require some care, and does not play nicely with some indexing operations; as such, Pandas does not accept structured NumPy dtypes for their built-in series.

To circumvent this limitation, there are several approaches to initialize distribution series.
The simplest one is to create a zero-initialized series, and update the fields separately:

```py
import distribution

x = pd.Series(index=range(5), dtype="dist[lognorm]")
x.dist["mu"] = [1, 2, 3, 4, 5]
x.dist["sigma"] = 0.1
```


## Relevant links

 * https://numpy.org/doc/stable/user/basics.rec.html
 * https://pandas.pydata.org/docs/development/extending.html
 * https://github.com/pandas-dev/pandas/blob/main/pandas/core/dtypes/dtypes.py
 * https://github.com/pandas-dev/pandas/blob/main/pandas/core/arrays/numpy_.py
 * https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/arrays.pyx
 * https://github.com/ContinuumIO/cyberpandas/blob/master/cyberpandas/base.py
 * https://github.com/geopandas/geopandas/blob/main/geopandas/array.py

import os
import tempfile

import numpy as np
import pandas as pd

from distribution import (
    DistributionArray,
    LogNormal,
)


def test_from_name():
    x = pd.Series(index=range(5), dtype="dist[lognorm]")
    x.dist["mu"] = [1, 2, 3, 4, 5]
    x.dist["sigma"] = 0.1
    assert x[2] == LogNormal(3.0, 0.1)


def test_from_array():
    dtype = LogNormal.dtype
    array = DistributionArray.empty(10, dtype)
    array.view_field("mu")[:] = range(10)
    array.view_field("sigma")[:] = 2
    x = pd.Series(array)
    assert x[8] == LogNormal(8.0, 2.0)


def test_from_ndarray():
    numpy_dtype = np.dtype([("mu", "f8"), ("sigma", "f8")])
    numpy_array = np.empty(3, dtype=numpy_dtype)
    numpy_array["mu"] = [0.5, 1.5, 2.5]
    numpy_array["sigma"] = 0.2
    dtype = LogNormal.dtype
    array = DistributionArray(numpy_array, dtype)
    x = pd.Series(array, dtype=dtype)
    assert x[2] == LogNormal(2.5, 0.2)


def test_pyarrow():
    x = pd.Series(index=range(5), dtype="dist[lognorm]")
    x.dist["mu"] = [1, 2, 3, 4, 5]
    x.dist["sigma"] = 0.1
    df = x.to_frame("x")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "foo.parquet")
        df.to_parquet(path)
        reloaded_df = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, reloaded_df)

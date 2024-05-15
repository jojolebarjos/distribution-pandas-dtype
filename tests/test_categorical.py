import os
import tempfile

import numpy as np
import pandas as pd

import pytest

from distribution import (
    Categorical,
    CategoricalDtype,
    DistributionArray,
)


def test_from_name():
    x = pd.Series(index=range(7), dtype="dist[categorical, a, b]")
    x.dist["a"] = [0.9] * 7
    x.dist["b"] = 0.1
    assert x[2].to_dict() == {"a": 0.9, "b": 0.1}


def test_from_dtype():
    dtype = CategoricalDtype(["alpha", "beta", "gamma"])
    x = pd.Series(index=range(5), dtype=dtype)
    x.dist["alpha"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    x.dist["beta"] = 0.5
    x.dist["gamma"] = 1.0 - x.dist["alpha"] - x.dist["beta"]
    assert x[4].to_tuple() == (0.5, 0.5, 0.0)


def test_from_ndarray():
    numpy_dtype = np.dtype([("alpha", "f8"), ("beta", "f8"), ("gamma", "f8")])
    numpy_array = np.empty(3, dtype=numpy_dtype)
    numpy_array["alpha"] = [0.1, 0.2, 0.3]
    numpy_array["beta"] = [0.4, 0.2, 0.0]
    numpy_array["gamma"] = [0.5, 0.6, 0.7]
    dtype = CategoricalDtype(["alpha", "beta", "gamma"])
    array = DistributionArray(numpy_array, dtype)
    x = pd.Series(array, dtype=dtype)
    assert x[2] == Categorical((0.3, 0.0, 0.7), dtype=dtype)


def test_pyarrow():
    dtype = CategoricalDtype(["alpha", "beta", "gamma"])
    x = pd.Series(index=range(3), dtype=dtype)
    x.dist["alpha"] = [0.1, 0.2, 0.3]
    x.dist["beta"] = [0.4, 0.2, 0.0]
    x.dist["gamma"] = [0.5, 0.6, 0.7]
    df = x.to_frame("x")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "foo.parquet")
        df.to_parquet(path)
        reloaded_df = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, reloaded_df)


def test_names():
    x = pd.Series(dtype="dist[categorical, hello world,   First Class, A-Z, o'clock]")
    assert x.dtype.names == ["hello world", "First Class", "A-Z", "o'clock"]
    with pytest.raises(ValueError, match=r"\["):
        x = pd.Series(dtype="dist[categorical, []")

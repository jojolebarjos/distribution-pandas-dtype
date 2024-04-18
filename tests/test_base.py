import numpy as np

import pandas as pd
from pandas.api.extensions import register_extension_dtype

from structured import StructuredDtype


@register_extension_dtype
class FooDtype(StructuredDtype):
    name = "foo"
    dtype = np.dtype([("a", "i4"), ("b", "f8")])
    type = tuple
    _na_values = (0, np.nan)


def test_simple():
    s = pd.Series(index=range(10), dtype="foo")
    assert s.isna().all()

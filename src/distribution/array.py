from __future__ import annotations

from functools import reduce
import numbers
import operator

import numpy as np

import pandas as pd
from pandas.api.extensions import ExtensionArray
from pandas.core.algorithms import take


class DistributionArray(ExtensionArray):
    """..."""

    def __init__(self, array, dtype):
        self._array = array
        self._dtype = dtype

    @classmethod
    def empty(cls, length, dtype):
        array = np.zeros(length, dtype=dtype._internal_dtype)
        return cls(array, dtype)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if dtype is None:
            raise ValueError
        array = np.zeros(len(scalars), dtype=dtype._internal_dtype)
        for i, scalar in enumerate(scalars):
            dtype._set_scalar(array, i, scalar)
        return cls(array, dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, original._dtype)

    def _values_for_factorize(self):
        return self._array, np.nan

    def _values_for_argsort(self):
        return self._array

    # TODO _values_for_json?

    # TODO _formatter?

    def __array__(self, dtype=None):
        # TODO would be nice to return the array itself, but structured arrays do not play nice with Pandas internals...
        return np.array(
            [self._dtype._get_scalar(self._array, i) for i in range(len(self))]
        )

    # TODO __array_ufunc__?

    def __getitem__(self, indexer):
        if isinstance(indexer, numbers.Integral):
            return self._dtype._get_scalar(self._array, int(indexer))
        indexer = pd.api.indexers.check_array_indexer(self, indexer)
        cls = type(self)
        return cls(self._array[indexer], self._dtype)

    def __setitem__(self, indexer, value):
        if isinstance(indexer, numbers.Integral):
            self._dtype._set_scalar(self._array, int(indexer), value)
        else:
            indexer = pd.api.indexers.check_array_indexer(self, indexer)
            # TODO more flexibility in accepted types?
            self._array[indexer] = value

    def __len__(self):
        return len(self._array)

    def __eq__(self, other):
        if not isinstance(other, DistributionArray):
            return NotImplemented
        if self._dtype != other._dtype:
            return NotImplemented
        return self._array == other._array

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._array.nbytes

    def isna(self):
        arrays = [np.isnan(self._array[name]) for name in self._dtype.names]
        return reduce(operator.or_, arrays)

    def copy(self):
        cls = type(self)
        return cls(self._array.copy(), self._dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(
            np.concatenate([array._array for array in to_concat]),
            dtype=to_concat[0]._dtype,
        )

    def take(
        self,
        indices,
        *,
        allow_fill=False,
        fill_value=None,
    ):
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        array = take(self._array, indices, fill_value=fill_value, allow_fill=allow_fill)
        cls = type(self)
        return cls(array, dtype=self.dtype)

    # TODO to_numpy? what does it depends on?
    # TODO check other methods that should be overridden

    def __arrow_array__(self, type=None):
        import pyarrow as pa

        names = self._dtype.names
        arrays = [self._array[name] for name in names]
        return pa.StructArray.from_arrays(arrays, names)

    def view_field(self, name: str):
        try:
            return self._array[name]
        except ValueError as e:
            raise KeyError(name) from e

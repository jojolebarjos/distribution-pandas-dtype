from __future__ import annotations

from functools import reduce
import numbers
import operator
from typing import Any, Union

import numpy as np

import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype,
    register_series_accessor,
)
from pandas.core.algorithms import take

import pyarrow as pa


# TODO structured dtype should be usable on its own (and registered), with proper format (in square brackets)


class StructuredDtype(ExtensionDtype):
    """..."""

    name: str
    dtype: np.dtype
    type: type

    na_value = None
    _na_values: tuple
    # TODO check the whole NA aspect (e.g. _can_hold_na, _hasna)

    @property
    def names(self) -> list[str]:
        return list(self.dtype.names)

    @classmethod
    def construct_array_type(cls) -> type[StructuredArray]:
        return StructuredArray

    @classmethod
    def construct_from_string(cls, string: str) -> type[StructuredArray]:
        if string == cls.name:
            return cls()
        raise TypeError

    def __from_arrow__(
        self, array: Union[pa.Array, pa.ChunkedArray]
    ) -> StructuredArray:
        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks()
        data = np.empty(len(array), dtype=self.dtype)
        for name in self.dtype.names:
            data[name] = array.field(name)
        cls = self.construct_array_type()
        return cls(data, self)

    def _get_scalar(self, array: np.ndarray, index: int) -> Any:
        return self.type(array[index])

    def _set_scalar(self, array: np.ndarray, index: int, value: Any) -> None:
        if value is None:
            array[index] = self._na_values
        else:
            array[index] = value


class StructuredArray(ExtensionArray):
    """..."""

    def __init__(self, array: np.ndarray, dtype: StructuredDtype):
        assert array.dtype == dtype.dtype
        self._array = array
        self._dtype = dtype

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # TODO improve this
        if dtype is None:
            raise ValueError
        if not isinstance(dtype, StructuredDtype):
            raise TypeError
        array = np.zeros(len(scalars), dtype=dtype.dtype)
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

    def __getitem__(self, indexer):
        if isinstance(indexer, numbers.Integral):
            return self._dtype._get_scalar(self._array, indexer)
        indexer = pd.api.indexers.check_array_indexer(self, indexer)
        cls = type(self)
        return cls(self._array[indexer], self._dtype)

    def __setitem__(self, indexer, value):
        if isinstance(indexer, numbers.Integral):
            self._dtype._set_scalar(self._array, indexer, value)
        else:
            indexer = pd.api.indexers.check_array_indexer(self, indexer)
            # TODO more flexibility in accepted types?
            self._array[indexer] = value

    def __len__(self):
        return len(self._array)

    def __eq__(self, other):
        if not isinstance(other, StructuredArray):
            return NotImplemented
        if self._dtype != other._dtype:
            return NotImplemented
        return self._array == other._array

    @property
    def dtype(self) -> StructuredDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._array.nbytes

    def isna(self):
        arrays = [np.isnan(self._array[name]) for name in self._dtype.dtype.names]
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
        names = self._dtype.dtype.names
        arrays = [self._array[name] for name in names]
        return pa.StructArray.from_arrays(arrays, names)

    def __getattr__(self, name) -> np.ndarray:
        if name in self._dtype.dtype.names:
            return self._array[name]
        raise AttributeError(name)

    def __dir__(self):
        return super().__dir__() + list(self._dtype.dtype.names)

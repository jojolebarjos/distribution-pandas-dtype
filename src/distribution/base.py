from __future__ import annotations

from typing import Any

import numpy as np

from pandas.api.extensions import ExtensionDtype

from .array import DistributionArray


class Distribution:
    dtype: DistributionDtype

    def to_tuple(self) -> tuple:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return dict(zip(self.dtype.names, self.to_tuple()))

    def to_scipy(self):
        raise NotImplementedError


class DistributionDtype(ExtensionDtype):
    name: str
    type: type[Distribution]

    @property
    def names(self) -> list[str]:
        return list(self._internal_dtype.names)

    _internal_dtype: np.dtype

    def _get_scalar(self, array, index):
        kwargs = dict(zip(self.names, array[index]))
        return self.type(**kwargs)

    def _set_scalar(self, array, index, value):
        if value is None:
            array[index] = np.nan
        else:
            array[index] = value.to_tuple()

    @property
    def na_value(self) -> Any:
        return None

    @classmethod
    def construct_array_type(cls) -> type[DistributionArray]:
        return DistributionArray

    def __from_arrow__(self, array):
        import pyarrow as pa

        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks()
        data = np.empty(len(array), dtype=self._internal_dtype)
        for name in self.names:
            data[name] = array.field(name)
        cls = self.construct_array_type()
        return cls(data, self)

    def _to_scipy(self, array):
        raise NotImplementedError

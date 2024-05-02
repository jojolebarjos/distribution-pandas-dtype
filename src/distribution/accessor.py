import pandas as pd
from pandas.api.extensions import register_series_accessor

from .array import DistributionArray
from .base import DistributionDtype


@register_series_accessor("dist")
class DistributionSeriesAccessor:
    """..."""

    def __init__(self, obj):
        if not isinstance(obj, pd.Series):
            raise AttributeError
        if not isinstance(obj.dtype, DistributionDtype):
            raise AttributeError
        self._obj: pd.Series = obj
        self._array: DistributionArray = obj.values

    def __getitem__(self, key: str) -> pd.Series:
        array = self._array.view_field(key)
        return pd.Series(array, index=self._obj.index, name=key)

    def __setitem__(self, key: str, value) -> None:
        array = self._array.view_field(key)
        if isinstance(value, pd.Series):
            indices = self._obj.index.get_indexer(value.index)
            mask = indices >= 0
            array[indices[mask]] = value.values[mask]
        else:
            array[:] = value

    def unpack(self) -> pd.DataFrame:
        arrays = {
            name: self._array.view_field(name) for name in self._array._dtype.names
        }
        return pd.DataFrame(arrays, index=self._obj.index)

    def to_scipy(self):
        return self._array._dtype._to_scipy(self._array._array)

    # def sample(self) -> pd.Series:
    #     dist = self.to_scipy()
    #     values = dist.rvs()
    #     return pd.Series(values, index=self._obj.index, name=self._obj.name)

    # TODO access to other methods from rv object (e.g. pdf, cdf)

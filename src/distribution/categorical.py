from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Sequence

import numpy as np

from pandas.api.extensions import register_extension_dtype

import scipy.stats

from .base import Distribution, DistributionDtype


STRING_PATTERN = re.compile(r"^dist\[categorical((?:\s*,\s*\w+)*)\s*\]$")


@dataclass
class Categorical(Distribution):
    theta: tuple[float, ...]
    dtype: CategoricalDtype = field(repr=False)

    def to_tuple(self) -> tuple[float, ...]:
        return self.theta

    def to_scipy(self):
        return scipy.stats.rv_discrete(
            name="categorical",
            values=(range(len(self.theta)), self.theta),
        )


@register_extension_dtype
class CategoricalDtype(DistributionDtype):
    type = Categorical

    _metadata = ("names",)

    def __init__(self, names: Sequence[str]) -> None:
        for name in names:
            if not name.isidentifier():
                raise ValueError(name)
        self._internal_dtype = np.dtype([(name, "f8") for name in names])

    @property
    def name(self):
        return f"dist[{', '.join(['categorical', *self.names])}]"

    @classmethod
    def construct_from_string(cls, string: str) -> CategoricalDtype:
        match = STRING_PATTERN.match(string)
        if match:
            names = [name.strip() for name in match.group(1).split(",")[1:]]
            return cls(names)
        raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    def _get_scalar(self, array, index):
        theta = array[index].tolist()
        return self.type(theta, self)

    def _set_scalar(self, array, index, value):
        if value is None:
            array[index] = np.NaN
        else:
            if value.dtype != self:
                raise TypeError
            array[index] = value.theta

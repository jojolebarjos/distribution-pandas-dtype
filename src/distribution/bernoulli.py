from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pandas.api.extensions import register_extension_dtype

import scipy.stats

from .base import Distribution, DistributionDtype


@dataclass
class Bernoulli(Distribution):
    p: float

    def to_tuple(self) -> tuple[float]:
        return (self.p,)

    def to_scipy(self):
        return scipy.stats.bernoulli(self.p)


@register_extension_dtype
class BernoulliDtype(DistributionDtype):
    name = "dist[bernoulli]"
    type = Bernoulli

    _internal_dtype = np.dtype([("p", "f8")])

    def _to_scipy(self, array):
        return scipy.stats.bernoulli(array["p"])


Bernoulli.dtype = BernoulliDtype()

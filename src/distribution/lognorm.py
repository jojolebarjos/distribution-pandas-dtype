from dataclasses import dataclass

import numpy as np

from pandas.api.extensions import register_extension_dtype

import scipy.stats

from .base import Distribution, DistributionDtype


@dataclass
class LogNormal(Distribution):
    mu: float
    sigma: float

    def to_tuple(self) -> tuple[float, float]:
        return self.mu, self.sigma

    def to_scipy(self):
        return scipy.stats.lognorm(self.sigma, loc=0.0, scale=np.exp(self.mu))


@register_extension_dtype
class LogNormalDtype(DistributionDtype):
    name = "dist[lognorm]"
    type = LogNormal

    _internal_dtype = np.dtype([("mu", "f8"), ("sigma", "f8")])

    def _to_scipy(self, array):
        return scipy.stats.lognorm(array["sigma"], loc=0.0, scale=np.exp(array["mu"]))


LogNormal.dtype = LogNormalDtype()

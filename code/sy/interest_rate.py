from __future__ import annotations
import typing
from datetime import date, timedelta, datetime
import numpy as np
import pandas as pd

# interest rate model
class CIR(object):
    def __init__(self, data: pd.DataFrame, params: typing.Dict):
        # TODO
        self.data = data
        """
        b: long term mean level: All future trajectories of r will evolve around a mean level b in the long run.
        a: speed of reversion: A characterizes the velocity at which such trajectories will regroup around b.
        sigma: instantaneous volatility: measures instant by instant the amplitude of randomness
        """
        self.a = None
        self.b = None
        self.sigma = None
        return None

    def fit(self, current_date: str):
        # TODO
        return self
    
    def generate_path(self):
        # TODO
        return None
import math
from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame



class VanillaOption:
    def __init__(self, params=None):
        if params is None:
            params = {'kind': None,
                      'premium': None,
                      'spot': None,
                      'r': None,
                      'delta': None,
                      'gamma': None,
                      'vega': None,
                      'theta': None,
                      'rho': None}

        self.__dict__.update(params)

        self.kind = params['kind']
        self.premium = params['premium']
        self.spot = params['spot']
        self.r = params['ret']
        self.delta = params['delta']
        self.gamma = params['gamma']
        self.vega = params['vega']
        self.theta = params['theta']
        self.rho = params['rho']


        def delta_call(self):
            pass
            # self.delta = 0.8

        def delta_c(spot, strike, time, r, sigma):
            pass
            ##return norm.cdf(d1(S, K, T, r, sigma))


def d1(option):
    r, spot, strike, time, sigma = option.r, option.spot, option.strike, option.time, option.sigma
    return (np.log(spot / strike) + (r + sigma**2 / 2) * time)/(sigma * np.sqrt(time))


def d2(option):
    time, sigma = option.time, option.sigma
    return d1(option) - sigma * np.sqrt(time)










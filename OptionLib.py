from scipy.stats import norm
import numpy as np


class VanillaOption:
    def __init__(self, params=None):
        if params is None:
            params = {'type': None,
                      'premium': None,
                      'spot': None,
                      'r': None,
                      'delta': None,
                      'gamma': None,
                      'vega': None,
                      'theta': None,
                      'rho': None,
                      'strike': None,
                      'time': None,
                      'sigma': None}

        self.__dict__.update(params)

        self.type = params['type']
        self.premium = params['premium']
        self.spot = params['spot']
        self.r = params['r']
        self.delta = params['delta']
        self.gamma = params['gamma']
        self.vega = params['vega']
        self.theta = params['theta']
        self.rho = params['rho']
        self.strike = params['strike']
        self.time = params['time']
        self.sigma = params['sigma']

    def greeks(self):
        """
            Returns dictionary of Delta, Gamma, Theta, Rho, Vega of an Option,
            also assign computed values to the class object
        """
        return {
            'delta': self.delta_compute(),
            'gamma': self.gamma_compute(),
            'theta': self.theta_compute(),
            'rho': self.rho_compute(),
            'vega': self.vega_compute()
        }

    def delta_compute(self):
        self.delta = delta(self.r, self.spot, self.strike, self.time, self.sigma, self.type)
        return self.delta

    def gamma_compute(self):
        self.gamma = gamma(self.r, self.spot, self.strike, self.time, self.sigma, self.type)
        return self.gamma

    def theta_compute(self):
        self.theta = theta(self.r, self.spot, self.strike, self.time, self.sigma, self.type)
        return self.theta

    def rho_compute(self):
        self.rho = rho(self.r, self.spot, self.strike, self.time, self.sigma, self.type)
        return self.rho

    def vega_compute(self):
        self.vega = vega(self.r, self.spot, self.strike, self.time, self.sigma, self.type)
        return self.vega


Standard_Error_Output = "BSM price is undefined, \
        please male sure you use correct types of options ('call'/'put'),\
        as well as needed parameters: ('interest rate', 'spot', 'strike', 'time', 'sigma')"


def d1(r, spot, strike, time, sigma):
    return (np.log(spot / strike) + (r + sigma ** 2 / 2) * time) / (sigma * np.sqrt(time))


def d2(r, spot, strike, time, sigma):
    return d1(r, spot, strike, time, sigma) - sigma * np.sqrt(time)


def BSM(r, spot, strike, time, sigma, type):
    """
    Returns Black-Scholes-Merton Price of and Call/Put option
    """
    try:
        d1_value = d1(r, spot, strike, time, sigma)
        d2_value = d2(r, spot, strike, time, sigma)

        if type == 'call':
            bsm_price = spot * norm.cdf(d1_value, 0, 1) - strike * np.exp(-r * time) * norm.cdf(d2_value, 0, 1)
        elif type == 'put':
            bsm_price = strike * np.exp(-r * time) * norm.cdf(-d2_value, 0, 1) - spot * norm.cdf(-d1_value, 0, 1)

        return bsm_price

    except:
        raise Standard_Error_Output


def delta(r, spot, strike, time, sigma, type):
    """
    Returns Delta greek of an option given
    """
    d1_value = d1(r, spot, strike, time, sigma)

    try:
        if type == 'call':
            delta_value = norm.cdf(d1_value, 0, 1)
        elif type == 'put':
            delta_value = -norm.cdf(-d1_value, 0, 1)

        return delta_value

    except:
        raise Standard_Error_Output


def gamma(r, spot, strike, time, sigma, type):
    """
    Returns Gamma greek of an option given
    """
    d1_value = d1(r, spot, strike, time, sigma)

    try:
        gamma_value = norm.pdf(d1_value, 0, 1) / (spot * sigma * np.sqrt(time))
        return gamma_value

    except:
        raise Standard_Error_Output


def theta(r, spot, strike, time, sigma, type):
    """
    Returns Theta greek of an option given
    """
    d1_value = d1(r, spot, strike, time, sigma)
    d2_value = d2(r, spot, strike, time, sigma)

    try:
        if type == 'call':
            theta_value = -spot * norm.pdf(d1_value, 0, 1) * sigma / (2 * np.sqrt(time)) - \
                          r * strike * np.exp(-r * time) * norm.cdf(d2_value, 0, 1)
        elif type == 'put':
            theta_value = - spot * norm.pdf(d1_value, 0, 1) * sigma / (2 * np.sqrt(time)) + \
                          r * strike * np.exp(- r * time) * norm.cdf(-d2_value, 0, 1)

        return theta_value
    except:
        raise Standard_Error_Output


def rho(r, spot, strike, time, sigma, type):
    """
    Returns Rho greek of an option given
    """
    d2_value = d2(r, spot, strike, time, sigma)
    # rho_calc = K*T*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    try:
        if type == 'call':
            rho_value = strike * time * np.exp(-r * time) * norm.cdf(d2_value, 0, 1)
        elif type == 'put':
            rho_value = -strike * time * np.exp(-r * time) * norm.cdf(-d2_value, 0, 1)

        return rho_value * 0.01

    except:
        raise Standard_Error_Output


def vega(r, spot, strike, time, sigma, type):
    """
    Returns Vega greek of an option given
    """
    d1_value = d1(r, spot, strike, time, sigma)

    try:
        vega_value = spot * norm.pdf(d1_value, 0, 1) * np.sqrt(time)
        return vega_value * 0.01
    except:
        raise Standard_Error_Output


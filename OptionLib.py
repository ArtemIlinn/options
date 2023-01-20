from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime


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
        self.vega = vega(self.r, self.spot, self.strike, self.time, self.sigma)
        return self.vega


Standard_Error_Output = f"Please make sure you use correct types of options ('call'/'put')," \
                        f" as well as needed parameters."


def PutCallParity(c, p, spot, strike, r, time):
    """
    Checks if Put-Call Parity holds, returns True is so, False otherwise
    """
    return c + strike * np.exp(-r * time) == p + spot


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
        raise Exception(Standard_Error_Output)


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
        raise Exception(Standard_Error_Output)


def gamma(r, spot, strike, time, sigma, type):
    """
    Returns Gamma greek of an option given
    """
    d1_value = d1(r, spot, strike, time, sigma)

    try:
        gamma_value = norm.pdf(d1_value, 0, 1) / (spot * sigma * np.sqrt(time))
        return gamma_value

    except:
        raise Exception(Standard_Error_Output)


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
        raise Exception(Standard_Error_Output)


def rho(r, spot, strike, time, sigma, type):
    """
    Returns Rho greek of an option given
    """
    d2_value = d2(r, spot, strike, time, sigma)
    
    try:
        if type == 'call':
            rho_value = strike * time * np.exp(-r * time) * norm.cdf(d2_value, 0, 1)
        elif type == 'put':
            rho_value = -strike * time * np.exp(-r * time) * norm.cdf(-d2_value, 0, 1)

        return rho_value * 0.01

    except:
        raise Exception(Standard_Error_Output)


def vega(r, spot, strike, time, sigma):
    """
    Returns Vega greek of an option given
    """
    d1_value = d1(r, spot, strike, time, sigma)

    try:
        vega_value = spot * norm.pdf(d1_value, 0, 1) * np.sqrt(time)
        return vega_value * 0.01
    except:
        raise Exception(Standard_Error_Output)

'''
Greek Graphs:
'''


def greek_func(greek_name, r, spot, strike=55, time=1, sigma=0.08, type='call'):
    if greek_name == 'delta':
        return delta(r, spot, strike, time, sigma, type)
    elif greek_name == 'gamma':
        return gamma(r, spot, strike, time, sigma, type)
    elif greek_name == 'rho':
        return rho(r, spot, strike, time, sigma, type)
    elif greek_name == 'theta':
        return theta(r, spot, strike, time, sigma, type)
    elif greek_name == 'vega':
        return vega(r, spot, strike, time, sigma)
    else:
        raise Exception("Greek has to be one of 'delta', 'gamma', 'rho', 'theta', 'vega'")


def greek_plot(df_prices, r=0.03, strike=55, time=1, sigma=0.08, type='both', greek_name='delta'):
    if 'Spot' not in df_prices.columns:
        raise Exception("Column 'Spot' is not in the data frame")

    df_pr = pd.DataFrame()

    if type == 'both':

        df_pr["{greek_name} for {type} {vol}% Vol".format(greek_name=greek_name, type='call', vol=sigma * 100)] = \
            df_prices['Spot'].apply(lambda x: greek_func(greek_name, r=r, spot=x, strike=strike,
                                                         time=time, sigma=sigma, type='call'))

        df_pr["{greek_name} for {type} {vol}% Vol".format(greek_name=greek_name, type='put', vol=sigma * 100)] = \
            df_prices['Spot'].apply(lambda x: greek_func(greek_name, r=r, spot=x, strike=strike,
                                                         time=time, sigma=sigma, type='put'))

    elif type == 'call' or type == 'put':
        df_pr["{greek_name} for {type} {vol}% Vol".format(greek_name=greek_name, type=type, vol=sigma * 100)] = \
            df_prices['Spot'].apply(lambda x: greek_func(greek_name, r=r, spot=x, strike=strike,
                                                         time=time, sigma=sigma, type=type))

    else:
        raise Exception(Standard_Error_Output)

    ax = df_pr.plot(kind='line')
    ax.axhline(0, linestyle='--', color='grey')
    plt.xlabel("Spot")
    plt.ylabel(greek_name)
    plt.title(f"Black-Scholes {greek_name}")
    plt.show()


def speed(r, spot, strike, time, sigma, type, D):
    """
    Returns the speed of an option - the rate of change of the gamma with respect to the stock
    price.
    """
    if type == 'call' or type == 'put':
        d1_value = d1(r, spot, strike, time, sigma)
        return -np.exp(-D * time) * norm.pdf(d1_value, 0, 1) / (spot**2 * sigma**2 * time) * (d1_value * np.sqrt(time))

    else:
        raise Exception(Standard_Error_Output)


def get_chain_yf():
    pass


def implied_vol(market_price, r, spot, strike, time, tol=0.00001, iterations=100):
    """
        Return Implied Volatility of European Option using Newton Raphson method,
        knowing standard parameters with addition of error tolerance and number of iterations.
        And choosing sigma to estimate:
    """

    sigma = 0.3  ## initial volatility guess

    for iteration in range(iterations):

        difference = BSM(r, spot, strike, time, sigma, type) - market_price

        if abs(difference) < tol:
            break

        sigma = sigma - difference / vega(r, spot, strike, time, sigma)

    return sigma  # IV




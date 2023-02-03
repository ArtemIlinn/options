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


def BSM(r, spot, strike, time, sigma, type='call'):
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


'''
Greeks:
'''


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
    # rho_calc = K*T*np.exp(-r*T)*norm.cdf(d2, 0, 1)
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
Greeks Graphs:
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


def imp_vol(market_price, spot, strike, time, r, type, precision=1.0e-5, iterations=200):

    """
        Return Implied Volatility of European Option using Newton Raphson method,
        knowing standard parameters with addition of error tolerance and number of iterations.
        And choosing sigma to estimate:
    """
    sigma = 0.5
    for i in range(iterations):

        price = BSM(r, spot, strike, time, sigma, type)
        vega_v = vega(r, spot, strike, time, sigma) * 100
        diff = market_price - price

        if abs(diff) < precision:
            return sigma

        sigma = sigma + diff / vega_v

    return sigma  # Best guess


def plot_iv_skew(date, threshold, df):
    """
    :param date: string in format of YYYY-MM-DD
    :param threshold: a criterion of low volatility filtration
    :param df: Options dataframe
    :return: plots a graph of Implied Volatility Skew
    """

    df_date = df[df["expDate"] == f'{date} 23:59:59'.format(date)]
    df_date_ = df_date[df_date.impliedVolatility >= threshold]

    df_date_[["strike", "impliedVolatility"]].set_index("strike").plot(title="Implied Volatility Skew", figsize=(8, 5))


def plot_iv_term(strike_value, threshold, df):
    """
    :param strike_value: value of a strike
    :param threshold: a criterion of low volatility filtration
    :param df: Options dataframe
    :return: plots a graph of Implied Volatility Term Structure
    """

    df_strike = df[df["strike"] == strike_value]
    df_strike_ = df_strike[df_strike.impliedVolatility >= threshold]

    df_strike_[["expDate", "impliedVolatility"]].set_index("expDate").plot(title="Implied Volatility Term Structure",
                                                                           figsize=(8, 5))


def plot_iv_surface(df):
    """
    :param df: Options Chains Dataframe
    :return: Plots a Graph of Implied Volatility Surface
    """

    vol_surface = (df[['daysToExp', 'strike', 'impliedVolatility']].pivot_table(
        values='impliedVolatility', index='strike', columns='daysToExp').dropna())

    plot = plt.figure(figsize=(8, 8))
    ax = plot.add_subplot(111, projection='3d')

    x, y, z = vol_surface.columns.values, vol_surface.index.values, vol_surface.values
    X, Y = np.meshgrid(x, y)

    ax.set_xlabel('Days to Expiration')
    ax.set_ylabel('Strike')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Implied Volatility Surface')

    ax.plot_surface(X, Y, z)


def yf_get_chains(ticker):
    """
    Loads option chains data for a particular ticker from Yahoo Finance
    """

    underlying = yf.Ticker(ticker)  # addressing the underlying asset of an options
    expiration_dates = underlying.options  # and dates of expiration

    opt_data = pd.DataFrame()  # constructing the dataframe we will return

    for ed in expiration_dates:
        opt = underlying.option_chain(ed)  # getting data for a specific date

        calls, puts = opt.calls, opt.puts  # call and puts
        calls['optionType'], puts['optionType'] = 'call', 'puts'  # adding types of options to the df

        ed_chain = pd.concat([calls, puts])
        ed_chain['expDate'] = pd.to_datetime(ed) + pd.DateOffset(hours=23, minutes=59, seconds=59)
        # adding date of expiration

        opt_data = pd.concat([opt_data, ed_chain])  # adding data to the main df

    opt_data['daysToExp'] = (opt_data.expDate - datetime.datetime.today()).dt.days + 1
    # additionally let's compute days until expiration date

    return opt_data



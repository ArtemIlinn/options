from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import scipy.stats as st
import scipy.optimize as optimize
import enum


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
        return -np.exp(-D * time) * norm.pdf(d1_value, 0, 1) / (spot ** 2 * sigma ** 2 * time) * (
                d1_value * np.sqrt(time))

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


def heston_simulation(spot_0, v_0, r, rho, kappa, theta, sigma, time_y, steps, simulations):
    """
    :param spot_0: spot price
    :param v_0: initial variance
    :param r: risk-free rate
    :param rho: correlation value
    :param kappa: rate of mean reversion
    :param theta: long-term mean of variance
    :param sigma: volatility of volatility
    :param time_y: time in years
    :param steps: number of steps in simulation
    :param simulations: number of simulations
    :return: performs Heston simulations and return history of asset prices and variances
    """

    dt = time_y / steps

    mu = np.array([0, 0])
    cov = np.array([[1, rho], [rho, 1]])

    # storing history of asset prices and variances
    spot = np.full(shape=(steps + 1, simulations), fill_value=spot_0)
    variance = np.full(shape=(steps + 1, simulations), fill_value=v_0)

    z = np.random.multivariate_normal(mu, cov, (steps, simulations))

    for i in range(1, steps + 1):
        spot[i] = spot[i - 1] * np.exp(
            (r - 0.5 * variance[i - 1]) * dt + np.sqrt(variance[i - 1] * dt) * z[i - 1, :, 0])
        variance[i] = np.maximum(variance[i - 1] + kappa * (theta - variance[i - 1]) * dt +
                                 sigma * np.sqrt(variance[i - 1] * dt) * z[i - 1, :, 1], 0)

    return spot, variance


def plot_heston_simulation(spot_0, v_0, r, rho, kappa, theta, sigma, time_y, steps, simulations, fig_size=(12, 4)):
    spots, variances = heston_simulation(spot_0, v_0, r, rho, kappa, theta, sigma, time_y, steps, simulations)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    t = np.linspace(0, time_y, steps + 1)

    ax1.plot(t, spots)

    ax1.set_title('Heston Model Asset Prices')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Asset Prices')

    ax2.plot(t, variances)

    ax2.set_title('Heston Model Variance Process')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Variance')

    plt.show()


complex_i = complex(0, 1)


def heston_func(s, spot, strike, r, time, v, sigma, kappa, theta, rho):
    """
    Supporting function for Heston pricing
    :param s: s1 or s2 from Heston_price()
    :param spot: price of an asset at the time
    :param strike: strike of an option
    :param r: risk-free rate
    :param time: the maturity
    :param v: v
    :param sigma: volatility of volatility
    :param kappa: rate of mean reversion
    :param theta: long-term mean of variance
    :param rho: correlation value
    :return: Supporting function for Heston pricing.
    """

    d1 = (rho * sigma * complex_i * s - kappa) ** 2
    d2 = (sigma ** 2) * (i * s + s ** 2)
    d = np.sqrt(d1 + d2)

    g1 = kappa - rho * sigma * complex_i * s - d
    g2 = kappa - rho * sigma * complex_i * s + d
    g = g1 / g2

    exp11 = np.exp(np.log(spot) * i * s) * np.exp(i * s * r * time)
    exp12 = 1 - g * np.exp(-d * time)
    exp13 = 1 - g
    exp1 = exp11 * np.power(exp12 / exp13, -2 * theta * kappa / (sigma ** 2))

    exp21 = theta * kappa * time / (sigma ** 2)
    exp22 = v / (sigma ** 2)
    exp23 = (1 - np.exp(-d * time)) / (1 - g * np.exp(-d * time))
    exp2 = np.exp((exp21 * g1) + (exp22 * g1 * exp23))

    return exp1 * exp2


def heston_price(spot, strike, r, time, v, sigma, kappa, theta, rho):
    """
    :param s: s1 or s2 from Heston_price()
    :param spot: price of an asset at the time
    :param strike: strike of an option
    :param r: risk-free rate
    :param time: the maturity
    :param v: v
    :param sigma: volatility of volatility
    :param kappa: rate of mean reversion
    :param theta: long-term mean of variance
    :param rho: correlation value
    :return: Pricing using heston model.
    """

    P, iterations, max_number = 0, 1000, 100
    ds = max_number / iterations

    part_1 = 0.5 * (spot - strike * np.exp(-r * time))

    for i in range(1, iterations):
        s1 = ds * (2 * i + 1) * 0.5
        s2 = s1 - complex_i

        numerator_1 = heston_func(s2, spot, strike, r, time, v, sigma, kappa, theta, rho)
        numerator_2 = strike * heston_func(s1, spot, strike, r, time, v, sigma, kappa, theta, rho)

        denominator = np.exp(np.log(strike) * complex_i * s1) * complex_i * s1

        P += ds * (numerator_1 - numerator_2) / denominator

    part_2 = P / np.pi

    return np.real((part_1 + part_2))


def hagan_iv(K, T, f, alpha, beta, rho, gamma):
    # We make sure that the input is of array type

    if type(K) == float:
        K = np.array([K])
    if K is not np.array:
        K = np.array(K).reshape([len(K), 1])

    # The strike prices cannot be too close to 0

    K[K < 1e-10] = 1e-10

    z = gamma / alpha * np.power(f * K, (1.0 - beta) / 2.0) * np.log(f / K)
    x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    A = alpha / (np.power(f * K, ((1.0 - beta) / 2.0)) * (1.0 + np.power(1.0 - beta, 2.0) / 24.0 *
                                                          np.power(np.log(f / K), 2.0) + np.power((1.0 - beta),
                                                                                                  4.0) / 1920.0 *
                                                          np.power(np.log(f / K), 4.0)))
    B1 = 1.0 + (np.power((1.0 - beta), 2.0) / 24.0 * alpha * alpha / (np.power((f * K),
                                                                               1 - beta)) + 1 / 4 * (
                        rho * beta * gamma * alpha) / (np.power((f * K),
                                                                ((1.0 - beta) / 2.0))) + (
                        2.0 - 3.0 * rho * rho) / 24.0 * gamma * gamma) * T
    impVol = A * (z / x_z) * B1
    B2 = 1.0 + (np.power(1.0 - beta, 2.0) / 24.0 * alpha * alpha /
                (np.power(f, 2.0 - 2.0 * beta)) + 1.0 / 4.0 * (rho * beta * gamma *
                                                               alpha) / np.power(f, (1.0 - beta)) + (
                        2.0 - 3.0 * rho * rho) / 24.0 * gamma * gamma) * T

    # Special treatment of ATM strike value

    impVol[np.where(K == f)] = alpha / np.power(f, (1 - beta)) * B2
    return impVol


def LocalVarianceBasedOnSABR(s0, frwd, r, alpha, beta, rho, volvol):
    # Define shock size for approximating derivatives

    dt = 0.001
    dx = 0.001

    # Function for Hagan's implied volatility approximation

    sigma = lambda x, t: hagan_iv(x, t, frwd, alpha, beta, rho, volvol)

    # Derivatives

    dsigmadt = lambda x, t: (sigma(x, t + dt) - sigma(x, t)) / dt
    dsigmadx = lambda x, t: (sigma(x + dx, t) - sigma(x - dx, t)) / (2.0 * dx)
    d2sigmadx2 = lambda x, t: (sigma(x + dx, t) + sigma(x - dx, t) - 2.0 * sigma(x, t)) / (dx * dx)
    omega = lambda x, t: sigma(x, t) * sigma(x, t) * t
    domegadt = lambda x, t: sigma(x, t) ** 2.0 + 2.0 * t * sigma(x, t) * dsigmadt(x, t)
    domegadx = lambda x, t: 2.0 * t * sigma(x, t) * dsigmadx(x, t)
    # d2omegadx2 = lambda x,t: 2.0*t*(dsigmadx(x,t))**2.0 + 2.0*t*sigma(x,t)*d2sigmadx2(x,t)
    d2omegadx2 = lambda x, t: 2.0 * t * np.power(dsigmadx(x, t), 2.0) + 2.0 * t * sigma(x, t) * d2sigmadx2(x, t)

    term1 = lambda x, t: 1.0 + x * domegadx(x, t) * (0.5 - np.log(x / (s0 * np.exp(r * t))) / omega(x, t))
    term2 = lambda x, t: 0.5 * np.power(x, 2.0) * d2omegadx2(x, t)
    term3 = lambda x, t: 0.5 * np.power(x, 2.0) * np.power(domegadx(x, t), 2.0) * (
            -1.0 / 8.0 - 1.0 / (2.0 * omega(x, t)) \
            + np.log(x / (s0 * np.exp(r * t))) * np.log(x / (s0 * np.exp(r * t))) / (2 * omega(x, t) * omega(x, t)))

    # Final expression for local variance

    sigmalv2 = lambda x, t: (domegadt(x, t) + r * x * domegadx(x, t)) / (term1(x, t) + term2(x, t) + term3(x, t))
    return sigmalv2


def BS_Call_Put_Option_Price(CP, S_0, K, sigma, tau, r):
    if K is list:
        K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0))
          * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if CP == 'call':
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == 'put':
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S_0
    return value


# Implied volatility method

def ImpliedVolatility(CP, marketPrice, K, T, S_0, r):
    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0, 2, 200)
    optPriceGrid = BS_Call_Put_Option_Price(CP, S_0, K, sigmaGrid, T, r)
    sigmaInitial = np.interp(marketPrice, optPriceGrid, sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))

    # Use already determined input for the local-search (final tuning)

    func = lambda sigma: np.power(BS_Call_Put_Option_Price(CP, S_0, K, sigma, T, r) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol


def EUOptionPriceFromMCPaths(CP, S, K, T, r):
    # S is a vector of Monte Carlo samples at T

    if CP == 'call':
        return np.exp(-r * T) * np.mean(np.maximum(S - K, 0.0))
    elif CP == 'put':
        return np.exp(-r * T) * np.mean(np.maximum(K - S, 0.0))


def mainCalculationSABR(beta=1.0, rho=0.0, volvol=0.2, s0=1.0, T=10.0, r=0.05, alpha=0.2, CP='call', NoOfPaths=25000):
    # For the SABR model we take beta =1 and rho =0 (as simplification)

    # Other model parameters
    f_0 = s0 * np.exp(r * T)

    # Monte Carlo settings

    NoOfSteps = (int)(100 * T)

    # We define the market to be driven by Hagan's SABR formula
    # Based on this formula we derive the local volatility/variance

    sigma = lambda x, t: hagan_iv(x, t, f_0, alpha, beta, rho, volvol)

    # Local variance based on the Hagan's SABR formula

    sigmalv2 = LocalVarianceBasedOnSABR(s0, f_0, r, alpha, beta, rho, volvol)

    # Monte Carlo simulation

    dt = T / NoOfSteps
    np.random.seed(4)
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    S = np.zeros([NoOfPaths, NoOfSteps + 1])

    S[:, 0] = s0;
    time = np.zeros([NoOfSteps + 1, 1])

    for i in range(0, NoOfSteps):

        # This condition is necessary as for t=0 we cannot compute implied
        # volatilities

        if time[i] == 0.0:
            time[i] = 0.0001

        # print('current time is {0}'.format(time[i]))

        # Standarize Normal(0,1)

        Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

        # Compute local volatility

        S_i = np.array(S[:, i]).reshape([NoOfPaths, 1])
        temp = sigmalv2(S_i, time[i])
        sig = np.real(temp)
        np.nan_to_num(sig)

        # Because of discretizations we may encouter negative variance which
        # is set to 0 here.

        sig = np.maximum(sig, 1e-14)
        sigmaLV = np.sqrt(sig)

        # Stock path

        S[:, i + 1] = S[:, i] * (1.0 + r * dt + sigmaLV.transpose() * Z[:, i] * np.sqrt(dt))

        # We force that at each time S(t)/M(t) is a martingale

        S[:, i + 1] = S[:, i + 1] - np.mean(S[:, i + 1]) + s0 * np.exp(r * time[i])

        # Make sure that after moment matching we don't encounter negative stock values

        S[:, i + 1] = np.maximum(S[:, i + 1], 1e-14)

        # Adjust time

        time[i + 1] = time[i] + dt

    # Plot some results

    K = np.linspace(0.2, 5.0, 25)
    # c_n = np.array([-1.5, -1.0, -0.5,0.0, 0.5, 1.0, 1.5])
    # K= s0*np.exp(r*T) * np.exp(0.1 * c_n * np.sqrt(T))
    OptPrice = np.zeros([len(K), 1])
    IV_Hagan = np.zeros([len(K), 1])
    IV_MC = np.zeros([len(K), 1])
    for (idx, k) in enumerate(K):
        OptPrice[idx] = EUOptionPriceFromMCPaths(CP, S[:, -1], k, T, r)
        IV_Hagan[idx] = sigma([k], T) * 100.0
        IV_MC[idx] = ImpliedVolatility(CP, OptPrice[idx], k, T, s0, r) * 100.0

    # Plot the option prices

    plt.figure(1)
    plt.plot(K, OptPrice)
    plt.grid()
    plt.xlabel('strike')
    plt.ylabel('option price')

    # Plot the implied volatilities

    plt.figure(2)
    plt.plot(K, IV_Hagan)
    plt.plot(K, IV_MC, '-r')
    plt.grid()
    plt.xlabel('strike')
    plt.ylabel('implied volatility')
    plt.legend(['Hagan', 'Monte Carlo'])
    plt.axis([np.min(K), np.max(K), 0, 40])


def crank_A_mat(m, lambda_):
    mat_A = np.zeros((m + 1, m + 1))

    for i in range(m):
        mat_A[i][i] = 1 + lambda_
        mat_A[i][i + 1] = - lambda_ * .5
        mat_A[i + 1][i] = - lambda_ * .5

    mat_A[m][m] = 1 + lambda_
    return mat_A


def crank_B_mat(m, lambda_):
    mat_B = np.zeros((m + 1, m + 1))

    for i in range(m):
        mat_B[i][i] = 1 - lambda_
        mat_B[i][i + 1] = lambda_ * .5
        mat_B[i + 1][i] = lambda_ * .5
    mat_B[m][m] = 1 - lambda_
    return mat_B


def crank_call(w, A, B, k, w_m, w_0, nu, lambda_):
    d = np.zeros(len(w))

    d[0] = lambda_ / 2 * (w_0[nu] + w_0[nu + 1])
    d[-1] = lambda_ / 2 * (w_m[nu] + w_m[nu + 1])

    return np.linalg.inv(A).dot(B.dot(w) + d)


def get_call_maturity_condition(x_min, x_max, delta_x, k):
    x = np.arange(x_min, x_max + delta_x, delta_x)
    payoff = np.exp((k + 1) * x * .5) - np.exp((k - 1) * x * .5)

    return np.maximum(payoff, 0)


def crank_price_eu_call(K, T, r, sigma, x_min, x_max, delta_x, delta_tau):
    k = 2 * r / sigma ** 2
    tau_max = T * (sigma ** 2) * .5

    x = np.arange(x_min, x_max + delta_x, delta_x)
    m = int((x_max - x_min) / delta_x)

    lambda_ = delta_tau / (delta_x ** 2)
    tau_array = np.arange(0, tau_max + delta_tau, delta_tau)

    mat_A = crank_A_mat(m, lambda_)
    mat_B = crank_B_mat(m, lambda_)

    w = get_call_maturity_condition(x_min, x_max, delta_x, k)
    w_m = np.exp((k + 1) * x_max * .5 + ((k + 1) ** 2) * tau_array / 4)
    w_0 = np.zeros(len(w_m))

    for nu in range((len(w_m)) - 1):
        w = crank_call(w, mat_A, mat_B, k, w_m, w_0, nu, lambda_)

    alpha = (1 - k) * .5
    beta = - (k + 1) ** 2 / 4

    V = K * np.exp(alpha * x + beta * tau_max) * w
    S = K * np.exp(x)

    return S, V


def plot_crank_(r=0.01, sigma=0.3, K=100, T=1, x_min=-5, x_max=5, delta_x=0.01, delta_tau=0.002):
    """
    Plots Cran
    """

    S, V = crank_price_eu_call(K, T, r, sigma, x_min, x_max, delta_x, delta_tau)

    plt.figure(figsize=(20, 10))
    plt.plot(S[(S < 200) & (S > 10)], V[(S < 200) & (S > 10)], 'C3', markersize=30,
             label=f'European Call Strike K={K}'.format(K))
    plt.legend(fontsize=20)
    plt.xlabel('Underlying Spot', color='white', fontsize=20)
    plt.grid()
    plt.show()


def hellothere():
    pass

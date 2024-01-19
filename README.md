# :feather: System of Options Pricing and Trading.

The objective of this project is to describe how the pricing, valuation, and risk management of vanilla stock options , as well as to present software, so that the approaches may be applied in practice. As a scientific task – after solving the problem of delta hedging, conducting computational experiments.

Project also includes OptionLib, which is package with operating functions for working with options, volatility surface construction, stochastic models, Black-Scholes, and more...

## Analytis and Results

👩‍🌾 [Presentation](https://github.com/ArtemIlinn/options/blob/main/Ilin%20Artem%20DSBA%20212%20Option%20pricing%20system%20presentation%202023pdf.pdf)

:fax: [Report](https://github.com/ArtemIlinn/options/blob/main/IlinArtemOptionPricingSystemReport%20_.pdf)

:scroll: [Article](https://github.com/ArtemIlinn/options/blob/main/Ilin%20Lukyanchenko%20article.pdf) - paper on Predicting Structural Breaks of Option Price with Implied Volatility

```OptionLib.py``` - python library with content and instruments of this repository

## Data Aggregation

```1.1 YahooFinance Option Chain.ipynb``` - getting option chains from YahooFinance

```1.2 MOEX Option Chain.ipynb``` - parsing option chains from Moscow Stock Exchange 

```1.3 Theta Data - Historical Data.ipynb``` - data mining techiques unsing Theta Data 

## Black–Scholes Model

```2.1 BSM Greeks Graphs.ipynb``` - option greeks

```2.2 Volatility.ipynbb``` - volatility: implied, skew, surface

## Stochastic & Numerical Models

```3.1. Heston.ipynb``` - Heston model & calibration

```3.2.0 SABR Calibrate.ipynb``` - SABR model & calibration

```3.3 Crank Nickolson.ipynb``` - Crank Nickolson model

## Hedging

```4.1 Delta Hedge.ipynb``` - constructing the portfolio, delta hedging simulaton

```4.2 Deltas.ipynb``` - Heston delta, SABR delta

```4.2.1 Deltas.ipynb``` - performing delta hedge

```4.3 Delta hedge & volatility.ipynb``` - performing delta hedge with volatility

## Structural Breaks

```5.1 Bifurcation.ipynb``` - Bifurcation analysis, structural breaks detection, algorithms, Chow test, Prophet

```5.2 Changing points & Vol.ipynb``` - change points for option price by Pelt, what happens with IV at change point: Regression, ARIMA, Parametric test

```5.2.1 Change Ps, Vol (AAPL230616C00150000).ipynb``` - experemrnts with option AAPL230616C00150000

```5.2.2 Change Ps, Vol (JPM240119C00150000).ipynb``` - experemrnts with option JPM240119C00150000

## Predictions & Experements

```6.1 Predictions.ipynb``` - LSTM predictions with and without volatility as feature

```6.1.1 More experiments.ipynb``` - more LSTM predictions with and without volatility as feature, options: JPM240119C00150000, AAPL230616C00150000, 

![alt text](https://cdn.phenompeople.com/CareerConnectResources/prod/IMC1GLOBAL/images/Untitleddesign57-1632396334098.png)




import time
import datetime as dt
from math import sqrt, pi
import numpy as np

import scipy
from scipy.stats import norm


#underlying stock price
S = 65.0

# strike price
K = 65.0

# time to expiration
t = 180.0 / 365.0

# risk free rate
r = 0.015

# volatility 
vol = 0.3


#Normal cumulative density function
def N(z):

    return norm.cdf(z)

# helper function phi
def phi(x):
    
    return np.exp(-0.5 * x * x) / (sqrt(2.0 * pi))

# shared
def gamma(S, K, r, t, vol): 
    
    d1 =  (np.log(S / K) + (r + (vol ** 2.0) / 2.0) * t) * (1.0 / (vol * np.sqrt(t)))

    return phi(d1) / (S * vol * sqrt(t))

def vega(S, K, r, t, vol): 
    
    d1 =  (np.log(S / K) + (r + (vol ** 2.0) / 2.0) * t) * (1.0 / (vol * np.sqrt(t))) 

    return (S * phi(d1) * sqrt(t)) / 100.0

# call options
def call_delta(S, K, r, t, vol): 
    d1 =  (np.log(S / K) + (r + (vol ** 2.0) / 2.0) * t) * (1.0 / (vol * np.sqrt(t))) 

    return N(d1)



def call_theta(S, K, r, t, vol): 

    d1 =  (np.log(S / K) + (r + (vol ** 2.0) / 2.0) * t) * (1.0 / (vol * np.sqrt(t)))
    d2 = d1 - (vol * np.sqrt(t))
    theta = -((S * phi(d1) * vol) / (2.0 * np.sqrt(t))) - (r * K * np.exp(-r * t) * N(d2))

    return theta / 365.0

def call_rho(S, K, r, t, vol):
    
    d1 =  (np.log(S / K) + (r + (vol ** 2.0) / 2.0) * t) * (1.0 / (vol * np.sqrt(t)))
    d2 = d1 - (vol * np.sqrt(t))
    rho = K * t * np.exp(-r * t) * N(d2) 

    return rho / 100.0

# put options
def put_delta(S, K, r, t, vol): 

    d1 =  (np.log(S / K) + (r + (vol ** 2.0) / 2.0) * t) * (1.0 / (vol * np.sqrt(t)))

    return N(d1) - 1.0


def put_theta(S, K, r, t, vol): 
   
    d1 =  (np.log(S / K) + (r + (vol ** 2.0) / 2.0) * t) * (1.0 / (vol * np.sqrt(t))) 
    d2 = d1 - (vol * np.sqrt(t))
    theta = -((S * phi(d1) * vol) / (2.0 * np.sqrt(t))) + (r * K * np.exp(-r * t) * N(-d2))
    
    return theta / 365.0

def put_rho(S, K, r, t, vol):
    
    d1 =  (np.log(S / K) + (r + (vol ** 2.0) / 2.0) * t) * (1.0 / (vol * np.sqrt(t))) 
    d2 = d1 - (vol * np.sqrt(t))
    rho = -K * t * np.exp(-r * t) * N(-d2) 

    return rho / 100.0


# print the results
print("Black-Scholes call delta %0.3f" % call_delta(S, K, r, t, vol)) 
print("Black-Scholes put delta %0.3f" % put_delta(S, K, r, t, vol)) 
print("Black-Scholes gamma %0.3f" % gamma(S, K, r, t, vol)) 
print("Black-Scholes vega %0.3f" % vega(S, K, r, t, vol)) 
print("Black-Scholes call theta %0.3f" % call_theta(S, K, r, t, vol)) 
print("Black-Scholes put theta %0.3f" % put_theta(S, K, r, t, vol)) 
print("Black-Scholes call rho %0.3f" % call_rho(S, K, r, t, vol)) 
print("Black-Scholes put rho %0.3f" % put_rho(S, K, r, t, vol))


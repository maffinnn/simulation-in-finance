# Parallel computation using numba
from numba import prange
import numpy as np
import logging
import sys
import cmath

# Optimizer
from lmfit import Parameters, minimize
import pandas as pd

# # logger_yq = logging.getLogger('yq')
epsilon = sys.float_info.epsilon  # Very small value to avoid division by zero or log of zero

# for handler in # logger_yq.handlers[:]:  # Iterate over a copy of the handlers list
#     if isinstance(handler, logging.FileHandler):
#         # logger_yq.removeHandler(handler)  # Remove only the file handler


i = complex(0,1)

# To be used in the Heston pricer
# @jit
def fHeston(s, St, K, r, T, sigma, kappa, theta, volvol, rho):
    """
    This function calculates the Heston model integrand for option pricing.

    Parameters:
    s (complex): Complex variable for integration.
    St (float): Current stock price.
    K (float): Strike price.
    r (float): Risk-free interest rate.
    T (float): Time to maturity.
    sigma (float): Volatility of the stock.
    kappa (float): Mean-reversion speed.
    theta (float): Long-term mean of variance.
    volvol (float): Volatility of volatility.
    rho (float): Correlation between stock and variance.

    Returns:
    complex: The value of the Heston model integrand.
    """
    # logger_yq.info(f"Received parameters for fHeston - s: {s}, St: {St}, K: {K}, r: {r}, T: {T}, sigma: {sigma}, kappa: {kappa}, theta: {theta}, volvol: {volvol}, rho: {rho}")

    prod = rho * sigma * i * s
    # logger_yq.info(f"Calculated prod: {prod}")

    d1 = pow(prod - kappa, 2)
    d2 = pow(sigma, 2) * (i * s + pow(s, 2))
    d = cmath.sqrt(d1 + d2)
    # logger_yq.info(f"Calculated d components - d1: {d1}, d2: {d2}, d: {d}")

    g1 = kappa - prod - d
    g2 = kappa - prod + d
    g = g1 / g2 if g2 != 0 else np.inf
    # logger_yq.info(f"Calculated g components - g1: {g1}, g2: {g2}, g: {g}")

    exp1 = np.exp(np.log(St) * i * s) * np.exp(i * s * r * T)
    exp2 = 1 - g * np.exp(-d * T)
    exp3 = 1 - g
    mainExp1 = exp1 * pow(exp2 / exp3, -2 * theta * kappa / pow(sigma, 2)) if exp3 != 0 else np.inf
    # logger_yq.info(f"Calculated mainExp1: {mainExp1}")

    # Calculating exp4, exp5, exp6, and mainExp2
    exp4 = theta * kappa * T / pow(sigma, 2)
    # logger_yq.info(f"Calculated exp4: {exp4}")

    try:
        exp5 = volvol / pow(sigma, 2)
        # logger_yq.info(f"Calculated exp5: {exp5}")
    except ZeroDivisionError:
        # logger_yq.error("Error in calculating exp5: Division by zero due to sigma being zero")
        exp5 = np.inf  # or handle it in a way that suits your application
    except Exception as e:
        # logger_yq.error(f"Unexpected error in calculating exp5: {e}")
        exp5 = np.nan  # Handle unexpected errors

    try:
        exp6_denominator = 1 - g * np.exp(-d * T)
        exp6 = (1 - np.exp(-d * T)) / exp6_denominator
        # logger_yq.info(f"Calculated exp6: {exp6}")
    except ZeroDivisionError:
        # logger_yq.error("Error in calculating exp6: Division by zero")
        exp6 = np.inf  # or handle it appropriately
    except Exception as e:
        # logger_yq.error(f"Unexpected error in calculating exp6: {e}")
        exp6 = np.nan  # Handle unexpected errors

    mainExp2 = np.exp((exp4 * g1) + (exp5 * g1 * exp6))
    # logger_yq.info(f"Calculated mainExp2: {mainExp2}")

    result = mainExp1 * mainExp2
    # logger_yq.info(f"Final result: {result}")
    
    return result

# Heston Pricer (allow for parallel processing with numba)
# @jit(forceobj=True)
def priceHestonMid(St, K, r, T, sigma, kappa, theta, volvol, rho):
    """
    This function calculates the Heston model price for an option.

    Parameters:
    St (float): Current stock price.
    K (float): Strike price.
    r (float): Risk-free interest rate.
    T (float): Time to maturity.
    sigma (float): Volatility of the stock.
    kappa (float): Mean-reversion speed.
    theta (float): Long-term mean of variance.
    volvol (float): Volatility of volatility.
    rho (float): Correlation between stock and variance.

    Returns:
    float: The calculated option price using the Heston model.
    """
    # logger_yq.info("Starting priceHestonMid calculation")
    P, iterations, maxNumber = 0, 1000, 100
    ds = maxNumber / iterations

    element1 = 0.5 * (St - K * np.exp(-r * T))
    # logger_yq.info(f"Element 1 calculated: {element1}")

    # Calculate the complex integral
    for j in prange(1, iterations):
        s1 = ds * (2 * j + 1) / 2
        s2 = s1 - i
        
        numerator1 = fHeston(s2, St, K, r, T, sigma, kappa, theta, volvol, rho)
        numerator2 = K * fHeston(s1, St, K, r, T, sigma, kappa, theta, volvol, rho)

        denominator = np.exp(np.log(K + epsilon) * i * s1) * i * s1 + epsilon
        P += ds * (numerator1 - numerator2) / denominator
    
    element2 = P / np.pi
    # logger_yq.info(f"Element 2 calculated: {element2}")

    result = np.real((element1 + element2))
    # logger_yq.info(f"Final result: {result}")

    return result



def calibrate_heston(St: float, max_sigma: float, options_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function calibrates the Heston model to market data.

    Parameters:
    St (float): Current stock price.
    max_sigma (float): Maximum allowable sigma value.
    options_data (pd.DataFrame): Market data of options including maturities, strikes, and prices.

    Returns:
    pd.DataFrame: The calibrated parameters for the Heston model.

    The function uses the Levenberg Marquardt algorithm to find optimal Heston model parameters that best fit the given market data.
    """
    volSurfaceLong = options_data
    # Define global variables to be used in optimization
    maturities = volSurfaceLong['maturity'].to_numpy('float')
    strikes = volSurfaceLong['strike'].to_numpy('float')
    marketPrices = volSurfaceLong['price'].to_numpy('float')
    rates = volSurfaceLong['rate'].to_numpy('float')
    # Can be used for debugging
    def iter_cb(params, iter, resid):
        parameters = [params['kappa'].value, 
                      params['theta'].value, 
                      params['volvol'].value, 
                      params['rho'].value, 
                      params['sigma'].value,
                      np.sum(np.square(resid))]
        # logger_yq.info(f"The parameters in heston are: {parameters}") 
        
    # This is the calibration function
    def calibratorHeston(St, initialValues = [0.5,0.5,0.5,-0.5,0.1], 
                                lowerBounds = [1e-2,1e-2,1e-2,-1,1e-2], 
                                upperBounds = [10,10,10,0, max_sigma]): 

        '''
        Implementation of the Levenberg Marquardt algorithm in Python to find the optimal value 
        based on a given volatility surface.
        
        Function to be minimized:
            Error = (MarketPrices - ModelPrices)/MarketPrices
        
        INPUTS
        ===========
        1) Volatility Surface
            - Obtained from webscrapping. 
        
        2) Risk Free Curve
            - Obtained from webscrapping. 
            
        3) initialValues
            - Initialization values for the algorithms in this order:
                [sigma, kappa, theta, volvol, rho]
                
            - Default value: [0.1,0.1,0.1,0.1,0.1]
            
        4) lowerBounds
            -Fix lower limit for the values
            - Default value: [0.001,0.001,0.001,0.001,-1.00]
            
        5) upperBounds
            -Fix upper limit for the values
            - Default value: [1.0,1.0,1.0,1.0,1.0]
            
        6) St is the stock price today.
        
        Set Up
        =======
        1) We define the limits of the parameters using the Parameters object
        2) We define an objective function that gives the relative difference between market prices and model prices
        3) We minimize the function using the Levenberg Marquardt algorithm
        '''
        

        # Define parameters
        params = Parameters()
        params.add('kappa',value = initialValues[1], min = lowerBounds[1], max = upperBounds[1])
        params.add('theta',value = initialValues[2], min = lowerBounds[2], max = upperBounds[2])
        params.add('volvol', value = initialValues[3], min = lowerBounds[3], max = upperBounds[3])
        params.add('rho', value = initialValues[4], min = lowerBounds[4], max = upperBounds[4])
        params.add('sigma',value = initialValues[0], min = lowerBounds[0], max = upperBounds[0])
        
        # if np.isnan(list(params.valuesdict().values())).any():
        #     # logger_yq.warning("NaN detected in initial parameters:", params)
        
        # Define objective function
        objectiveFunctionHeston = lambda paramVect: (marketPrices - priceHestonMid(St, strikes,  
                                                                            rates, 
                                                                            maturities, 
                                                                            paramVect['sigma'].value,                         
                                                                            paramVect['kappa'].value,
                                                                            paramVect['theta'].value,
                                                                            paramVect['volvol'].value,
                                                                            paramVect['rho'].value))/(marketPrices + epsilon) 
        
        # Optimise parameters
        result = minimize(objectiveFunctionHeston, 
                        params, 
                        method = 'leastsq', # 'leastsq' previously
                        # iter_cb = iter_cb, # Can be commented if don't want debugging
                        max_nfev=500,
                        ftol = 1e-6)
        return(result)

    # Note: Only for demonstration purposes.
    # This calibration takes a while because the option surface contains 1000 points (About 30minutes)
    # Either reduce the number of options or implement an approximation
    # Best approximation: https://econ-papers.upf.edu/papers/1346.pdf
    result = calibratorHeston(St)
    return result
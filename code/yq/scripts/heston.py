# Parallel computation using numba
from numba import jit, prange
import numpy as np

# Optimizer
from lmfit import Parameters, minimize
import pandas as pd

i = complex(0,1)

# To be used in the Heston pricer
@jit
def fHeston(s, St, K, r, T, sigma, kappa, theta, volvol, rho):
    # To be used a lot
    prod = rho * sigma *i *s 
    
    # Calculate d
    d1 = (prod - kappa)**2
    d2 = (sigma**2) * (i*s + s**2)
    d = np.sqrt(d1 + d2)
    
    # Calculate g
    g1 = kappa - prod - d
    g2 = kappa - prod + d
    g = g1/g2
    
    # Calculate first exponential
    exp1 = np.exp(np.log(St) * i *s) * np.exp(i * s* r* T)
    exp2 = 1 - g * np.exp(-d *T)
    exp3 = 1- g
    mainExp1 = exp1 * np.power(exp2/ exp3, -2 * theta * kappa/(sigma **2))
    
    # Calculate second exponential
    exp4 = theta * kappa * T/(sigma **2)
    exp5 = volvol/(sigma **2)
    exp6 = (1 - np.exp(-d * T))/(1 - g * np.exp(-d * T))
    mainExp2 = np.exp((exp4 * g1) + (exp5 *g1 * exp6))
    
    return (mainExp1 * mainExp2)

# Heston Pricer (allow for parallel processing with numba)
@jit(forceobj=True)
def priceHestonMid(St, K, r, T, sigma, kappa, theta, volvol, rho):
    P, iterations, maxNumber = 0,1000,100
    ds = maxNumber/iterations
    
    element1 = 0.5 * (St - K * np.exp(-r * T))
    
    # Calculate the complex integral
    # Using j instead of i to avoid confusion
    for j in prange(1, iterations):
        s1 = ds * (2*j + 1)/2
        s2 = s1 - i
        
        numerator1 = fHeston(s2,  St, K, r, T, sigma, kappa, theta, volvol, rho)
        numerator2 = K * fHeston(s1,  St, K, r, T, sigma, kappa, theta, volvol, rho)
        denominator = np.exp(np.log(K) * i * s1) *i *s1
        
        P = P + ds *(numerator1 - numerator2)/denominator
    
    element2 = P/np.pi
    
    return np.real((element1 + element2))



def calibrate_heston(St: float, options_data: pd.DataFrame) -> pd.DataFrame:
    volSurfaceLong = options_data
    #Initialize parameters
    # sigma, kappa, theta, volvol, rho = 0.1, 0.1, 0.1, 0.1, 0.1

    # Define global variables to be used in optimization
    maturities = volSurfaceLong['maturity'].to_numpy('float')
    strikes = volSurfaceLong['strike'].to_numpy('float')
    marketPrices = volSurfaceLong['price'].to_numpy('float')
    rates = volSurfaceLong['rate'].to_numpy('float')
    # Can be used for debugging
    # def iter_cb(params, iter, resid):
    #     parameters = [params['sigma'].value, 
    #                   params['kappa'].value, 
    #                   params['theta'].value, 
    #                   params['volvol'].value, 
    #                   params['rho'].value, 
    #                   np.sum(np.square(resid))]
    #     print(parameters) 
        
    # This is the calibration function
    def calibratorHeston(St, initialValues = [0.5,0.5,0.5,0.5,-0.5], 
                                lowerBounds = [1e-2,1e-2,1e-2,1e-2,-1], 
                                upperBounds = [100,100,100,100,0]):
        # changed upper bound to 100
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
        params.add('sigma',value = initialValues[0], min = lowerBounds[0], max = upperBounds[0])
        params.add('kappa',value = initialValues[1], min = lowerBounds[1], max = upperBounds[1])
        params.add('theta',value = initialValues[2], min = lowerBounds[2], max = upperBounds[2])
        params.add('volvol', value = initialValues[3], min = lowerBounds[3], max = upperBounds[3])
        params.add('rho', value = initialValues[4], min = lowerBounds[4], max = upperBounds[4])
        
        
        # Define objective function
        objectiveFunctionHeston = lambda paramVect: (marketPrices - priceHestonMid(St, strikes,  
                                                                            rates, 
                                                                            maturities, 
                                                                            paramVect['sigma'].value,                         
                                                                            paramVect['kappa'].value,
                                                                            paramVect['theta'].value,
                                                                            paramVect['volvol'].value,
                                                                            paramVect['rho'].value))/marketPrices   
        
        # Optimise parameters
        result = minimize(objectiveFunctionHeston, 
                        params, 
                        method = 'leastsq',
    #                     iter_cb = iter_cb,
                        ftol = 1e-6)
        return(result)

    # Note: Only for demonstration purposes.
    # This calibration takes a while because the option surface contains 1000 points (About 30minutes)
    # Either reduce the number of options or implement an approximation
    # Best approximation: https://econ-papers.upf.edu/papers/1346.pdf
    result = calibratorHeston(St)
    return result
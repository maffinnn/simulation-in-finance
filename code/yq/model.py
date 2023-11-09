import pandas as pd
import typing
import constants as cs
import numpy as np
import utils

# In the main codes, we can create different models, and at the end access the sim_data to plot all the payoff_df aaginst the product prices
class PricingModel:
    def __init__(self, params: typing.Dict):
        self.data = params.get('data') # Stock price data for underlying asset (can include max data, in the training function can customise what dates to use)
        self.ticker_list = params.get('ticker_list')
        self.time_steps_per_year = 252
        self.dt = 1/self.time_steps_per_year
        self.num_ticker = len(self.ticker_list) # Number of stocks
        self.sim_data = None # To store simulated paths of the underlying assets
        self.payoff_df = None # To store payoff calculations based on simulated data
        self.Z_list = None

    # Implementation of the multi-asset Geometric Brownian Motion model
    def multi_asset_gbm(self, sim_start_date: pd.Timestamp, hist_window: int, sim_window: int) -> pd.DataFrame: 
        interest_rate = 1.750/100 

        try:
            last_avai_price_date = utils.add_trading_day(sim_start_date, -1)
            S_t = [self.data.loc[last_avai_price_date, ticker] for ticker in self.ticker_list] # Stock price of the 0th day of simulation
            
            hist_data = self.data[self.data.index < sim_start_date].tail(hist_window)
        
        except Exception as e:
            raise Exception("Error at wrangling historical data.")

        try:
            log_returns_list = []
            for ticker in self.ticker_list:
                # display(data[ticker], data[ticker].shift(1))
                log_returns = np.log(hist_data[ticker] / hist_data[ticker].shift(1)) # np.log is natural log, (P_i/P_i-1)
                log_returns.dropna(inplace = True) # A series
                log_returns_list.append(log_returns)
                # print(type(log_returns))

            # print(log_returns_list)
            # print(np.shape(log_returns_list))

            # print(f"np.array {np.array(log_returns_list)}")
        
        except Exception as e:
            raise Exception("Error at generating log return.")
        
        try:
            cov_matrix = np.cov(np.array(log_returns_list))
            print(f"Covariance matrix is:\n {cov_matrix}\n")
            print(f"The shape is {np.shape(cov_matrix)}\n")

            print(f"Correlation between the two var is {cov_matrix[0][1] / (cov_matrix[0][0] * cov_matrix[1][1]) ** 0.5}") # Correct

            L = np.linalg.cholesky(cov_matrix)
            print(f"The matrix after Cholesky decomposition is:\n {L}\n")

            print(f"The multiplication of L and L transpose is:\n {np.dot(L, L.T)}\n") 

            sim_data = pd.DataFrame(np.zeros((sim_window, self.num_ticker)), columns = [self.ticker_list])

        except Exception as e:
            raise Exception("Error at covariance matrix.")

        # display(sim_data)
        
        if self.Z_list == None: self.Z_list = np.random.normal(0, 1, (self.num_ticker, sim_window))
        # print(sim_data.loc[0, "LONN.SW"])
        try:
            for t in range(sim_window):
                 # returns a scalar if size is not specified
                Z = self.Z_list[:, t]
                for i in range(self.num_ticker): # day need to go first, 
                    if t == 0: prev_price = S_t[i]
                    else: prev_price = sim_data.loc[t - 1, self.ticker_list[i]].item()
                    LZ = np.dot(L, Z.T) # For 1D vector the transpose doesn't matter, but for higher dimension yes
                    print("The 3 matrices are", L, Z, LZ)

                    print(type(prev_price), type(cov_matrix[i][i]), type(LZ[i]))
                    print(interest_rate, cov_matrix[i][i], LZ[i])
                    sim_data.loc[t, self.ticker_list[i]] = prev_price * np.exp(interest_rate * self.dt - 0.5 * cov_matrix[i][i] * self.dt + LZ[i]) # The cov matrix and L need to be computed on the fly

            return sim_data
        
        except Exception as e:
            raise Exception("Error at simulating.")
        pass

    def interest_rate_model(self, parameters):
        # Implementation of the interest rate model (e.g., Vasicek, CIR)
        pass
    
    def heston_model(self, parameters):
        # Implementation of the Heston model for option pricing
        pass

    def monte_carlo_simulation(self, parameters):
        # General Monte Carlo simulation method
        pass
    
    def apply_control_variate(self, control_parameters):
        # Method to apply control variate techniques to reduce simulation variance
        pass

    def calculate_payoff_all_features(self, parameters):
        # Method to calculate payoff with all features like barrier, autocallable, stock conversion
        pass

    def calculate_payoff_barrier(self, parameters):
        # Method to calculate payoff with barrier feature only (for eg.)
        pass

    def plot_simulated_data(self, some_paramenters):
        # Not sure whether this should be subplot to compare with other pricing models or a plot for this current model only
        pass



# Example usage:
# hist_stock_data = pd.DataFrame(...)  # This would be your historical stock data
# pricing_model = PricingModel(hist_stock_data)
# Call methods as needed:
# pricing_model.multidimensional_gbm(some_gbm_parameters)
# pricing_model.apply_control_variate(control_variate_parameters)
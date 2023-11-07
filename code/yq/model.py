import pandas as pd
import typing
import constants as cs
import numpy as np

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

    # Implementation of the multidimensional Geometric Brownian Motion model
    def multidimensional_gbm(self, sim_start_date: pd.Timestamp, hist_window: int, sim_window: int) -> pd.DataFrame: 
        interest_rate = 1.750/100 

        last_avai_price_date = sim_start_date - pd.Timedelta(days = 1)
        S_t = [self.data.loc[last_avai_price_date, ticker] for ticker in self.ticker_list] # Stock price of the 0th day of simulation
        
        hist_data = self.data[self.data.index < sim_start_date].tail(hist_window)

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
        cov_matrix = np.cov(np.array(log_returns_list))
        print(f"Covariance matrix is:\n {cov_matrix}\n")
        print(f"The shape is {np.shape(cov_matrix)}\n")

        print(f"Correlation between the two var is {cov_matrix[0][1] / (cov_matrix[0][0] * cov_matrix[1][1]) ** 0.5}") # Correct

        L = np.linalg.cholesky(cov_matrix)
        print(f"The matrix after Cholesky decomposition is:\n {L}\n")

        print(f"The multiplication of L and L transpose is:\n {np.dot(L, L.T)}\n") 

        sim_data = pd.DataFrame(np.zeros((sim_window, self.num_ticker)), columns = [self.ticker_list])

        # display(sim_data)
        # TODO: N number of simulations
        
        # print(sim_data.loc[0, "LONN.SW"])
        for t in range(sim_window): # TODO: change to num of days to sim (date range or sth)
            Z = np.random.normal(0, 1, self.num_ticker) # returns a scalar if size is not specified
            for i in range(self.num_ticker): # day need to go first, 
                if t == 0: prev_price = S_t[i]
                else: prev_price = sim_data.loc[t - 1, self.ticker_list[i]].item()
                LZ = np.dot(L, Z)

                print(type(prev_price), type(cov_matrix[i][i]), type(LZ[i]))
                print(interest_rate, cov_matrix[i][i], LZ[i])
                sim_data.loc[t, self.ticker_list[i]] = prev_price * np.exp(interest_rate * self.dt - 0.5 * cov_matrix[i][i] * self.dt + LZ[i]) # The cov matrix and L need to be computed on the fly

        return sim_data
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

    def remove_SIX_holidays(self, data: pd.DataFrame) -> pd.DataFrame:
        # Ensure that the dates you are trying to drop exist in the index
        dates_to_drop = [date for date in cs.SIX_HOLIDAY_DATES if date in data.index]
        print(dates_to_drop)

        # Drop the dates
        dropped_data = data.drop(dates_to_drop)
        print(dropped_data)
        return dropped_data

# Example usage:
# hist_stock_data = pd.DataFrame(...)  # This would be your historical stock data
# pricing_model = PricingModel(hist_stock_data)
# Call methods as needed:
# pricing_model.multidimensional_gbm(some_gbm_parameters)
# pricing_model.apply_control_variate(control_variate_parameters)
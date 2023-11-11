import pandas as pd
import typing
import numpy as np
import time
from yq.scripts import heston_func as hf
from yq.utils import option, calendar
from sc import constants as cs

class PricingModel:
    def __init__(self, params: typing.Dict):
        self.data = params.get('data') # Stock price data for underlying asset (can include max data, in the training function can customise what dates to use)
        self.calendar = calendar.SIXTradingCalendar()
        self.ticker_list = params.get('ticker_list')
        self.time_steps_per_year = 252
        self.dt = 1/self.time_steps_per_year
        self.num_ticker = len(self.ticker_list) # Number of stocks
        self.prod_date = params.get('prod_date')
        self.S_0_vector = None
        self.sim_data = None # To store simulated paths of the underlying assets
        self.payoff_df = None # To store payoff calculations based on simulated data
        self.Z_list = None
        self.Z_list_heston = None
        self.params_list_heston = None
        self.L_lower = None
        
    
    # Implementation of the multi-asset Geometric Brownian Motion model
    def multi_asset_gbm(self, sim_start_date: pd.Timestamp, hist_window: int, 
                        sim_window: int, h_adjustment: typing.List) -> pd.DataFrame: 
        """
        Perform a simulation of multi-asset Geometric Brownian Motion (GBM) with
        a h_adjustment to the initial stock price for all assets.

        This function simulates asset paths over a specified window without 
        considering the actual dates. After the simulation, the business days 
        must be concatenated with the simulated data.

        Parameters:
        - sim_start_date (pd.Timestamp): The first day of simulation, which is 
        one business day after the product estimation date.
        - hist_window (int): The number of days to refer in yfinance, usually 252 
        which corresponds to the number of trading days in a year.
        - sim_window (int): The number of days to simulate.
        - h_adjustment (List): The adjustment to the stock price St.

        Returns:
        - pd.DataFrame: A DataFrame containing the simulated asset paths. Each 
        column represents an asset, and each row represents a simulated value 
        on a given day.

        Raises:
        - ValueError: If the provided `sim_start_date` is not a business day or 
        other specific conditions are not met (if applicable).

        Example:
        >>> sim_data = instance.multi_asset_gbm(pd.Timestamp('2023-01-01'), 252, 180)
        >>> print(sim_data)
        """
        interest_rate = 1.750/100 

        try:
            last_avai_price_date = self.calendar.add_trading_day(sim_start_date, -1)
            S_0_vector = [self.data.loc[last_avai_price_date, self.ticker_list[i]] + h_adjustment[i]
                    for i in range(self.num_ticker)] # Stock price of the 0th day of simulation            
            hist_data = self.data[self.data.index < sim_start_date].tail(hist_window)
            print(f"S_0_vector: {S_0_vector}")
        except Exception as e:
            raise Exception("Error at wrangling historical data.")

        try:
            log_returns_list = []
            for ticker in self.ticker_list:
                # print(data[ticker], data[ticker].shift(1))
                log_returns = np.log(hist_data[ticker] / hist_data[ticker].shift(1)) # np.log is natural log, (P_i/P_i-1)
                log_returns.dropna(inplace = True) # A series
                log_returns_list.append(log_returns)
                # print(type(log_returns))

            # print(f"np.array {np.array(log_returns_list)}")
        
        except Exception as e:
            raise Exception("Error at generating log return.")
        
        try:
            cov_matrix = np.cov(np.array(log_returns_list))
            # print(f"Covariance matrix is:\n {cov_matrix}\n")
            # print(f"The shape is {np.shape(cov_matrix)}\n")

            # print(f"Correlation between the two var is {cov_matrix[0][1] / (cov_matrix[0][0] * cov_matrix[1][1]) ** 0.5}") # Correct

            L = np.linalg.cholesky(cov_matrix)
            # print(f"The matrix after Cholesky decomposition is:\n {L}\n")

            # print(f"The multiplication of L and L transpose is:\n {np.dot(L, L.T)}\n") 

            sim_data = pd.DataFrame(np.zeros((sim_window, self.num_ticker)), columns = [self.ticker_list])

        except Exception as e:
            raise Exception("Error at covariance matrix.")

        # print(sim_data)
        # If run h_adjustment != 0 right after one simulation for one price path
        if h_adjustment == [0, 0]:
            self.Z_list = np.random.normal(0, 1, (self.num_ticker, sim_window))

        # print(sim_data.loc[0, "LONN.SW"])
        print(f"S_0_vector: {S_0_vector}")
        try:
            S_t_vector = S_0_vector # Needs to be updated every time step
            for t in range(sim_window):
                Z = self.Z_list[:, t]
                # print(f"Z matrix is \n{Z}\n")
                for i in range(self.num_ticker):
                    LZ = np.dot(L, Z.T) # For 1D vector the transpose doesn't matter, but for higher dimension yes
                    # print("The 3 matrices are", L, Z, LZ)
                    S_t_vector[i] = S_t_vector[i] * np.exp(interest_rate * self.dt - 0.5 * cov_matrix[i][i] * self.dt + LZ[i]) # The cov matrix and L need to be computed on the fly
                    sim_data.loc[t, self.ticker_list[i]] = S_t_vector[i]
        
        except Exception as e:
            raise Exception("Error at simulating.")
        
        dates = self.calendar.create_six_trading_dates(sim_start_date, cs.FINAL_FIXING_DATE)
        # print(f"The length of sim_data and dates is {len(sim_data)} and {len(dates)}\n")
        if (len(sim_data) == len(dates)):
            sim_data.index = dates.index
            sim_data.columns = self.ticker_list
            # print(sim_data)
            return sim_data
        else:  
            raise Exception("Length of sim_data and dates are different.")

    def interest_rate_model(self, parameters):
        # Implementation of the interest rate model (e.g., Vasicek, CIR)
        pass
    
    def multi_asset_heston_model(self, sim_start_date: pd.Timestamp, hist_window: int, 
                        sim_window: int, h_adjustment: typing.List) -> pd.DataFrame:
        interest_rate = 1.750/100 # TODO: Change heston interest rate to refer to the table

        # Do calculations on r, volatility, rho etc.?
    
        # Get the correlation between the log returns of the Si
        if self.S_0_vector is None:
            try:
                last_avai_price_date = self.calendar.add_trading_day(sim_start_date, -1)
                self.S_0_vector = [self.data.loc[last_avai_price_date, self.ticker_list[i]] + h_adjustment[i]
                        for i in range(self.num_ticker)] # Stock price of the 0th day of simulation            
                hist_data = self.data[self.data.index < sim_start_date].tail(hist_window)
                print(f"S_0_vector is {self.S_0_vector}\n")
            except Exception as e:
                raise Exception("Error at wrangling historical data.")

        if (self.params_list_heston is None):
            try:
                # TODO: Change to calibrate every day if have time, hardcoded the calibration S_t
                    # Read options data 
                    lonn_call = option.read_options_data("lonn_call.csv")
                    print(f"Lonza options:\n{lonn_call}")

                    sika_call = option.read_options_data("sika_call.csv")
                    print(f"Sika options:\n{sika_call}")

                    # Calibrate 2 sets of parameters for 2 individual assets
                    start_time = time.time()

                    lonn_result = hf.calibrate_heston(self.data.loc['2023-11-07']['LONN.SW'], lonn_call)
                    sika_result = hf.calibrate_heston(self.data.loc['2023-11-07']['SIKA.SW'], sika_call)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    min, sec = divmod(elapsed_time, 60)
                    print(f"The elapsed time for 2 calibration is {int(min)} minutes, {int(sec)} seconds")
            
                    self.params_list_heston = np.zeros((self.num_ticker, 5))
                    params_order = ['kappa', 'theta', 'volvol', 'rho', 'sigma']
                    # print(f"Enumerate order is: {enumerate(params_order)}\n")
                    for i, param in enumerate(params_order):
                        self.params_list_heston[0, i] = lonn_result.params[param].value  # For lonn_result
                        self.params_list_heston[1, i] = sika_result.params[param].value  # For sika_result

                    np.set_printoptions(suppress=True, precision=4)  # 'precision' controls the number of decimal places
                    print(f"The parameters list for Heston is:\n{self.params_list_heston}")
            except Exception as e:
                raise Exception("Error at calibrating Hestonmodel parameters.")

            # Calculate the rho_S1S2
            try:
                log_returns_list = []
                for ticker in self.ticker_list:
                    # print(data[ticker], data[ticker].shift(1))
                    log_returns = np.log(hist_data[ticker] / hist_data[ticker].shift(1)) # np.log is natural log, (P_i/P_i-1)
                    log_returns.dropna(inplace = True) # A series
                    log_returns_list.append(log_returns)
                    # print(type(log_returns))
                corr_matrix = np.corrcoef(log_returns_list)
                rho_S1S2 = corr_matrix[0][1]
                # print(f"rho_S1S2 is {rho_S1S2}\n")
                # print(f"Correlation between log returns of each asset:\n{corr_matrix}\n")

            except Exception as e:
                raise Exception("Error at calculating the correlation.")
        
            # Calculate the Cholesky matrix
            try:
                rho_S1V1, rho_S2V2 = self.params_list_heston[0][3], self.params_list_heston[1][3]
                simplified_corr = np.array([
                    [1., rho_S1V1, rho_S1S2, 0.],
                    [rho_S1V1, 1., 0., 0.],
                    [rho_S1S2, 0., 1., rho_S2V2],
                    [0., 0., rho_S2V2, 1.]
                ])

                # print(f"Simplified correlation matrix is:\n{simplified_corr}\n")

                self.L_lower = np.linalg.cholesky(simplified_corr)

                # print(f"The lower triangular matrix by Cholesky decomposition is:\n {L_lower}\n")
            
            except Exception as e:
                raise Exception("Error at calculating Cholesky.")
        
        # Generate random variable, assume the original path must be simulated before
        # simulating other +h, -h paths for each asset
        if h_adjustment == [0, 0]:
            self.Z_list_heston = np.random.normal(0, 1, (self.num_ticker * 2, sim_window))
        print(f"Z_list_heston is:\n {self.Z_list_heston}\n")
        # Perform heston for each time step, each asset (diff set of params)
        
        try:
            sim_data = pd.DataFrame(np.zeros((sim_window, self.num_ticker)), columns = [self.ticker_list])
            S_t_vector = self.S_0_vector # Price vector t (to be updated after every step)
            V_t_vector = self.params_list_heston[:, 4] # Initial V_0 (to be updated after every step also)
            for t in range(sim_window):
                Z = self.Z_list_heston[:, t]
                for i in range(self.num_ticker):
                    S_t = S_t_vector[i]
                    kappa = self.params_list_heston[i][0]
                    theta = self.params_list_heston[i][1]
                    xi = self.params_list_heston[i][2]
                    V_t = V_t_vector[i]
                    
                    LZ = np.dot(self.L_lower, Z.T) # For 1D vector the transpose doesn't matter, but for higher dimension yes
                    # print(f"The 3 matrices are: \n", L_lower, Z.T, LZ)

                    S_t_vector[i] = S_t * np.exp((interest_rate - 0.5 * V_t) * self.dt + np.sqrt(V_t) * np.sqrt(self.dt) * LZ[2 * i])
                    V_t = V_t + kappa * (theta - V_t) * self.dt + xi * V_t * np.sqrt(self.dt) * LZ[2 * i + 1]
                    
                    if (i == 0): # LONN.SW
                        print(f"Ratio is: {S_t_vector[i] / S_t}")
                    print(f"The V_t value for {t}th iteration asset {i} is: {V_t}\n")
                    if (V_t < 0): print ("V_t SMALLER THAN 0")
                    V_t_vector[i] = max(V_t, 0) # Truncated V_t
                    sim_data.loc[t, self.ticker_list[i]] = S_t_vector[i] 
        
        except Exception as e:
            raise Exception("Error at simulating.")
        dates = self.calendar.create_six_trading_dates(sim_start_date, cs.FINAL_FIXING_DATE)
        print(f"The length of sim_data and dates is {len(sim_data)} and {len(dates)}\n")
        if (len(sim_data) == len(dates)):
            sim_data.index = dates.index
            sim_data.columns = self.ticker_list
            # print(sim_data)
            return sim_data
        
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

if __name__ == "__main__":
    pass






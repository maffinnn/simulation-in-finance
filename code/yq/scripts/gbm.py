import pandas as pd
import numpy as np
import logging

from yq.scripts.heston import PricingModel
from yq.utils.time import timeit
from yq.scripts import simulation as sm
from sc import constants as cs

logger_yq = logging.getLogger('yq')

class MultiGBM(PricingModel):
    """
    A class representing a multi-asset Geometric Brownian Motion (GBM) model for pricing.

    This class extends the PricingModel class and is used to simulate multiple paths of asset prices using
    the GBM model.

    Attributes:
        h_array (np.array): An array of initial price shifts for sensitivity analysis.
        hist_data (pd.DataFrame): Historical data of asset prices.
        Z_list (np.array): A list of random variables for simulation.
        L_lower (np.array): Lower triangular matrix from Cholesky decomposition of the covariance matrix.
        cov_matrix (np.array): Covariance matrix of log returns.
    """
    def __init__(self, params):
        """
        Initialize the MultiGBM object with given parameters.

        Parameters:
        params (dict): A dictionary containing model parameters.
        """
        super().__init__(params)
        self.h_array = np.array(params.get('h_array')) # All sublists must have the same length
        self.hist_data = None
        self.Z_list = None
        self.L_lower = None
        self.cov_matrix = None

    @timeit
    def sim_n_path(self, n_sim):
        """
        This function simulates n paths of both asset prices.

        Parameters:
        n_sim (int): The number of simulation paths to generate.

        This method simulates multiple paths of asset prices, each path using different initial price shifts (from h_array) for sensitivity analysis. The simulated data is stored and optionally plotted.
        """
        self.hist_data = self.data[self.data.index < self.sim_start_date].sort_index().tail(self.hist_window)
        logger_yq.info(f"The historical data is\n {self.hist_data.head()}")
        self.calc_L_lower()

        for sim in range(n_sim):
            # Diff h need to use the same Z, but diff sim use diff Z
            self.Z_list = np.random.normal(0, 1, (self.num_ticker, self.sim_window))
            sim_data_comb = pd.DataFrame()
            for pair in range(len(self.h_array[0])):
                sim_data = self.sim_path(h_vector=self.h_array[:, pair])
                logger_yq.info(f"The simulated data for {sim}th iteration is:\n {sim_data.head()}")

                sim_data_comb = pd.concat([sim_data_comb, sim_data], axis=1)
            
            # Format the df
            dates = self.calendar.create_six_trading_dates(self.sim_start_date, cs.FINAL_FIXING_DATE)
            if (len(sim_data) == len(dates)):
                sim_data_comb.index = dates.index
            else:
                logger_yq.warning(f"The length of sim_data and dates is different: {len(sim_data)} and {len(dates)}\n")
            logger_yq.info(f"1 sim, diff h, sim_data_comb:\n{sim_data_comb.head()}")
            # Save the every path into folder
            sm.store_sim_data(uid=f"{self.start_time_acc.strftime('%Y%m%d_%H%M%S')}_{self.hist_window}",
                           model_name=self.model_name,
                           sim_data=sim_data_comb,
                           product_est_date=self.prod_date,
                           sim=sim)
            # Plot the graph of every sim
            if (self.plot):
                self.plot_sim_path(plot_hist=True, 
                                   sim_data_comb=sim_data_comb,
                                   sim=sim,
                                   uid=f"{self.start_time_acc.strftime('%Y%m%d_%H%M%S')}_{self.hist_window}")

    @timeit
    def calc_L_lower(self):
        """
        This function calculates the lower triangular matrix from the Cholesky decomposition of the covariance matrix.

        It computes the log returns of asset prices, forms a covariance matrix, and then performs Cholesky decomposition to obtain the lower triangular matrix L_lower.
        """
        log_returns_list = []
        for ticker in self.ticker_list:
            log_returns = np.log(self.hist_data[ticker] / self.hist_data[ticker].shift(1)) # np.log is natural log, (P_i/P_i-1)
            log_returns.dropna(inplace = True) # A series
            log_returns_list.append(log_returns)
        logger_yq.info(f"The log returns list for all assets is: {log_returns_list}")

        self.cov_matrix = np.cov(np.array(log_returns_list))
        self.L_lower = np.linalg.cholesky(self.cov_matrix)
        logger_yq.info(f"Lower triangular matrix L after Cholesky decomposition is:\n{self.L_lower}\n")
        
    def sim_path(self, h_vector: np.array):
        """
        This function simulates a single path of asset prices with a given set of initial price shifts.

        Parameters:
        h_vector (np.array): An array of initial price shifts for sensitivity analysis.

        Returns:
        pd.DataFrame: A DataFrame containing the simulated asset prices for the path.

        This method simulates asset prices for a single path using the provided initial price shifts and the GBM model.
        """
        sim_data = pd.DataFrame(np.zeros((self.sim_window, self.num_ticker)), columns=[self.ticker_list])
        adj_S_0 = self.S_0_vector + h_vector
        logger_yq.info(f"The adjusted S_0 is {adj_S_0}")
        S_t_vector = adj_S_0.copy() # Copy the adjusted S_0 vector
        
        for t in range(self.sim_window):
            Z = self.Z_list[:, t]
            LZ = np.dot(self.L_lower, Z.T) # For 1D vector the transpose doesn't matter, but for higher dimension yes
            # logger_yq.info(f"The 3 matrices L_lower, Z.T, LZ are:\n{self.L_lower}\n{Z.T}\n{LZ}", )

            for i in range(self.num_ticker):
                S_t_vector[i] = S_t_vector[i] * np.exp(self.interest_rate * self.dt - 
                                                       0.5 * self.cov_matrix[i][i] * self.dt + LZ[i]) 
                sim_data.loc[t, self.ticker_list[i]] = S_t_vector[i]
        
        # col_names = [f"{asset}_{h_vector[i]}" for i, asset in enumerate(self.ticker_list)]
        col_names = [f"{asset}" for i, asset in enumerate(self.ticker_list)]
        logger_yq.info(f"The new column names are {col_names}")
        sim_data.columns = col_names
        return sim_data
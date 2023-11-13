import pandas as pd
import typing
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from yq.scripts.heston import PricingModel
from yq.utils import io
from yq.utils.time import timeit
from yq.scripts import heston_func as hf
from yq.scripts import simulation as sm
from yq.utils import option, calendar, log, path as yq_path
from sc import constants as cs
from sc import payoff as po

logger_yq = logging.getLogger('yq')

class MultiGBM(PricingModel):
    def __init__(self, params):
        super().__init__(params)
        self.h_array = np.array(params.get('h_array')) # All sublists must have the same length
        self.hist_data = None
        self.Z_list = None
        self.L_lower = None

    @timeit
    def sim_n_path(self, n_sim):
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
        log_returns_list = []
        for ticker in self.ticker_list:
            log_returns = np.log(self.hist_data[ticker] / self.hist_data[ticker].shift(1)) # np.log is natural log, (P_i/P_i-1)
            log_returns.dropna(inplace = True) # A series
            log_returns_list.append(log_returns)
        logger_yq.info(f"The log returns list for all assets is: {log_returns_list}")

        cov_matrix = np.cov(np.array(log_returns_list))
        self.L_lower = np.linalg.cholesky(cov_matrix)
        logger_yq.info(f"Lower triangular matrix L after Cholesky decomposition is:\n{self.L_lower}\n")
        
    def sim_path(self, h_vector: np.array):
        sim_data = pd.DataFrame(np.zeros((self.sim_window, self.num_ticker)), columns=[self.ticker_list])
        adj_S_0 = self.S_0_vector + h_vector
        logger_yq.info(f"The adjusted S_0 is {adj_S_0}")
        S_t_vector = adj_S_0.copy() # Copy the adjusted S_0 vector
        
        for t in range(self.sim_window):
            Z = self.Z_list[:, t]
            LZ = np.dot(self.L_lower, Z.T) # For 1D vector the transpose doesn't matter, but for higher dimension yes
            # logger_yq.info(f"The 3 matrices L_lower, Z.T, LZ are:\n{self.L_lower}\n{Z.T}\n{LZ}", )

            for i in range(self.num_ticker):
                S_t_vector[i] = S_t_vector[i] * np.exp(self.interest_rate * self.dt - 0.5 * cov_matrix[i][i] * self.dt + LZ[i]) 
                sim_data.loc[t, self.ticker_list[i]] = S_t_vector[i]
        
        col_names = [f"{asset}_{h_vector[i]}" for i, asset in enumerate(self.ticker_list)]
        logger_yq.info(f"The new column names are {col_names}")
        sim_data.columns = col_names
        return sim_data
    
    
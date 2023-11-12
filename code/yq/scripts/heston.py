import pandas as pd
import typing
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from yq.scripts import heston_func as hf
from yq.scripts import simulation as sm
from yq.utils import option, calendar, log, path as yq_path
from sc import constants as cs
from sc import payoff as po
from sy.interest_rate import populate_bond_table
import datetime

logger_yq = logging.getLogger('yq')

# This is a pricing model to price a specific item on one date. Includes multiple
# simulations with different models, calibrations (if needed), 
class PricingModel:
    def __init__ (self, params):
        self.data = po.get_historical_assets_all()
        self.ticker_list = cs.ASSET_NAMES
        self.calendar = calendar.SIXTradingCalendar()
        self.prod_date = params.get('prod_date')
        self.hist_window = params.get('hist_window')
        self.start_time_acc = params.get('start_time_acc')
        self.time_steps_per_year = 252
        self.dt = 1/self.time_steps_per_year
        self.interest_rate = cs.INTEREST_RATE
        self.sim_end_date = cs.FINAL_FIXING_DATE

        # Derived attributes
        self.num_ticker = len(self.ticker_list) # Number of stocks
        self.sim_start_date = self.calendar.add_trading_day(self.prod_date, 1)
        self.sim_window = self.calendar.calculate_business_days(self.sim_start_date, cs.FINAL_FIXING_DATE)
        self.S_0_vector = np.array([self.data.loc[self.prod_date, asset]
                    for asset in self.ticker_list])
        self.sim_data = None # To store one simulated path of all the underlying assets
        self.payout = None # To store payoff calculations based on simulated data


class multi_heston(PricingModel):
    """
    A class for simulating asset prices using the Multi-Factor Heston model.

    Attributes:
        h_array (np.array): Array of h-values for the Heston model.
        hist_data (DataFrame): Historical data used for calibration.
        Z_list (np.array): Random values for simulation.
        params_list_heston (np.array): Parameters for the Heston model.
        L_lower (np.array): Lower triangular matrix from Cholesky decomposition.

    Methods:
        sim_n_path(n_sim): Simulate 'n_sim' paths of asset prices.
        sim_path(S_0_vector): Simulate a single path of asset prices.
        calibrate(prod_date): Calibrate the model parameters.
        calc_L_lower(): Calculate the lower triangular matrix L.
    """
    def __init__(self, params):
        super().__init__(params)
        self.h_array = np.array(params.get('h_array')) # All sublists must have the same length
        self.hist_data = None
        self.Z_list = None
        self.params_list_heston = None
        self.L_lower = None

    def sim_n_path(self, n_sim):
        start_time = time.time()
        self.hist_data = self.data[self.data.index < self.sim_start_date].sort_index().tail(self.hist_window)
        logger_yq.info(f"The historical data is\n {self.hist_data}")
        self.calibrate(self.prod_date)
        self.calc_L_lower()
        
        for sim in range(n_sim):
            # Diff h need to use the same Z, but diff sim use diff Z
            self.Z_list_heston = np.random.normal(0, 1, (self.num_ticker * 2, self.sim_window))
            sim_data_comb = pd.DataFrame()
            for pair in range(len(self.h_array[0])):
                sim_data = self.sim_path(h_vector=self.h_array[:, pair])
                logger_yq.info(f"The simulated data for {sim}th iteration is:\n {sim_data}")

                sim_data_comb = pd.concat([sim_data_comb, sim_data], axis=1)
            
            # Format the df
            dates = self.calendar.create_six_trading_dates(self.sim_start_date, cs.FINAL_FIXING_DATE)
            if (len(sim_data) == len(dates)):
                sim_data_comb.index = dates.index
            else:
                logger_yq.warning(f"The length of sim_data and dates is different: {len(sim_data)} and {len(dates)}\n")
            logger_yq.info(f"1 sim, diff h, sim_data_comb:\n{sim_data_comb}")
            # Save the every path into folder
            sm.store_sim_data(start_time_acc=self.start_time_acc,
                           model_name='heston',
                           sim_data=sim_data_comb,
                           product_est_date=self.prod_date,
                           sim=sim)
            # Plot the graph of every sim
            sim_data_comb.plot()
            stor_dir = yq_path.get_plots_path(Path(__file__).parent).joinpath('sim_data_comb_heston', self.start_time_acc.strftime('%Y%m%d_%H%M%S_%f'), self.prod_date.strftime('%Y_%m_%d'))
            stor_dir.mkdir(parents=True, exist_ok=True)
            file_path = stor_dir.joinpath(f'{sim}.png')
            plt.savefig(file_path)
            plt.close()

        end_time = time.time()
        elapsed_time = end_time - start_time
        min, sec = divmod(elapsed_time, 60)
        logger_yq.info(f"The elapsed time for {n_sim}th sim is {int(min)} minutes, {int(sec)} seconds")
    
    def sim_path(self, h_vector: np.array):
        sim_data = pd.DataFrame(np.zeros((self.sim_window, self.num_ticker)), columns=[self.ticker_list])
        adj_S_0 = self.S_0_vector + h_vector
        logger_yq.info(f"The adjusted S_0 is {adj_S_0}")
        S_t_vector = adj_S_0.copy() # Copy the adjusted S_0 vector
        V_t_vector = self.params_list_heston[:, 4].copy() # Initial V_0 (to be updated after every step also)
        for t in range(self.sim_window):
            Z = self.Z_list_heston[:, t]
            LZ = np.dot(self.L_lower, Z.T) # For 1D vector the transpose doesn't matter, but for higher dimension yes
            # logger_yq.info(f"The 3 matrices L_lower, Z.T, LZ are:\n{self.L_lower}\n{Z.T}\n{LZ}", )

            for i in range(self.num_ticker):
                S_t = S_t_vector[i]
                kappa, theta, xi = self.params_list_heston[i][:3]
                V_t = V_t_vector[i]
                

                # if (i == 0): # LONN.SW
                #     logger_yq.info("LZ values are %s, %s", LZ[2 * i], LZ[2 * i + 1])
                # logger_yq.info("The values for %sth iteration asset %s are %s, %s, %s, %s, %s", t, i, S_t, kappa, theta, xi, V_t)
                
                S_t_vector[i] = S_t * np.exp((self.interest_rate - 0.5 * V_t) * self.dt + np.sqrt(V_t) * np.sqrt(self.dt) * LZ[2 * i])
                V_t = V_t + kappa * (theta - V_t) * self.dt + xi * V_t * np.sqrt(self.dt) * LZ[2 * i + 1]
                
                # if (i == 0): # LONN.SW
                #     # print(f"Ratio is: {S_t_vector[i] / S_t}")
                #     logger_yq.info("New S_t, new V_t and S_t+1/S_t is %s, %s, %s", S_t_vector[i], V_t_vector[i], S_t_vector[i] / S_t)

                # logger_yq.info("The V_t value for %sth iteration asset %s is: %s", t, i, V_t)

                if (V_t < 0): logger_yq.warning("V_t SMALLER THAN 0")
                V_t_vector[i] = max(V_t, 0) # Truncated V_t
                sim_data.loc[t, self.ticker_list[i]] = S_t_vector[i]
        col_names = [f"{asset}_{h_vector[i]}" for i, asset in enumerate(self.ticker_list)]
        logger_yq.info(f"The new column names are {col_names}")
        sim_data.columns = col_names
        return sim_data

    def calibrate(self, prod_date: pd.Timestamp):
        lonn_call = option.read_clean_options_data(options_dir='options-cleaned', curr_date=prod_date, file_name="lonn_call.csv")
        sika_call = option.read_clean_options_data(options_dir='options-cleaned', curr_date=prod_date, file_name="sika_call.csv")

        lonn_call = lonn_call[['maturity', 'strike', 'price', 'rate']]
        sika_call = sika_call[['maturity', 'strike', 'price', 'rate']]

        logger_yq.info(f"LONZA and SIKA df are \n{lonn_call}\n{sika_call}")

        # Calibrate 2 sets of parameters for 2 individual assets
        start_time = time.time()
        lonn_S_0 = self.data.loc[prod_date][cs.ASSET_NAMES[0]]
        sika_S_0 = self.data.loc[prod_date][cs.ASSET_NAMES[1]]

        try: 
            logger_yq.info(f"Calibrating LONZA.")
            lonn_result = hf.calibrate_heston(lonn_S_0, lonn_call)
            logger_yq.info(f"Calibrating SIKA.")
            sika_result = hf.calibrate_heston(sika_S_0, sika_call)
            logger_yq.info(f"The S_0 for LONZA and SIKA are {lonn_S_0} and {sika_S_0}")
            logger_yq.info(f"Calibration results for LONZA and SIKA are \n{lonn_result}\n {sika_result}")
        except Exception as e:
            logger_yq.error(f"Error during calibration on {self.prod_date}: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        min, sec = divmod(elapsed_time, 60)
        logger_yq.info(f"The elapsed time for 2 calibration is {int(min)} minutes, {int(sec)} seconds")

        self.params_list_heston = np.zeros((self.num_ticker, 5))
        params_order = ['kappa', 'theta', 'volvol', 'rho', 'sigma']
        # print(f"Enumerate order is: {enumerate(params_order)}\n")
        for i, param in enumerate(params_order):
            self.params_list_heston[0, i] = lonn_result.params[param].value  # For lonn_result
            self.params_list_heston[1, i] = sika_result.params[param].value  # For sika_result

        logger_yq.info("The calibrated params_list_Heston is:\n %s", self.params_list_heston)

    def calc_L_lower(self):
        log_returns_list = []
        for ticker in self.ticker_list:
            log_returns = np.log(self.hist_data[ticker] / self.hist_data[ticker].shift(1)) # np.log is natural log, (P_i/P_i-1)
            log_returns.dropna(inplace = True) # A series
            log_returns_list.append(log_returns)
        logger_yq.info(f"The log returns list for all assets is: {log_returns_list}")

        corr_matrix = np.corrcoef(log_returns_list)
        rho_S1S2 = corr_matrix[0][1]
        logger_yq.info(f"rho_S1S2 is {rho_S1S2}\n")
        logger_yq.info(f"Correlation between log returns of each asset:\n{corr_matrix}\n")

        rho_S1V1, rho_S2V2 = self.params_list_heston[0][3], self.params_list_heston[1][3]
        simplified_corr = np.array([
            [1., rho_S1V1, rho_S1S2, 0.],
            [rho_S1V1, 1., 0., 0.],
            [rho_S1S2, 0., 1., rho_S2V2],
            [0., 0., rho_S2V2, 1.]
        ])

        logger_yq.info(f"Simplified correlation matrix is:\n{simplified_corr}\n")

        self.L_lower = np.linalg.cholesky(simplified_corr)
        logger_yq.info(f"Lower triangular matrix L after Cholesky decomposition is:\n{self.L_lower}\n")
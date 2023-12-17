import pandas as pd
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from yq.utils import io
from yq.utils.time import timeit
from yq.scripts import heston_func as hf
from yq.scripts import simulation as sm
from yq.utils import option, calendar, path as yq_path
from sc import constants as cs
from sc import payoff as po
from sy.interest_rate import populate_bond_table

logger_yq = logging.getLogger('yq')
 
class PricingModel:
    """
    This class represents a pricing model to price a product on one date. 
    It includes multiple simulations with different models and calibrations if needed.

    Attributes:
        model_name (str): Name of the pricing model.
        data (pd.DataFrame): Historical data of asset prices.
        ticker_list (list): List of asset names.
        calendar (SIXTradingCalendar): Trading calendar object.
        prod_date (pd.Timestamp): The production date for pricing.
        hist_window (int): Historical window for data.
        start_time_acc (datetime): Accurate start time.
        plot (bool): Flag to determine if plots should be generated.
        time_steps_per_year (int): Number of time steps per year.
        dt (float): Delta time for simulation steps.
        interest_rate (float): Interest rate for pricing.
        sim_end_date (pd.Timestamp): End date for simulation.
        num_ticker (int): Number of tickers (assets).
        sim_start_date (pd.Timestamp): Start date for simulation.
        sim_window (int): Simulation window length.
        S_0_vector (np.array): Initial price vector for assets.
    """
    def __init__ (self, params):
        self.model_name = params.get('model_name')
        self.data = po.get_historical_assets_all()
        self.ticker_list = cs.ASSET_NAMES
        self.calendar = calendar.SIXTradingCalendar()
        self.prod_date = params.get('prod_date')
        self.hist_window = params.get('hist_window')
        self.start_time_acc = params.get('start_time_acc')
        self.plot: bool = params.get('plot')
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

    @timeit
    def plot_sim_path(self, plot_hist: bool, sim_data_comb: pd.DataFrame, sim: int, uid: str) -> None:
        # Plot sim paths from prod pricing date (for each h) for each asset
        fig, ax = plt.subplots(figsize=(10,6)) 
        # If use figure then cannot have fig, ax; just use plt in lines below
        if plot_hist: 
            logger_yq.info(f"The hist_data is\n{self.hist_data}")
            hist_df = self.data[(self.data.index >= cs.INITIAL_FIXING_DATE) 
                           & (self.data.index <= cs.FINAL_FIXING_DATE)]
            for asset in cs.ASSET_NAMES:
                ax.plot(hist_df.index, hist_df[asset], alpha=0.5, label=asset)
        data = {}
        for i in range(len(cs.ASSET_NAMES)):
            data[sim_data_comb.columns[i]] = self.S_0_vector[i] # Access the price on the prod date
        prod_date_df = pd.DataFrame(data, index=[self.prod_date])
        smooth_path = pd.concat([prod_date_df, sim_data_comb], axis=0)
        logger_yq.info(f"The smooth path df is:\n{smooth_path}")
        smooth_path.plot(ax=ax, alpha=0.5) # Another way of plotting

        if self.model_name == 'heston':
            title_str = f"Model: {self.model_name}, PPD: {self.prod_date.strftime('%Y-%m-%d')}, hist_wdw: {self.hist_window}, max_sigma: {self.max_sigma}"
        else:
            title_str = f"Model: {self.model_name}, PPD: {self.prod_date.strftime('%Y-%m-%d')}, hist_wdw: {self.hist_window}"
        
        subtitle_str = f"sim_start_date: {self.sim_start_date.strftime('%Y-%m-%d')}, sim_wdw: {self.sim_window}"
        plt.title(f"{title_str}\n{subtitle_str}")
        plt.legend(loc='upper right')

        stor_dir = yq_path.get_plots_path(Path(__file__).parent).joinpath(f'{self.model_name}',
                                                                          uid, 
                                                                          self.prod_date.strftime('%Y_%m_%d'))
        stor_dir.mkdir(parents=True, exist_ok=True)
        file_path = stor_dir.joinpath(f'{sim}.png')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        logger_yq.info(f"Path of 1 sim is plotted")


class MultiHeston(PricingModel):
    """
    This class extends PricingModel for specific use with the Heston model in simulations.

    Attributes:
        h_array (np.array): An array of initial price shifts for sensitivity analysis.
        max_sigma (float): Maximum sigma value for calibration.
        hist_data (pd.DataFrame): Historical data of asset prices.
        Z_list (np.array): A list of random variables for simulation.
        params_list_heston (np.array): Parameters list for the Heston model.
        L_lower (np.array): Lower triangular matrix from Cholesky decomposition.
    """
    def __init__(self, params):
        """
        This function initialises a PricingModel object with given parameters.

        Parameters:
        params (dict): Parameters for the pricing model including model name, production date, historical window, start time for accounting, plot flag, and other relevant parameters.
        """
        super().__init__(params)
        self.h_array = np.array(params.get('h_array')) # All sublists must have the same length
        self.max_sigma = params.get('max_sigma')
        self.hist_data = None
        self.Z_list = None
        self.params_list_heston = None
        self.L_lower = None

    @timeit
    def sim_n_path(self, n_sim):
        """
        This function simulates 'n' paths of asset prices using the Heston model.
        """
        self.hist_data = self.data[self.data.index < self.sim_start_date].sort_index().tail(self.hist_window)
        logger_yq.info(f"The historical data is\n {self.hist_data.head()}")
        self.calibrate(self.prod_date)
        self.calc_L_lower()
        
        for sim in range(n_sim):
            # Diff h need to use the same Z, but diff sim use diff Z
            self.Z_list = np.random.normal(0, 1, (self.num_ticker * 2, self.sim_window))
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
            sm.store_sim_data(uid=f"{self.start_time_acc.strftime('%Y%m%d_%H%M%S')}_{self.hist_window}_{self.max_sigma}",
                           model_name=self.model_name,
                           sim_data=sim_data_comb,
                           product_est_date=self.prod_date,
                           sim=sim)
            # Plot the graph of every sim
            if (self.plot):
                self.plot_sim_path(plot_hist=True, 
                                   sim_data_comb=sim_data_comb,
                                   sim=sim,
                                   uid=f"{self.start_time_acc.strftime('%Y%m%d_%H%M%S')}_{self.hist_window}_{self.max_sigma}")
    
    def sim_path(self, h_vector: np.array):
        """
        This function simulates 'n' paths of asset prices using the Heston model.

        Parameters:
        n_sim (int): The number of simulation paths to generate.
        """
        sim_data = pd.DataFrame(np.zeros((self.sim_window, self.num_ticker)), columns=[self.ticker_list])
        adj_S_0 = self.S_0_vector + h_vector
        logger_yq.info(f"The adjusted S_0 is {adj_S_0}")
        S_t_vector = adj_S_0.copy() # Copy the adjusted S_0 vector
        V_t_vector = self.params_list_heston[:, 4].copy() # Initial V_0 (to be updated after every step also)
        for t in range(self.sim_window):
            Z = self.Z_list[:, t]
            LZ = np.dot(self.L_lower, Z.T) # For 1D vector the transpose doesn't matter, but for higher dimension yes
            # logger_yq.info(f"The 3 matrices L_lower, Z.T, LZ are:\n{self.L_lower}\n{Z.T}\n{LZ}", )

            for i in range(self.num_ticker):
                S_t = S_t_vector[i]
                kappa, theta, xi = self.params_list_heston[i][:3]
                V_t = V_t_vector[i]
                

                # if (i == 0): # LONN.SW
                #     logger_yq.info("LZ values are %s, %s", LZ[2 * i], LZ[2 * i + 1])
                # logger_yq.info("The values for %sth iteration asset %s are %s, %s, %s, %s, %s", t, i, S_t, kappa, theta, xi, V_t)
                # logger_yq.info("The V_t value for %sth iteration asset %s is: %s", t, i, V_t)
                S_t_vector[i] = S_t * np.exp((self.interest_rate - 0.5 * V_t) * 
                                             self.dt + np.sqrt(V_t * self.dt) * LZ[2 * i])
                V_t = V_t + kappa * (theta - V_t) * self.dt + xi * np.sqrt(V_t * self.dt) * LZ[2 * i + 1]
                
                # if (i == 0): # LONN.SW
                #     # print(f"Ratio is: {S_t_vector[i] / S_t}")
                #     logger_yq.info("New S_t, new V_t and S_t+1/S_t is %s, %s, %s", S_t_vector[i], V_t_vector[i], S_t_vector[i] / S_t)

                if (V_t < 0): logger_yq.warning("V_t SMALLER THAN 0")
                V_t_vector[i] = max(V_t, 0) # Truncated V_t
                sim_data.loc[t, self.ticker_list[i]] = S_t_vector[i]
        # col_names = [f"{asset}_{h_vector[i]}" for i, asset in enumerate(self.ticker_list)]
        col_names = [f"{asset}" for i, asset in enumerate(self.ticker_list)]
        logger_yq.info(f"The new column names are {col_names}")
        sim_data.columns = col_names
        return sim_data

    @timeit
    def calibrate(self, prod_date: pd.Timestamp):
        """
        This function simulates a single path of asset prices with a given set of initial price shifts.

        Parameters:
        h_vector (np.array): Array of initial price shifts for the simulation.

        Returns:
        pd.DataFrame: A DataFrame containing the simulated asset prices for the path.
        """
        try:
            self.params_list_heston = io.read_hparams(self.prod_date, self.max_sigma)
            return
        except:
            # raise # Don't raise because I want to continue
        
            lonn_call = option.read_clean_options_data(options_dir='options-cleaned', curr_date=prod_date, file_name="lonn_call.csv")
            sika_call = option.read_clean_options_data(options_dir='options-cleaned', curr_date=prod_date, file_name="sika_call.csv")

            lonn_call = lonn_call[['maturity', 'strike', 'price', 'rate']]
            sika_call = sika_call[['maturity', 'strike', 'price', 'rate']]

            logger_yq.info(f"LONZA and SIKA df are \n{lonn_call.head()}\n{sika_call.head()}")

            # Calibrate 2 sets of parameters for 2 individual assets
            start_time = time.time()
            lonn_S_0 = self.data.loc[prod_date][cs.ASSET_NAMES[0]]
            sika_S_0 = self.data.loc[prod_date][cs.ASSET_NAMES[1]]

            try: 
                logger_yq.info(f"Calibrating LONZA on {self.prod_date}.")
                lonn_result = hf.calibrate_heston(lonn_S_0, self.max_sigma, lonn_call)
                logger_yq.info(f"Calibrating SIKA on {self.prod_date}.")
                sika_result = hf.calibrate_heston(sika_S_0, self.max_sigma, sika_call)
                logger_yq.info(f"The S_0 for LONZA and SIKA are {lonn_S_0} and {sika_S_0}")
                # logger_yq.info(f"Calibration results for LONZA and SIKA are \n{lonn_result}\n {sika_result}")
            except Exception as e:
                logger_yq.error(f"Error during calibration on {self.prod_date.strftime('%Y-%m-%d')}: {e}")
                raise

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

            io.write_hparams(prod_date=self.prod_date, max_sigma=self.max_sigma, hparams_list=self.params_list_heston)
            logger_yq.info("The calibrated params_list_Heston is:\n %s", self.params_list_heston)

    @timeit
    def calc_L_lower(self):
        """
        This function performs calibration for the Heston model based on historical data.

        Parameters:
        prod_date (pd.Timestamp): The production date for which calibration is performed.
        """
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

        try:
            self.L_lower = np.linalg.cholesky(simplified_corr)
        except Exception as e:
            logger_yq.error(f"Error with max_sigma{self.max_sigma}, day: {self.prod_date}")

        logger_yq.info(f"Lower triangular matrix L after Cholesky decomposition is:\n{self.L_lower}\n")
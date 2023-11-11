# # Main

# # multi_gbm
# start_time_acc
# n_sim
# for prod_date in date_range:
    



# # multi_heston
# # Main
# start_time_acc
# n_sim
# for prod_date in date_range:
#     heston = Heston()
#     heston.calibrate(prod_date)
#     sim_data_df
#     heston.simulate(prod_date, start_time_acc, n_sim, h_adjustment)
        
# # multi_heston class
# class HestonSim:
#     params_for_prod_date
# def calibrate:
#     store all the 
# def simulate:
#     - Calculate sim_start_date
#     - Calculate Z list
#     for sim in range(n_sim):
#         - sim_data = gbm.multi_asset_gbm(
#                 sim_start_date=sim_start_date, 
#                 hist_window=252, 
#                 sim_window=trading_calendar.calculate_business_days(sim_start_date, cs.FINAL_FIXING_DATE), 
#                 h_adjustment=[1, 0])

#         sm.store_sim_data(start_time_acc=start_time_acc,
#                             model_name='heston',
#                             sim_data=sim_data,
#                             product_est_date=product_est_date,
#                             sim=sim)
# diff h
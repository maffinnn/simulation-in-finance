# Time log
## Optimisation
- Need to identify the bottleneck. Consider the speed of read/write operations, the 
speed of numerical calculations and storage operations

- Numpy is way more optimised for numerical calculations but pandas can be faster in 
other scenarios too

- Consider storing simple numbers rather than string or datetime objects

- Storing into python objects might be convenient but the cost of runtime shall
be considered

- Spatial locality might be important for time optimisation. In this project the sim_data
is stored quite far from the models (io.py)

- Consider parallelisation in the future

## Multi GBM
- Before improvement
    - 100 sim about 20 sec

- After improvement
    - 100 sim about 11 sec (similar to heston)

## Multi Heston
- Before improvement
    - About 2 minutes per simulation

- After improvement
    - 2 calibration is 8-9 sec

    - Sometimes up to 1 minute

    - n simulations is about 11 sec. (hparams are calculated in advance)

## Grid search
```    
hist_windows = [63]
n_sims = [3]
models = ['gbm','heston']
max_sigmas = [0.5, 1.5, 10]
sim_grid_search_heston(hist_windows=hist_windows,
                n_sims=n_sims,
                models=models,
                max_sigmas=max_sigmas)
```

- 6 combinations 4m 57s


```[178 rows x 2 columns], sim=2, uid='20231114_000048_63_10')
2023-11-14 00:02:09,139 - yq - INFO - Runtime of sim_n_path: 0h 0m 0s 857ms
(<yq.scripts.heston.MultiHeston object at 0x16faacfd0>, n_sim=3)
2023-11-14 00:02:09,139 - yq - INFO - Simulated 3 paths for 67 days.
2023-11-14 00:02:09,139 - yq - INFO - Runtime of sim_price_period: 0h 1m 20s 777ms
(start_date=Timestamp('2023-08-09 00:00:00'), end_date=Timestamp('2023-11-09 00:00:00'), hist_window=63, n_sim=3, plot=True, max_sigma=10, model='heston')
2023-11-14 00:02:09,139 - yq - INFO - Runtime of sim_grid_search_heston: 0h 4m 57s 545ms```
```

- 1 combo, heston

```   
hist_windows = [63]
n_sims = [100]
models = ['heston']
max_sigmas = [1.5]

Runtime of sim_n_path: 0h 0m 11s 524ms
(<yq.scripts.heston.MultiHeston object at 0x166028550>, n_sim=100)
```

- 1 combo, gbm

```    
hist_windows = [63]
n_sims = [100]
models = ['gbm']
max_sigmas = [1.5]
```
## Full run
Start time: 20231114_024525
End time: 20231115_070152
About 28 hours

## Bottleneck analysis of the GBM sim_n_path functions

2023-11-17 00:32:44,452 - yq - INFO - Runtime of sim_path: 0h 0m 0s 17.881000ms
(<yq.scripts.gbm.MultiGBM object at 0x137fab810>, h_vector=array([0, 0]))
2023-11-17 00:32:44,537 - yq - INFO - Runtime of store_sim_data: 0h 0m 0s 0.846000ms

Estimated runtime: total hyperparameters combinations x number of product pricing dates x n_sims

For the case above: 1 x 67 x 1000 x (17.881 + 0.846) / 1000ms/s / 60s/min = 20 min

CPU is going very high around 2023-11-17 00:38 - suspect file read write will be slower (because of indexing issues)

- One of the storage time is high, sim path ranges from 20 to 40ms
2023-11-17 00:40:10,249 - yq - INFO - Runtime of sim_path: 0h 0m 0s 19.573000ms
(<yq.scripts.gbm.MultiGBM object at 0x14fe84c10>, h_vector=array([0, 0]))
2023-11-17 00:40:10,350 - yq - INFO - Runtime of store_sim_data: 0h 0m 0s 12.056000ms

- Printing the signature (function params) is a bad idea because it will print all the dataframes also (even slower)
logger_yq.info(f"Runtime of {func.__name__}: {int(hours)}h {int(minutes)}m {seconds}s {milliseconds:.6f}ms\n({signature})")

2023-11-17 00:54:09,259 - yq - INFO - Runtime of sim_price_period: 0h 13m 59s 440.897000ms
(start_date=Timestamp('2023-08-09 00:00:00'), end_date=Timestamp('2023-11-09 00:00:00'), hist_window=252, n_sim=100, plot=False, max_sigma=0, model='gbm')

- After taking out the constant in the exponential:
    - Highest runtime is 63ms (2 paths)
    - 2023-11-17 01:03:05,361 - yq - INFO - Runtime of sim_path: 0h 0m 0s 63.342000ms
    - 585/670 is under 20 seconds

- 29% of the store_item is taking 1ms and above now (bad thing)
    - 2023-11-17 01:03:28,119 - yq - INFO - Runtime of store_sim_data: 0h 0m 0s 1.306000ms

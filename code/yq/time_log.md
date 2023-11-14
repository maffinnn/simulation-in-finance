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

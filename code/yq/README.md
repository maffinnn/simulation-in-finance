# README
## Introduction
- TODO: 

## Results
- The performance of GBM and Heston is against expectation
    - Expected Heston that utilise stochastic volatility in the path simulatio would perform a lot better but it  underperformed. A lot of effort is put into downloading options data, cleaning options data, calibrating model  parameters, manually adjusting problematic options data (more than half out of 134 (2 x 67 PPD).)

    - Performed further analysis on the mean forecasted prices against the actual share price, and realised that for   SIKA.SE, both of the models were able to capture the trends pretty well, probably due to the less volatile nature.  However, both models were challenged when they needed to predict the steep downward trend for LONN.SE in October,  Heston frequently going against the trend, hence the higer RMSE.

    - Even the n_sim=10 GBM managed to rank number 6 out of all the models during the hyperparameter tuning, which only  needs a mockingly short amount of time to build and simulate compared to other models with large n_sim.

    - The RMSE 26.08 to 528.32, which is quite a big range. Unexpectedly, again  the top 3 of the models are all GBM. The top 5 models have RMSE smaller or equal to 52.74.

## Navigation
- Start reading the codes in yq_script.py in code/
    - Grid search for different models, hyperparameters.

    - Declaration of each class (gbm, heston) for each product pricing date.

    - Performing pre-simulation calculations and generating the entire path for n_sim.

    - Evaluating these results using model_eval (did not include in the classes because
    the data are stored and read separately).

- Directory and file descriptions
    - Cleaning, wrangling of options data files and options data in option.py.

    - Logging of errors in yq/logs.

    - SIX calendars, io operations, logging setup, finding paths, timeit operations are all in utils/.

    - yq/scripts folder consists of mostly core functions of the pricing model like gbm 
    and heston classes.

    - models.py is outdated. This file includes the first version of gbm and heston that
    needed to calibrate every simulation on each product pricing date (heston) and
    calculate the lower triangular matrix.

    - Calibrated heston parameters are stored in hparams folder (organised based on the 
    max_sigma upper bound used in the heston_func.py).

    - The final evaluation of the RMSEs of all the models built during hyperparameter tuning is under the yq_pg.ipynb,  and the evaluation of the mean paths and individual paths with again

### Challenges
- Calibration error debugging
    - Potential pitfalls: taking sqrt(x) or log(x) where x is negative.
    doing x**y where x is negative. Since y is real, there will be a fractional component, and a negative number to a   fractional exponent is not a real number.
    doing x/y where both x and y are 0.

- Code writing
    - It is very time consuming to write codes in IPYNB that requires import since every
    time the kernel needs to be restarted to re-import the updated modules. Ended up
    using `yq_scripts.py`

    - There are countless ways to organise. Choose one based on the priorities and be settled
    with that way.

- Mathematical models
    - Translating from the models' form to codes need time. Consider at least one day to understand the concepts per model,   and more days for those with calibration, exogenous data,
    discretisation methods, multiple model parameters

    - Kudos to Prof Patrick who helped us a lot in understanding the models! [Link to his NTU profile](https://www.ntu.edu.sg/research/faculty-directory/detail/rp00948)

    - Understanding that the math models are not direct implication of final results is crucial.
    However it is important to have some form of estimation or capturing of the trend.

> "All models are wrong, but some are useful." - George E. P. Box

- Mathematical interpretations
    - Consideration of using normalised graphs (each asset's path divided by its initial levels)
        - Might cause problem while calculating covariance matrix, calibrations since we need  real world data.

        - Might be efficient in calculating payouts since barriers, worst performing share etc.
        are considering relative levels.

        - If the difference is very huge, plotting will introduce some bias (cannot eyeball volatility) but provides real-world value interpretations.

    - Consideration of doing h adjustments (for Greeks) by calculating from the simulated data directly (need 3 different simulations (0, 0), (1, 1), (-1, -1) to get 5 combinations by fixing one asset constant and another varying).
        - After experimentation, we realised that both the gbm and heston models generate very 
        similar paths, so we only simulate using the original S_0.

- Options data
    - Took about 2-3 whole day sessions in Bloomberg to download the data. At time-critical situations, consider performing renaming and organising afterwards.
    
    - A lot of options data are broken. Nearly half produces calibration error (out of 67 product pricing days x 2 assets of data sheet).
    
    - After fixing calibration issues there are issues with matrix being not positive
    definite also (rho > 1 or some other issues).
    
    - Example Heston params that cause positive definite errors (Others are rho > 1 but solved by replacing the params on next day).
    ```0,1,2,3,4
    9.999999999946297,0.21012698681534034,-1.755398786329465e-07,0.938343282563145,0.015286348429149565
    2.9317569580728815,0.052301410532130485,-0.07320997319345202,0.7633558338662152,0.19698549018994033```

    - The best way is to write functions such that they could be run multiple times
    clean the raw data replace the cleaned_data.
    
    - Some data needs human intervention so consider storing them in CSV formats rather
    than machine codes but consider runtime constraints.

- Codebase
    - The larger the code base, the harder it is to refactor. 
    
    - Consider writing unit test scripts.

    - sim_data, log files and plots can be huge when n_sim can go up to 1M. Consider adding
    these into .gitignore and sharing these files separately with online storage.

    - Need to understand how python packages work.

- Collaborating with others
    - It is crucial to discuss the format of data needed, parameters needed, how to access
    certain information that is lost after storing the data (for eg. calculating the payoffs
    of each price path (simulated data) will lose the hyperparameters like the max_sigma, model,
    hist_window etc.).

    - It is important to consider the dominos effect of one change to the entire codebase and team.

    - Logging errors and including simple documentation of the inputs are very important for others to debug.

- Analysis
    - Sometimes our assumptions could be wrong: for example feeling that GBM works very bad, but actually it can also capture the payouts quite well (need to compare RMSE for further
    analysis).

    - Plotting graphs is very useful.

## Future improvements
- Consider planning out the final presentation style before even starting the project,
which can save time on refactoring gbm and heston classes multiple times.

- Use logger at different levels .

- Try to capture the steep downward trend in October. Key findings: On October 17 the Capital  Markets Day was held, and before that there were warnings about worse outlooks  related to the production of Moderna's decision to reduce substance production for the COVID  vaccine.

- Perform bottleneck analysis in the simulate n paths function to reduce runtime
    - In-memory databases like SQLite or Redis might be worth exploring for fast read-write capabilities.
    
    - Distributed computing like Apache Spark might be able to handle large datasets more efficiently.
    
    - Write asynchronous functions or perform multithreading (be aware of Global Interpretation Lock for python).

    - Batch multiple simulation results together if read and write time is significant.

    - Consider different data formats like HDF5 or Parquet.

    - Choose representative samples to represent the data instead of the entire dataset.

    - Numba, packages or purchase GPU computing power to increase the number  of simulations.

    - Check whether the constraint is on the CPU or GPU of the hardware.

- Calibration (prof's suggestion)
    - Use the first PPD's calibrated hparams to generate the entire path.

    - Plot out the volatility smile curve for the entire path (using its V_t  as implied volatility and the options data to for moneyness).

    - If the volatility smile curve is not good, regenerate.

    - Do this for all the underlying assets.

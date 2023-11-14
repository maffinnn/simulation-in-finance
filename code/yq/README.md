# README
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

### Challenges
- Calibration error debugging
    - Potential pitfalls: taking sqrt(x) or log(x) where x is negative.
    doing x**y where x is negative. Since y is real, there will be a fractional component, and a negative number to a fractional exponent is not a real number.
    doing x/y where both x and y are 0.

- Code writing
    - It is very time consuming to write codes in IPYNB that requires import since every
    time the kernel needs to be restarted to re-import the updated modules. Ended up
    using `yq_scripts.py`

    - There are countless ways to organise. Choose one based on the priorities and be settled
    with that way.

- Mathematical models
    - Translating from the models' form to codes need time. Consider at least one day to understand the concepts per model, and more days for those with calibration, exogenous data,
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
- Consider planning out the final presentation style first before even starting,
can save time on refactoring gbm and heston classes multiple times.

- Use logger at different levels .

- Take care of wellbeing! Do not sleep too late because of one project!.


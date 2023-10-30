# Chapter 10 - Resampling for performance evaluation -----

## 10.1 The resubstitution approach ----

# When we measure performance on the same data that we used for training data
# (as opposed to new data), we say we have *resubstituted* the data.


# In Section 8.8 we have summarized the current state of our Ames housing data analysis.

# It includes:
# - A "recipe" object `ames_rec`
# - A linear model
# - A workflow using that recipe and model, called `lm_wflow`
# - The workflow fit to the training set `ames_train`, resulting in `lm_fit`


# We can also fit a *random forest* as a tree ensemble method that creates a large number
# of decision trees from slightly different versions of the training set.

# This collection of trees makes up the ensemble.

# When predicting a new sample, each ensemble member makes a separate prediction.
# These are averaged to create the final ensemble prediction for the new data point.


# While random forest models can be computationally intensive, they are very low maintenance.
# Very little preprocessing is required.


# Using the same predictor set as the linear model, but without the extra preprocessing steps,
# we fit a random forest model to the training set via the `"ranger"` engine,
# which uses the "ranger" R package for computation.

# This model requires no preprocessing, so a simple formula can be used:
library(tidymodels)
tidymodels_prefer()

data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))


set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)


## Linear model
ames_rec <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train
  ) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |> 
  step_ns(Latitude, Longitude, deg_free = 20)

lm_model <- linear_reg() |> 
  set_engine("lm")

lm_wflow <- workflow() |> 
  add_model(lm_model) |> 
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)


## Random forest model
rf_model <- rand_forest(trees = 1000) |> 
  set_engine("ranger") |> 
  set_mode("regression")

rf_wflow <- workflow() |> 
  add_formula(
    Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude
  ) |> 
  add_model(rf_model)

rf_fit <- rf_wflow |> 
  fit(data = ames_train)


# How should we compare the linear model and the random forest model?
# We will predict the training set to produce an *apparent metric* or
# *resubstitution metric*.

# We create a function that predicts and formats the results:
estimate_perf <- function(model, dat) {
  # Capture the names of the `model` and `dat` objects
  cl <- match.call()
  obj_name <- as.character(cl$model)
  data_name <- as.character(cl$dat)
  data_name <- gsub(pattern = "ames_", replacement = "", x = data_name)
  
  
  # Estimate these metrics:
  reg_metrics <- metric_set(rmse, rsq)
  
  return(
    model |> 
      predict(dat) |> 
      bind_cols(dat |> select(Sale_Price)) |> 
      reg_metrics(Sale_Price, .pred) |> 
      select(-.estimator) |> 
      mutate(
        object = obj_name,
        data = data_name
      )
  )
}

# Both RMSE and R-squared are computed.
# The resubstitution statistics are:
estimate_perf(model = rf_fit, dat = ames_train)
# rmse: 0.0364, rsq: 0.960

estimate_perf(model = lm_fit, dat = ames_train)
# rmse = 0.0754, rsq: 0.816

# For prediction, the random forest is twice as good as the linear model.

# Next, we apply the random forest model to the test set:
estimate_perf(model = rf_fit, dat = ames_test)
# rmse: 0.0701, rsq: 0.853

estimate_perf(model = lm_fit, dat = ames_test)
# rmse: 0.0735, rsq: 0.836

# The linear model is as good as the random forest. 
# This means the random forest model was clearly overfitted.


# Predictive models that are capable of learning complex trends in the data
# are called *low bias models*.

# *Bias* refers to the difference between the true pattern in the data and the
# patterns the model can emulate.

# Many black-box machine learning models have low bias, meaning they can reproduce
# complex, non-linear relationships.

# Other models, such as linear regression, logistic regression, or discriminant analysis,
# are not as adaptable and are called *high bias models*.


# For a low bias model, the high degree of predictive capacity can result in the model
# nearly memorizing the training set data.

# Consider a 1-nearest neighbor model.
# It will always provide perfect predictions for the training set no matter how well it
# truly works for other data sets.


## 10.2 Resampling methods ----

# Resampling methods emulate the process of using some data for modeling and different data
# for evaluation.

# Most resampling methods are iterative.

# Resampling is exclusively conducted on the training set.

# For each iteration of resampling, the training set is partitioned into two subsamples:

# - The model is fit with the *analysis set*.

# - The model is evaluated with the *assessment set*.


### 10.2.1 Cross-validation ----

# The most common cross-validation method is *V-fold cross-validation*.

# The training set is randomly partitioned into V sets of roughly equal size,
# called *folds*.

# At each iteration, one fold is held out for assessment while the remaining V-1
# folds are substrate for the model.

# This process continues for each fold so that V models produce V sets of performance
# statistics.

# The final resampling estimate of performance averages each of the V replicates.

# In practice, values of V are most often 5 or 10, and 10-fold cross-validation
# is a good default choice.

# The primary input is the training set data frame and the number of folds
# (defaulting to 10):
set.seed(1001)
ames_folds <- vfold_cv(ames_train, v = 10)
print(ames_folds)


# The column named `splits` contains the information on how to split the data,
# similar to the object used to create the initial training/test partition.

# While each row of `splits` has an embedded copy of the entire training set,
# R is smart enough not to make copies of the data in memory.

# The `print()` method inside of the tibble shows the frequency of each:
# `[2107/235]` or `[2108/234]` indicates that about two thousand samples are 
# in the analysis set and over 200 are in an assessment set.


# These objects always contain a character column called `id` that labels the partition.

# To manually retrieve the partitioned data, the `analysis()` and `assessment()`
# functions return the corresponding data frames:

# For the first fold:
ames_folds$splits[[1]] |> 
  analysis() |> 
  dim()
# 2107 74

# The "tidymodels" packages, such as "tune", contain interfaces like `analysis()`
# that are generally not needed for day-to-day work.


### Repeated cross-validation ----

# To create R repeats of V-fold cross-validation, the same fold generation process
# is done R times to generate R collections of V partitions.

# Now, instead of averaging V statistics, V x R statistics produce the final
# resampling estimate.

# Due to the Central Limit Theorem (CLT), the summary statistics from each
# model tend toward a normal distribution, as long as we have a lot of data
# relative to V x R.


# Consider the Ames housing data.
# On average, 10-fold cross-validation (V = 10) uses assessment sets that
# contain roughly 234 properties.

# If RMSE is the performance statistic of choice, we can denote that estimate's
# standard deviation as sigma.

# With simple 10-fold cross-validation, the standard error of the mean RSME
# is sigma / sqrt(10).

# If this is too noisy, repeats reduce the standard error to
# sigma / sqrt(10xR)


# Note that larger numbers of replicates tend to have less impact on the standard error.

# However, if the baseline value of sigma is impractically large,
# the diminishing returns on replication may still be worth the extra computational costs.


# To create repeats, invoke `vfold_cv()` with the additional argument `repeats`:
vfold_cv(ames_train, v = 10, repeats = 5)


### Leave-one-out cross-validation -----

# An extreme form of V-fold cross-validation is called leave-one-out cross-validation (LOOCV).

# If there are n observations in your training set, you'll fit n models on
# n-1 observations and use each model to predict the single excluded data point.

# Pool these n predictions to produce a single performance statistic.


# For large samples, LOOCV is computationally excessive.

# While the "rsample" package contains a `loo_cv()` function, we advice against its usage.


### Monte carlo cross-validation -----

# A variant of V-fold cross-validation is called Monte Carlo cross-validation (MCCV).

# For MCCV, the proportion of the data that is allocated to the assessment set is
# randomly selected each time.

# This results in assessment sets that are not mutually exclusive.

# to create such resampling objects, call:
mc_cv(ames_train, prop = 9/10, times = 20)



### 10.2.2 Validation sets ----

# A validation set is a single partition that is set aside to estimate the performance.

# Split the initially available data into a training set, a test set, and a validation set.


# The "rsamples" package provides the function `validation_set()` that takes the results
# of `initial_validation_split()` and converts it to an "rset" object:

# Previously:
set.seed(52)
# To put 60% into training, 20% into validation, and 20% into testing:
ames_val_split <- initial_validation_split(ames, prop = c(0.6, 0.2))
print(ames_val_split)


# Object used for resampling:
val_set <- validation_set(ames_val_split)
print(ames_val_split)


### 10.2.3 Bootstrapping ----

# Bootstrap resampling was originally invented to approximate sampling distributions of 
# statistics whose theoretical properties are intractable.

# Using it to estimate model performance is a secondary application of the method.


# A bootrap sample of the training set is a sample that is the same set as the
# training set but is drawn *with replacement*.

# This means that some observations from the training set are selected multiple times for 
# the analysis set.

# The probability of a single observation of being included at least once
# in the analysis set is 0.632.

# The assessment set contains all observations that were not selected for the analysis set,
# which means, on average 36.8% of the training set.

# The assessment set is sometimes called the *out-of-bag* sample.


# Using the "rsample" package, we create bootstrap resamples:
bootstraps(ames_train, times = 5)

# Bootstrap samples produce performance estimates that are low in variance 
# (as opposed to cross-validation performance estimates that are high in variance),
# but have pessimistic bias.


# This means that, if the true accuracy of a model were 0.9, the bootstrap would estimate
# the accuracy to be less than 0.9.

# The amount of bias cannot be empirically determined with sufficient accuracy.

# Additionally, the amount of bias changes over the scale of the performance metric.


# The bootstrap is used inside of many models.
# The random forest model mentioned before contained 1,000 individual decision trees,
# where each tree was the product of a different bootstrap sample of the training set.


### 10.2.4 Rolling forecast origin resampling ----

# When data have a strong time component, a resampling method should support modeling
# to estimate seasonal and other temporal trends within the data.

# A technique that randomly samples values from the training set can disrupt the
# model's ability to estimate these patterns.


# Hyndman and Athanasopoulos (2018) propose a method called
# *rolling forecast origin resampling*.

# This method emulates how time series data is often partitioned in practice,
# estimating the model with historical data and evaluating it with the most
# recent data.


# For this method, the size of the initial analysis and assessment sets are specified.

# The first iteration of resampling uses these sizes, starting from the beginning
# of the series.

# The second iteration uses the same data sizes, but shifts over by a set number
# of samples.


# There are two different configurations of this method:

# - The analysis set can cumulatively grow (as opposed to remaining the same size).
#   After the first initial analysis set, new samples can accrue without discarding
#   the earlier data.

# - The resamples need not increment by one. 
#   For large data sets, the incremental block could be a week or a month instead
#   of a day.


# For one year's worth of data, suppose that six sets of 30-day blocks
# define the analysis set.

# For assessment sets of 30 days with a 29-day skip, we can use the
# "rsample" package to specifiy:
time_slices <- tibble(x = 1:365) |> 
  rolling_origin(initial = 6 * 30, assess = 30, skip = 29, cumulative = FALSE)

head(time_slices)
# A tibble: 6 * 2
# splits          id
# <list>          <chr>
# <split [180/3]> Slice 1
# ....
# <split [180/3]> Slice6


data_range <- function(x) {
  summarise(x, first = min(x), last = max(x))
}


map_dfr(.x = time_slices$splits, .f = ~ analysis(.x) |> data_range())
# A tibble: 6 * 2
# first   last
# <int>  <int>
#   1     180
#  31     210
#  61     240
#  91     270
# 121     300
# 151     330

# The difference remains constant at 179 observations.


## 10.3 Estimating performance ----

# The resampling methods presented in this chapter can be used to evaluate
# the modeling process, including preprocessing and model fitting.

# The methods are effective because different groups of data are used
# to train the model and to assess the model.


# To reiterate the process of resampling:

# 1. During resampling, the analysis set is used to preprocess the data,
#    apply the preprocessing itself, and use these processed data to fit the model.

# 2. The preprocessing statistics produced by the analysis set are applied to the 
#    assessment set. 
#    The predictions from the assessment set estimate performance on new data.


# This sequency repeats for every resample.

# If there are *B* resamples, there are *B* replicates of each of the performance
# metrics.

# The final resampling estimate is the average of these *B* statistics.

# If B=1, as with a validation set, the individual statistics represent overall performance.


# Let's consider the previous random forest model contained in
# the `rf_wflow` object.

# `fit_resamples()` is analogous to `fit()`, but instead of having a
# `data` argument it has a `resamples` argument that expects an "rset"
# object.

# The possible interfaces to the function are:

# - `model_spec |> fit_resamples(formula, resamples, ...)`

# - `model_spec |> fit_resamples(recipe, resamples, ...)`

# - `workflow |> fit_resamples(resamples, ....)`



# There are a number of other optional arguments, such as:

# - `metrics`: A metric set of performance statistics to compute.
#   By default, regression models use RMSE and R-squared, while
#   classification models use the area under the ROC curve and overall accuracy.

# - `control`: A list created with `control_resamples()` with several options such as:
#   - `verbose`: A logical for printing logging.
#   - `extract`: A function for retaining objects from each model iteration.
#   - `save_pred`: A logical for saving the assessment set predictions.


# For our example, we save the predictions to visualize the model fit
# and the residuals:
keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

set.seed(1003)
rf_res <- rf_wflow |> 
  fit_resamples(resamples = ames_folds, control = keep_pred)

print(rf_res)
# Resampling results
# 10-fold cross-validation
# A tibble: 10 * 5
# splits id .metrics .notes .predictions
# <list> <chr> <list> <list> <list>
# split [2107/235] Fold01 <tibble [2 * 4]> <tibble> [0 * 3]> <tibble>
# ...
# split [2108/234] Fold10 <tibble [2 * 4]> <tibble [0 * 3]> <tibble>


# The return value is a tibble similar to the input resamples, along
# with some extra columns:

# - `.metrics` is a list of column of tibbles containing the assessment set
#   performance statistics.

# - `.notes` is another list-column of tibbles cataloging any warnings or
#   errors generated during resampling.
#   Note that errors will not stop subsequent execution of resampling.

# - `.predictions` is present when `save_pred = TRUE`.
#   This list-column contains tibbles with the out-of-sample predictions.


# While these list-columns may look daunting, they can easily be reconfigured
# with "tidyr" or with "tidymodels" convenience functions.

# To return the performance metrics in a more usable format:
collect_metrics(rf_res)
# A tibble: 2 * 6
# .metric .estimator  .mean      n    std_err   .config
# <chr>   <chr>       <dbl>   <int>     <dbl>   <chr>
# rmse    standard    0.0721    10    0.00305   Preprocessor1_Model1
# rsq     standard    0.831     10    0.0108    Preprocessor1_Model1


# These are the resampling estimates over the individual replicates.
# To get the metrics for each resample, use the option `summarize = FALSE`.

# Notice how much more realistic the performance estimates are
# compared to the resubstitution estimates from Section 10.1.


# To obtain the assessment set predictions, call:
assess_res <- collect_predictions(rf_res)
print(assess_res)


# For consistency and ease of use, the prediction column follows the conventions
# used by "parsnip" models from Chapter 6.

# The observed outcome column always uses the original column name from the source data.

# The `.row` column is an integer that matches the row of the original training set
# so that these can be properly arranged and joined with the original data.


# For some resampling methods such as the bootstrap or repeated cross-validation,
# there will be multiple predictions per row of the original data set.

# To obtain summarized values, such as averages of the replicate predictions,
# use `collect_predictions(object, summarize = TRUE)`.


# Since this analysis used 10-fold cross-validation, there is one unique prediction
# for each training set sample.

# These data can generate helpful plots of the modle to understand where it
# potentially failed.

# Compare the observed and held-out predicted values:
assess_res |> 
  ggplot(mapping = aes(x = Sale_Price, y = .pred)) +
  geom_point(alpha = 0.15) +
  geom_abline(color = "red") +
  coord_obs_pred() +
  ylab("Predicted") +
  theme_bw()

graphics.off()

# There are two houses in the training set with a low observed sale price
# that are significantly overpredicted by the model.

# We can find these houses from the `assess_res` result:
over_predicted <- assess_res |> 
  mutate(
    residual = Sale_Price - .pred
  ) |> 
  arrange(desc(abs(residual))) |> 
  slice(1:2)

print(over_predicted)


ames_train |> 
  slice(over_predicted$.row) |> 
  select(Gr_Liv_Area, Neighborhood, Year_Built, Bedroom_AbvGr, Full_Bath)

# Identifying examples like these with especially poor performance can help us
# follow up and investigate why these specific predictions are so poor.


# Let's move back to the homes overall.
# How can we use a validation set instead of cross-validation?
# From the previous "rsample" object:
val_res <- rf_wflow |> 
  fit_resamples(resamples = val_set)
# Warning:
# In `[.tbl_df`(x, is.finite(x <- as.numeric(x))) :
#   NAs introduced by coercion

print(val_res)

collect_metrics(val_res)
# rmse: 0.0727, rsq: 0.823

# These results are also much closer to the test set results than the
# resubstitution estimates of performance.


# In these analyses, the resampling results are very close to the test set results.

# The two types of estimates tend to be well correlated.

# However, this could be from random chance.

# A seed value of `55` fixed the random numbers before creating
# the resamples.

# Try changing this value and re-running the analyses to investigate whether the
# resampled estimates match the test set results as well.


## 10.4 Parallel processing ----

# The models created during resampling are independent of one another.

# Computations of this kind are sometimes called *embarrassingly parallel*;
# each model could be fit simultaneously without issues.


# The "tune" package uses the "foreach" package to facilitate parallel computations.

# These computations could be split across processors on the same computer or 
# across different computers, depending on the technology chosen.


# For computations conducted on a single computer, the number of possible
# worker processes is determined by the "parallel" package:

# The number of physical cores in the hardware:
parallel::detectCores(logical = FALSE)
# 6

# The number of possible independent processes that can be simultaneously used:
parallel::detectCores(logical = TRUE)
# 12


# The difference between these two values is related to the computer's processor.

# For example, most Intel processors use hyperthreading,
# which creates two virtual cores for each physical core.

# While these extra resources can improve performance, most of the speed-ups
# produced by parallel processing occur when processing uses fewer than the
# number of physical cores.


# For `fit_resamples()` and other functions in "tune", parallel processing 
# occurs when the user registers a parallel backend package.

# These R packages define how to execute parallel processing.


# An alternative approach to parallelizing computations uses network sockets.

# The "doParallel" package enables this method.
library(doParallel)

# Create a cluster object and then register:
cl <- makePSOCKcluster(2)
registerDoParallel(cl)

# Now run `fit_resamples()` ...

stopCluster(cl)


# Another R package that facilitates parallel processing is the "future" package.

# Like "foreach", it provdes a framework for parallelism.

# This package is used in conjunction with "foreach" via the "doFuture" package.

# Note that all R packages with parallel backends for "foreach" start with
# the prefix "do".


# Parallel processing with the "tune" package tends to provide linear speed-ups
# for the first few cores.

# This means that with two cores, the computations are twice as fast.

# Depending on the data and model type, the linear speed-up deteriorates
# after four to five cores.


# A final note about parallelism:
# For each of these technologies, the memory requirements multiply
# for each additional core used.

# For example, if the current data set is 2GB in memory and three cores are used,
# the total memory requirement is 8GB (2 for each worker process plus the original).

# Using too many cores will cause these computations and your computer to slow considerably.


## 10.5 Saving the resampled objects ----

# The models created during resampling are not retained.
# These models are trained for the purpose of evaluating performance,
# and we discard them after having computed the performance statistics.


# If a particular modeling approach turns out to be optimal for our data set,
# the best choice is to fit this model once more but this time
# on the whole training data set.
# That way, the model parameters are estimated with more data.


# There is a method for keeping these models or some of their components.
# The `control_resamples()` functions has an `extract` argument that specifies
# a function that takes a single argument like `x`.

# When executed, `x` results in a fitted workflow object,
# regardless of whether you provided `fit_resamples()` with a workflow.


# Recall that the "workflows" package has functions that can pull the different
# components of the objects (model, recipe, etc.).


# We fit a linear regression model using the "recipe" developed in Chapter 8:
ames_rec <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train
  ) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |> 
  step_ns(Latitude, Longitude, deg_free = 20)

lm_wflow <- workflow() |> 
  add_recipe(ames_rec) |> 
  add_model(linear_reg() |> set_engine("lm"))

lm_fit <- lm_wflow |> 
  fit(data = ames_train)


# Select the recipe:
extract_recipe(lm_fit, estimated = TRUE)


# We can save the linear model coefficients for a fitted model object from a workflow:
get_model <- function(x) {
  extract_fit_parsnip(x) |> 
    tidy()
}

# Test it using: `get_model(lm_fit)`


# Now we apply the function `get_model()` to the ten resampled fits.
# The results of the extraction function are wrapped in a "list" object
# and returned in a "tibble":
ctrl <- control_resamples(extract = get_model)

lm_res <- lm_wflow |> 
  fit_resamples(resamples = ames_folds, control = ctrl)

print(lm_res)

# Now there is a `.extracts` column with nested tibbles.
lm_res$.extracts[[1]]

# To get the results
lm_res$.extracts[[1]][[1]]

# For our simple example, all of the results can be flattened and collected using:
all_coef <- map_dfr(.x = lm_res$.extracts, .f = ~ .x[[1]][[1]])

# Show all replications for a single predictor:
all_coef |> 
  filter(term == "Year_Built")


# END
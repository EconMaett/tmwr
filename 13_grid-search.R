# Chapter 13 - Grid Search ----

# In the previous chapter we demonstrated how you can use the `tune()` function to
# mark arguments in preprocessing recipes and/or model specifications.

# Once we know what parameter to tune, it's time to address the question of how to optimize them.


# This chapter describes the *grid search* methods that specify a possible set of values
# for the parameter *a priori*.


## 13.1 Regular and nonregular grids ----

# There are two main types of grids:

# - A regular grid combines each parameter (with its corresponding set of possible values)
#   factorially, i.e., by using all combinations of sets.

# - A nonregular grid is one where the parameter combinations are not formed from a small set of points.


# Consider a multilayer perceptron model (a.k.a. single layer artificial neural network).
# The parameters for tuning are:

# - the number of hidden units
# - the number of fitting epochs/iterations in model training
# - the amount of weight decay penalization


# Historically, the number of epochs was determined by early stopping;
# a separate validation set determined the length of training based on the error rate,
# since re-predicting the training set led to overfitting.

# In our case, the use of a weight decay penalty should prohibit overfitting,
# and there is little harm in tuning the penalty and the number of epochs.


# Using "parsnip", the specification for a classification model fit using the 
# "nnet" package is:
library(tidymodels)
tidymodels_prefer()


mlp_spec <- mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |> 
  set_engine("nnet", trace = 0) |> 
  set_mode("classification")

# The `trace = 0` argument prevents extra logging of the training process.

# `extract_parameter_set_dials()` extracts the set of arguments with unknown values
# and sets their "dials" objects:
mlp_param <- extract_parameter_set_dials(mlp_spec)

mlp_param |> 
  extract_parameter_dials("hidden_units")
# Hidden Units (quantitative)
# Range: [1, 10]

mlp_param |> 
  extract_parameter_dials("epochs")
# Epochs (quantitative)
# Range: [10, 1000]

# The parameter objects are complete and their default range are set.


### Regular grids 

# Regular grids combine separate sets of parameters.

# First, the user creates a distinct set of values for each parameter.
# The number of parameters need not be the same.

# The "tidyr" function `crossing()` is one way to create a regular grid:
crossing(
  hidden_units = 1:3, 
  penalty = c(0.0, 0.1),
  epochs = c(100, 200)
)

# The parameter object knows the ranges of the parameters.
# The "dials" package contains a set of `grid_*()` functions that take the 
# parameter object as input to produce different types of grids.

grid_regular(mlp_param, levels = 2)
# The `levels` argument is the number of levels per parameter to create.

# It can also take a vnamed vector of values:
mlp_param |> 
  grid_regular(levels = c(hidden_units = 3, penalty = 2, epochs = 2))

# There are techniques for creating regular grids that do not use all possible values
# of each parameter set.

# Such *fractional factorial designs* are explained on the CRAN Task View for experimental design.


# Regular grids are computationally expensive when many tuning parameters are used.


# There are, however, some models whose tuning time *decreases* with a regular grid!


### Irregular grids

# One way to create a non-regular grid is to use random sampling across the range
# of parameters with the `grid_random()` function.

# `grid_random()` produces uniform random numbers across the parameter ranges.


# If the parameter object has an associated transformation, such as for `penalty`,
# `grid_random()` will produce random numbers on the same transformed scale.


# Create a random grid for the parameters from our neural network:
set.seed(1301)

mlp_param |> 
  grid_random(size = 1000) |> # `size` is the number of combinations
  summary()


# With small grids, random values can result in overlapping parameter combinations.

# The random grid needs to cover the whole parameter space, 
# but the likelihood of good coverage increases with the number of grid values.

# For a sample of 15 candidate points, the following figure shows some overlap between
# points for our multilayer perceptron:
library(ggforce)

set.seed(1302)

mlp_param |> 
  # `original = FALSE` keeps penalty in log10 units
  grid_random(size = 20, original = FALSE) |> 
  ggplot(mapping = aes(x = .panel_x, y = .panel_y)) +
  geom_point() +
  geom_blank() +
  ggforce::facet_matrix(vars(hidden_units, penalty, epochs), layer.diag = 2) +
  labs(
    title = "Random design with 20 candidates"
  ) +
  theme_bw()

graphics.off()


# A better approach is to use a set of experimental designs called
# *space-filling* designs.

# While different design methods have different goals, they generally find a configuration
# of points that cover the parameter space with the smallest chance of overlapping or redundant values.


# The "dials" package contains functions for *Latin hypercube* or *maximum entropy* designs.

# Compare a random design with a Latin hypercube design for 20 candidate parameter values:
set.seed(1303)

mlp_param |> 
  grid_latin_hypercube(size = 20, original = FALSE) |> 
  ggplot(mapping = aes(x = .panel_x, y = .panel_y)) +
  geom_point() +
  geom_blank() +
  ggforce::facet_matrix(vars(hidden_units, penalty, epochs), layer.diag = 2) +
  labs(
    title = "Latin Hypercube design with 20 candidates"
  ) +
  theme_bw()

graphics.off()

# While not perfect, this Latin hypercube design spaces the point farther away from one another
# and allows for a better exploration of the hyperparameter space.


# Such space-filling designs can be effective at representing the parameter space.


# The default design used by the "tune" package is maximum entropy design.

# These tend to produce grids that cover the candidate space well
# and increase the chances of finding good results.


## 13.2 Evaluating the grid ---

# To choose the best combination of tuning parameters, every candidate set
# is assessed using the data that were not used to train that model.

# Resampling methods or a single validation set work well for that purpose.


# The user then selects the appropriate candidate parameter set.


# We use a classification data set from Hill et al. (2007), who developed an automated
# microscopy laboratory tool for cancer research.

# The data consists of 56 imaging measurements on 2019 human breast cancer cells.

# The predictors represent shape and intensity characteristics of different parts
# of the cells, such as the nucleus, the cell boundary, etc.

# There is a high degree of correlation between the predictors.

# Some predictors will have a skewed distribution.


# Each cell belongs into one of two classes.

# The data are in the "modeldata" package.
# We remove the `case` column that is not needed for the analysis:
library(tidymodels)

data(cells)

cells <- cells |> 
  select(-case)

dim(cells)
# 2019 57


# We compute performance metrics with 10-fold cross-validation:
set.seed(1304)

cell_folds <- vfold_cv(cells)

# We use PCA feature extraction to decorrelate the predictors.

# The following recipe contains steps to transform the predictors to:
# - increase symmetry
# - normalize their scale
# - conduct feature extraction.

# The number of PCA components to retain is also tuned, along with the model parameters.


# We also add a recipe step that estimates a Yeo-Johnson transformation for each predictor
# since extreme values will influence PCA, which is based on variance.

mlp_rec <- recipe(class ~ ., data = cells) |> 
  step_YeoJohnson(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_pca(all_numeric_predictors(), num_comp = tune()) |> 
  step_normalize(all_numeric_predictors())


mlp_wflow <- workflow() |> 
  add_model(mlp_spec) |> 
  add_recipe(mlp_rec)


# We create the parameter object `mlp_param` to adjust the default ranges.

# We change the number of epochs to the smaller range 50-200
# and the default range for `num_comp()` is increased from 1-4 to 0-40:
mlp_param <- mlp_wflow |> 
  extract_parameter_set_dials() |> 
  update(
    epochs = epochs(c(50, 200)),
    num_comp = num_comp(c(0, 40))
  )


# `tune_grid()` is the function to conduct grid search. Its arguments are:

# - `grid`: An integer or data frame.

# - `param_info`: Used when `grid` is an integer. Defines the parameter range.


# `tune_grid()` requires an "rsample" object like `cell_folds`.


# Evaluate a regular grid with three levels across the resamples:
roc_res <- metric_set(roc_auc)

set.seed(1305)


# Caution: This takes a few minutes to run.
mlp_reg_tune <- mlp_wflow |> 
  tune_grid(
    cell_folds,
    grid = mlp_param |> grid_regular(levels = 3),
    metrics = roc_res
  )


print(mlp_reg_tune)


# The `autoplot()` method for regular grids shows the performance profiles
# across tuning parameters:
theme_set(theme_bw())

autoplot(mlp_reg_tune) +
  scale_color_viridis_d(direction = -1) +
  theme(legend.position = "top")

graphics.off()

# For these data, the amount of penalization has the largest impact on the area under the ROC curve.

# The number of epochs on the other hand, does not appear to have a pronounced effect on performance.

# The change in the nuber of hidden units appears to matter most when the amount of
# regularization is low, and thus harms performance.


# Several parameter configurations have roughly equivalent performance,
# as shown by `show_best()`:
show_best(mlp_reg_tune) |> 
  select(-.estimator)


# Based on these results, it makes sense to conduct another grid search run
# with larger values of the weight decay penalty.


# To use a space-filling design, either the `grid` argument can be given an integer
# or one of the `grid_*()` functions can produce a data frame.

# To evaluate the same range using a maximum entropy design with 20 candidate values:

set.seed(1306)


# Caution: This takes a moment to run
mlp_sfd_tune <- mlp_wflow |> 
  tune_grid(
    cell_folds, 
    grid = 20,
    # Pass in the parameter object to use the appropriate range:
    param_info = mlp_param,
    metrics = roc_res
  )

print(mlp_sfd_tune)


# `autoplot()` also works with this design, although the format is somewhat different:
autoplot(mlp_sfd_tune) +
  scale_color_viridis_d(direction = -1) +
  theme(legend.position = "top")

graphics.off()

# This marginal effects plot shows the relationship of each parameter with the 
# performance metric.

# Note that since no regular grid is used, the values of the other tuning parameters
# can affect each pane!


# The penalty parameter appears to result in better performance with smaller amounts
# of weight decay.

# This is the opposite of the results from the regular grid.


# `show_best()` reports the numerically best results:
show_best(mlp_sfd_tune) |> 
  select(-.estimator)


## 13.3 Finalizing the model ----

# After you chose one of the sets of possible model parameters from `show_best()`,
# you can evaluate them on the test set.


# To fit a final model, a final set of parameter values is determined.

# You can then either:

# - manually pick values that appear appropriate
# - use a `select_*()` function


# `select_best()` choses the parameters with the numerically best results:
select_best(mlp_reg_tune, metric = "roc_auc")
# hidden_units: 5
# penalty: 1
# epochs: 50
# num_comp: 0
# .config: Preprocessor1_Model08


# The model with a single hidden unit trained for 125 epochs on the original predictors
# with a large amount of penalization has performance competitive with this option
# and is simpler.

# This is basically a penalized logistic regression!


# We can manually specify these parameters and create a tibble to use with a
# *finalization* function to splice the values into the workflow:
logistic_param <- tibble(
  num_comp = 0,
  epochs = 125,
  hidden_units = 1,
  penalty = 1
)

final_mlp_wflow <- mlp_wflow |> 
  finalize_workflow(logistic_param)

print(final_mlp_wflow)

# No more values of `tune()` are included in the finalized workflow.

# Fit the model to the entire training set:
final_mlp_fit <- final_mlp_wflow |> 
  fit(cells)


# Finalization can also be done with `finalize_model()` or `finalize_recipe()`.


## 13.4 Tools for creating tuning specifications ----

# The "usemodels" package can take a data frame and model formula,
# then write out R code for tuning the model.


# The code also creates an appropriate recipe whose steps depend on the requested model
# as well as the predictor data.


# Consider this `xboost` modeling code fit to the Ames housing data:
library(usemodels)

data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)


# Call `usemodels::use_xgboost()`:
use_xgboost(
  formula = Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train,
  # Add comments explaining some of the code:
  verbose = TRUE
)


# The code printed in the console window is as follows:

# Recipe
xgboost_recipe <- recipe(
  formula = Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude, data = ames_train
  ) |> 
  step_zv(all_predictors())

# Model specification
xgboost_spec <- boost_tree(
  trees = tune(), 
  min_n = tune(), 
  tree_depth = tune(), 
  learn_rate = tune(), 
  loss_reduction = tune(), 
  sample_size = tune()
  ) |> 
  set_mode("classification") |> 
  set_engine("xgboost") 

# Workflow
xgboost_workflow <- workflow() |> 
  add_recipe(xgboost_recipe) |> 
  add_model(xgboost_spec) 


set.seed(7019)

# Hyperparameter tuning
# You need to define `resamples` and `grid`!
xgboost_tune <- tune_grid(
  xgboost_workflow, 
  resamples = stop("add your rsample object"), 
  grid = stop("add number of candidate points")
  )


# Based on what "usemodels" understands about the data, this code is the minimal preprocessing required.

# For other models, operations like `step_normalize()` are added to fulfill the needs of the model.

# Notice that it is the user's responsibility to choose what `resamples` to use for tuning,
# as well as what kind of `grid`.


# The "usemodels" package can also be used to create model fitting code without any
# tuning parameters by setting the argument `tune = FALSE`:
usemodels::use_xgboost(
  formula = Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train,
  # Add comments explaining some of the code:
  verbose = TRUE,
  tune = FALSE
)

# The code printed in the console window is:

# Recipe
xgboost_recipe <- recipe(
  formula = Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude, 
  data = ames_train
  ) |> 
  step_zv(all_predictors()) 

# Model specification
xgboost_spec <- boost_tree() |> 
  set_mode("classification") |> 
  set_engine("xgboost") 

# Workflow
xgboost_workflow <- workflow() |> 
  add_recipe(xgboost_recipe) |> 
  add_model(xgboost_spec)


# You could fit the model to the data with the following command:
xgbootst_fit <- fit(xgboost_workflow, ames_train)

# However, the outcome data (Sales_Price) is numeric, but since `xgboost`
# is used for classification, the outcome data needs to be a `factor()`.


## 13.5 Tools for efficient grid search ----

# You can make grid search more efficient with several techniques.


### 13.5.1 Submodel optimization ----

# There are types of models where multiple tuning parameters can be evaluated without 
# refitting the model, so only a single model fit is needed.


# Partial least squares (PLS) is a supervised version of principal component analysis (PCA).

# It creates components that maximize the variation in the predictor, like PCA, but simultaneously
# tries to maximize the correlation between these predictors and the outcome.

# One tuning parameter is the number of PLS to retain.

# If your data set has 100 predictors, the number of possible components to retain
# ranges from 1 to 50.

# Usually, a single model fit can compute the predicted values across many values of
# `num_comp`.

# As a result, the PLS model with 100 components can also make predictions for any 
# `num_comp <= 100`, which saves a lot of time.


# Other models that exploit this feature are:

# - Boosting models can make predictions across multiple values for the number of boosting iterations.

# - Regularization methods such as the "glmnet" model can make simultaneous predictions
#   across the amount of regularization used to fit the model.

# - Multivariate adaptive regression splines (MARS) adds a set of nonlinear features to
#   linear regression models.
#   The number of terms to retain is a tuning parameter, and it is computationally efficient
#   to make predictions across many values of this parameter from a single model fit.


# The "tune" package automatically applies this type of optimization whenever an applicable model is tuned.

# If a boosted C5.0 classification model is fit to the cell data, we can tune the number
# of boosting iterations, `trees`.l

# All other parameters set at their default values, we evaluate iterations from 1 to 100
# on the same resamples as used previously:
c5_spec <- boost_tree(trees = tune()) |> 
  set_engine("C5.0") |> 
  set_mode("classification")


set.seed(1307)

# Caution: This takes a moment to run
c5_spec |> 
  tune_grid(
    class ~ .,
    resamples = cell_folds,
    grid = data.frame(trees = 1:100),
    metrics = roc_res
  )


# Without submodel optimization, the call to `tune_grid()` takes about an hour
# to resample 100 submodels!

# With the optimization, the same call takes about 100 seconds, a 37-fold speed-up.

# The reduced time is the difference in `tune_grid()` fitting 1,000 models
# versus 10 models.

# Note that even if you fit the model with and without the submodel prediction trick,
# this optimization is automatically applied by "parsnip".


### 13.5.2 Parallel processing ----

# Parallel processing is an effective method for decreasing execution time when resampling models.

# This advantage conveys to model tuning via grid search, but there are some considerations.


# Consider two parallel processing schemes.


# When tuning models via grid search, there are two distinct loops:

# - one over resamples
# - one over the unique tuning parameter combinations

# Pseudocode:
for (rs in resamples) {
  # Create analysis and assessment sets
  # Preprocess data (e.g. formula/recipe)
  
  for (mod in cofnigurations) {
    # Fit model {mod} to the {rs} analysis set
    # Predict the {rs} assessment set
  }
}


# By default, "tune" parallelizes over the resamples (the outer loop).


# This is the optimal scenario when the preprocessing method is expensive. 

# The two potential downsides to this approach are:

# - It limits the achievable speed-ups when preprocessing is inexpensive.

# - The number of parallel workers is limited by the number of resamples.
#   With 10-fold cross-validation, you can only use 10 parallel workers even if
#   the computer has more than 10 cores.


# Assume we have 7 model tuning parameter values and 5-fold cross-validation.

# These tasks are then allocated to 5 worker processes.

# Each fold is assigned to its own worker process and, since only model parameters
# are being tuned, the preprocessing is conducted once per fold/worker.

# If fewer than five worker processes were used, some workers would receive multiple folds.


# In the control functions for the `tune_*()` functions, the argument
# `parallel_over` controls how the process is executed.

# The previous palatalization strategy uses `parallel_over = "resamples"`.


# Instead of parallel processing the resamples, an alternative scheme combines the
# loops over resamples and models into a single loop.

# Pseudocode:
all_tasks <- crossing(resamples, configurations)


for (iter in all_tasks) {
  # Create analysis and assessment sets for {iter}
  # Preprocess data (e.g. formula/recipe)
  # Fit model {iter} to the {iter} analysis set
  # Predict the {iter} assessment set
}


# Parallelization now occurs over a single loop.


# If we use 5-fold cross-validation with M tuning parameter values,
# the loop is executed over 5 x M iterations.

# This increases the number of potential workers that can be used.

# However, the work related to data preprocessing is repeated multiple times.

# If those steps are expensive, this approach is inefficient.


# In "tidymodels", validations sets are treated as a single resample.

# Here, the following parallelization scheme is optimal:

# Each of the 10 worker processes handles multiple folds, and the preprocessing is
# needlessly repeated.

# The control function argument is `parallel_over = "everything"`.


### 13.5.3 Benchmarking boosted trees ----

# To compare different parallelization schemes,
# a boosted tree with the "xgboost" engine is tuned
# using a data set of 4,000 samples,
# with 5-fold cross-validation 
# and 10 candidate models.


# The data required some baseline preprocessing that did not require any estimation.

# The preprocessing is handled in three different ways:

# 1. Preprocess the data prior to modeling using a "dplyr" pipeline ("none").

# 2. Conduct the same preprocessing via a recipe ("light").

# 3. With a recipe, add an additional step that has a high computational cost ("expensive").


# The first and second preprocessing options are designed for comparison,
# to measure the computational cost of the recipe in the second option.


# The third option measures the cost of performing redundant computations with
# `parallel_over = "everything"`.


# We evaluated this process using variable numbers of worker processes and using
# two `parallel_over` options,
# on a computer with 10 physical and 20 virtual cores (via hyper-threading).


# Since there were only five resamples, the number of cores used when 
# `parallel_over = "resamples"` is limited to five.


# Comparing the curves in the first two panels for "none" and "light":

# - There is little difference in the execution times between the panels.
#   This indicates, for these data, there is no computational penalty for the
#   preprocessing steps in the recipe.

# - There is some benefit for using `parallel_over = "everything"` with many cores.
#   The majority of the benefit occurs in the first five workers.


# With the "expensive" preprocessing step, there are considerable differences in execution times.

# `parallel_over = "everything"` is problematic since even when using all available cores,
# the execution time of `parallel_over = "resamples"` is never achieved.

# This is because the costly preprocessing step is unnecessarily repeated.


# The best speed-ups occur when `parallel_over = "resamples"` and when the
# computations are expensive.


### 13.5.4 Access to global variables ----

# When using "tidymodels", it is possible to use values in your local environment
# (usually the global environment) in model objects.

# If we define a variable to use as a model parameter and then pass it to a function
# like `linear_reg()`, the variable is typically defined in the global environment:
coef_penalty <- 0.1

spec <- linear_reg(penalty = coef_penalty) |> 
  set_engine("glmnet")

print(spec)
# Linear Regression Model Specification (regression)

# Main Arguments:
#   penalty = coef_penalty

# Computational engine: glmnet


# Models created with the "parsnip" package save arguments like these
# as *quosures*;
# These are objects that track both the name of the object as well as
# the environment where it lives:

spec$args$penalty
# <quosure>
# expr: ^coef_penalty
# env:  global

# Notice that we have `env: global` because this variable was created in the global environment.

# The model specification defined by `spec` works correctly when run in a user's regular
# session because that session is also using the global environment;
# R can easily find the object `coef_penalty`.


# When this model is evaluated by parallel workers, it may fail however.


# If you want this code to run in parallel, you should insert the actual data
# into the objects rather than the reference to the object.

# The "rlang" and "dplyr" packages can be very helpful for this.

# The "bang-bang" operator, `!!`, can splice a single value into an object:
spec <- linear_reg(penalty = !!coef_penalty) |> 
  set_engine("glmnet")

spec$args$penalty
# <quosure>
# expr: ^0.1
# env:  empty

# Now the output `expr` is `^0.1` instead of `^coef_penalty`, and
# the environment `env` is `empty` instead of `global`.


# When you have multiple external values to insert into an object,
# use the "bang-bang-bang" operator, `!!!`:
mcmc_args <- list(
  chains = 3,
  iter = 1000,
  cores = 3
)

linear_reg() |> 
  set_engine("stan", !!!mcmc_args)
# Linear Regression Model Specification (regression)

# Engine-Specific Arguments:
#   chains = 3
#   iter = 1000
#   cores = 3

# Computational engine: stan


# Recipe selectors are another place where you want to access global variables.

# Suppose you ahve a recipe step that should use all of the predictors in the cell
# data that were measured using the second optical channel.

# We create a vector of these column names:
library(stringr)

ch_2_vars <- str_subset(string = names(cells), pattern = "ch_2")
print(ch_2_vars)
# "avg_inten_ch_2" "total_inten_ch_2"


# We could hard-code these into a recipe step but it would be better to reference them
# programmatically in case the data change.

# There are two ways to achieve this:


# Still uses a reference to global data 
recipe(class ~ ., data = cells) |> 
  step_spatialsign(all_of(ch_2_vars))
# --- Recipe -----
# --- Inputs
# Number of variables by role
# outcome: 1
# predictor: 56

# --- Operations
# * Spatial sign on: all_of(ch_2_vars)


# Inserts the values into the step
recipe(class ~ ., data = cells) |> 
  step_spatialsign(!!!ch_2_vars)
# --- Recipe ----
# --- INputs
# Number of variables by role
# outcome: 1
# predictor: 56

# --- Operations
# * Spatial sign on: "avg_inten_ch_2", "total_inten_ch_2"


# The latter is better for parallel processing because all of the needed information
# is embedded in the "recipe" object.


### 13.5.5 Racing methods ----

# An issue with grid search is that all models need to be fit across resamples before any
# tuning parameters can be evaluated.

# It would be useful to conduct some type of interim analysis at some point during
# the tuning process.

# This would be similar to *futility analysis* in clinical trials, where a drug trial
# is stopped if the new drug performs excessively poor (or well).


# *Racing methods* can be used to evaluate all models on an initial subset of resamples.

# Based on their current performance metrics, some parameter sets are not considered in
# subsequent resamples.


# Consider the multilayer perceptron tuning process with a regular grid.

# We can fit a model where the outcome is the resampled area under the ROC curve
# and the predictor is an indicator for the parameter combination.

# Taking the resample-to-resample effect into account and producing point and interval estimates
# for each parameter setting we get one-sided 95% confidence intervals that
# measure the loss of the ROC value relative to the currently best performing parameters.


# Any parameter set whose confidence interval includes zero lacks evidence that its
# performance is statistically different from the best results.

# We retain 6 settings that are resampled more.
# The remaining 14 submodels are discarded.


# This process continues for each resample;
# after the next set of performance metrics, a new model is fit to these statistics,
# and more submodels are potentially discarded.


# Racing methods are more efficient than classic grid search as long as the
# interim analysis is fast and some parameter setting show poor performance.


# The "finetune" package contains functions for racing methods.
# `tune_race_anova()` creates an ANOVA model to test for statistical significance
# of different model configurations.

# The syntax to reproduce the filtering shown previously is:
library(finetune)

set.seed(1308)

# Caution: This takes a moment to run.
mlp_sfd_race <- mlp_wflow |> 
  tune_race_anova(
    cell_folds,
    grid = 20,
    param_info = mlp_param,
    metrics = roc_res,
    control = control_race(verbose_elim = TRUE)
  )

# The `control_race()` function has options for the elimination procedure.


# There are two tuning parameter combinations under consideration once the full set of
# resamples is evaluated.

# Use `show_best()` to returns the best models, ranked by performance.
show_best(mlp_sfd_race, n = 10)
# hidden_units: 8
# penalty: 0.814
# epochs: 177
# num_comp: 15
# mean: 0.890
# std_err: 0.00966


# Only the configurations that were never eliminated are returned.


# END
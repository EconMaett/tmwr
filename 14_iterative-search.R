# Chapter 14 - Iterative search ----

# Grid search takes a pre-defined set of candidate values, evaluates them, then
# chooses the best settings.

# Iterative search methods instead predict which values to test next.

# Two search methods are outlined in this chapter:

# - Bayesian optimization, which uses a statistical model to predict parameter settings.

# - Simulated annealing, which is a global search method.

## Previous analysis
library(tidymodels)

tidymodels_prefer()

data("cells")

cells <- cells |> 
  select(-case)

set.seed(1304)

cell_folds <- vfold_cv(cells)


roc_res <- metric_set(roc_auc)


## 14.1 A support vector machine model ----

# We fit a support vector machine (SVM) on the cell segmentation data.

# The two tuning parameters to optimize are:

# - The SVM cost value

# - The radial basis function kernel parameter, sigma.


# The SVM model uses a dot product which makes it necessary to normalize, that is
# center and scale the predictors.

# We chose not to apply PCA for feature extraction because we want to use only two
# tuning parameters which we can visualize in two dimensional plots.


# The "tidymodels" objects `svm_rec`, `svm_spec`, and `svm_wflow` define the model process:
svm_rec <- recipe(class ~ ., data = cells) |> 
  step_YeoJohnson(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors())

svm_spec <- svm_rbf(
  cost = tune(), 
  rbf_sigma = tune()
  ) |> 
  set_engine("kernlab") |> 
  set_mode("classification")

svm_wflow <- workflow() |> 
  add_model(svm_spec) |> 
  add_recipe(svm_rec)


# The default parameter ranges for the tuning parameters `cost` and `rbf_sigma` are:
cost()
# Cost (quantitative)
# Transformer: log-2 [1e-100, Inf]
# Range (transformed scale): [-10, 5]


rbf_sigma()
# Radial Basis Function sigma (quantitative)
# Transformer: log-10 [1e-100, Inf]
# Range (transformed scale): [-10, 0]


# We slightly change the kernel parameter range to improve the visualizations of the search:
svm_param <- svm_wflow |> 
  extract_parameter_set_dials() |> 
  update(rbf_sigma = rbf_sigma(c(-7, -1)))



# We constructed a very large regular grid, composed of 2,500 candidate values,
# and evaluated the grid using resampling.

# This is impractical/inefficient in practice.

# But it allows for the creation of a  heatmap of the mean area under the ROC curve
# for a high density grid of tuning parameters with a visible plateau in the 
# upper-right corner.


# The following search procedures require at least some resampled performance statistics 
# before proceeding.

# The following code creates a small regular grid that resides in the flat portion
# of the parameter space.

# `tune_grid()` resamples the grid.
set.seed(1401)

start_grid <- svm_param |> 
  update(
    cost = cost(c(-6, 1)),
    rbf_sigma = rbf_sigma(c(-6, -4))
  ) |> 
  grid_regular(levels = 2)

set.seed(1402)

svm_initial <- svm_wflow |> 
  tune_grid(
    resamples = cell_folds,
    grid = start_grid,
    metrics = roc_res
  )


collect_metrics(svm_initial)

# This initial grid shows fairly equal results without a clear optimum.


## 14.2 Bayesian optimization ----

# Bayesian optimization techniques analyze the current resampling results and create
# a predictive model to suggest tuning parameter values that have yet to be evaluated.

# The suggested parameter combination is then resampled.

# These results are then used in another predictive model that
# recommends more candidate values for testing, and so on.

# This process proceeds until no further improvements occur.


### 14.2.1 A Gaussian process model ----

# The most commonly used Bayesian optimization technique is the Gaussian process (GP) model.

# It originated in spatial statistics under the name of *kriging methods*.

# A Gaussian process is a collection of random variables whose joint probability 
# distribution is a multivariate Gaussian.

# It is the collection of performance metrics for the tuning parameter candidate values.

# For the previous initial grid of four samples, the realization of these four random
# variables was 0.864, 0.863, 0.863 and 0.866.

# These are assumed to be distributed as a multivariate Gaussian.

# The inputs that define the independent variables/predictors for the Gaussian process
# are the corresponding tuning parameter values.


# Gaussian process models are specified by their mean and covariance function, 
# with the latter having the largest effect.

# The covariance function is often parameterized in terms of the input values,
# denoted with x.

# A commonly used covariance function is the
# squared exponential function:
# Cov[x_i, x_j] = exp[-1/2|x_i - x_j|^2] + sigma_ij^2

# Where sigma_ij^2 is a constant error variance term that is zero whenever i = j.

# This equation translates to:

#   As the distance between two tuning parameter combination increases,
#   the covariance between the performance metrics increases exponentially.


# The equation also implies that the variation of the outcome metric is minimized
# at the points that have already been observed, i.e., when |x_i - x_j|^2 is zero.


### 14.2.2 Acquisition functions ----

# A class of objective functions, called *acquisition functions*, facilitate the
# trade-off between mean and variance.

# The predicted variance of the GP models are mostly driven by how far away they are
# from the existing data.

# The trade-off between the predicted mean and variance for new candidates is frequently 
# viewed through the lens of exploration and exploitation:

# - *Exploration* biases the selection towards region where there are fewer, if any
#   observed candidate models. 
#   This gives more weight to candidates with higher variance and focuses on finding
#   new results.

# - *Exploitation* principally relies on the mean prediction to find the best (mean) value.
#   It focuses on existing results.


# A toy example with a single tuning parameter has values between [0, 1]
# and the performance metric is the R-squared.

# The true function is plotted, along with five candidate values that have existing results.

# For these data, a GP model fit is shown with a shaded region indicating the mean +/- 1
# standard error.

# Two vertical dashed lines indicate two candidate points that are examined later.


# The shaded confidence region demonstrates the squared exponential variance function;
# it becomes very large between points and converges to zero at the existing data points.


# One of the most commonly used acquisition functions is *expected improvement*.

# The notion of improvement requires a value for the current best results
# (unlike the confidence bound approach).

# Since the Gaussian process can describe a new candidate point using a distribution,
# we can weight the parts of the distribution that show improvement using the
# probability of the improvement occurring.


# Consider two candidate parameter values of 0.10 and 0.25, the ones indicated
# by vertical dashed lines earlier.

# Using the fitted Gaussian process model, their predicted R-squared distributions
# can be plotted.

# This helps us chose the best next candidate.

# While a narrow distribution means a point is better, a flat distribution means
# a point has more overall probability area above the current best,
# and as such, a larger expected improvement (due to the larger variance).


# In "tidymodels", expected improvement is the default acquisition function.


### 14.2.3 The `tune_bayes()` function ----

# Use `tune_bayes()` to implement iterative search via Bayesian optimization
# with the following arguments:

# - `iter`: The maximum number of search iterations

# - `initial`: An integer or an object returned by `tune_grid()`, or one of the racing functions.
#   An integer specifies the size of a space-filling design that is sampled prior to the first GP model.

# - `objective`: Which acquisition function should be used.
#   The "tune" package contains functions to pass here, like `exp_improve()` or `conf_bound()`.

# - `param_info`: Specifies the range of the parameters as well as any transformations used.
#   Used to define the search space.


# The `control` argument uses the results of `control_bayes()` with the arguments:

# - `no_improve`: Integer that will stop the search if improved parameters are not discovered
#   within `no_improve` iterations.

# - `uncertain`: Integer or `Inf` that will take an *uncertainty sample* if there is no
#   improvement within `uncertain` iterations.
#   Selects the next candidate that has large variation.
#   Does not consider the mean prediction.

# - `verbose`: Logical, will print logging information as the search proceeds.


# We use the SVM results from the previous example as the initial substrate of the
# Gaussian process model.
# The goal is to maximize the area under the ROC curve.
ctrl <- control_bayes(verbose = TRUE)

set.seed(1403)

# Caution: This takes a moment to run.
svm_bo <- svm_wflow |> 
  tune_bayes(
    resamples = cell_folds,
    metrics = roc_res,
    initial = svm_initial,
    param_info = svm_param,
    iter = 25,
    control = ctrl
  )


print(svm_bo)


show_best(svm_bo)


# Plot how the outcome changed over the search:
autoplot(svm_bo, type = "performance") +
  theme_bw()

graphics.off()


## 14.3 Simulated annealing -----

# Simulated annealing (SA) is a general nonlinear search routine inspired by the process 
# in which metal cools.

# It is a global search method that can effectively navigate many different types of
# search landscapes, including discontinuous functions.

# Unlike most gradient-based optimization routines,
# simulated annealing can reassess previous solutions.


### 14.3.1 Simulated annealing search process ----

# Simulated annealing wants to accept fewer suboptimal values as the search proceeds.

# From these two factors, the *acceptance probability3 for a bad result is formalized as:

# Pr[accept suboptimal parameters as iteration i] = exp(c * D_i * i),

# where i is the iteration number, c is a user-specified constant, and
# D_i is the percent difference between the old and the new values,
# where negative values imply worse results.

# The user can adjust the coefficients to find a probability profile that suits his/her needs.

# In `finetune::control_sim_anneal()`, the default for this `cooling_coef` argument is 0.02.


### 14.3.2 The `tune_sim_anneal()` function ----

# To implement iterative search with simulated annealing, use `tune_sim_anneal()`.

# The arguments that define the local neighborhood and cooling schedule are:

# - `no_imporve`: An integer that will stop the search if no global best or improved reults
#    are discovered within `no_imporve` iterations.

# - `restart`: The number of iterations with no new best results before starting from the previous results.

# - `radius`: Numeric vector on (0, 1) defining the minimum and maximum radius of the local
#   neighborhood around the initial point.

# - `flip`: The probability defining the chances of altering the value of categorical or integer parameters.

# - `cooling_coef`: The c coefficient in exp[c * D_i * i] that modulates how quickly the acceptance
#    probability decreases over iterations.
#    Larger values decrease the probability of accepting a suboptimal parameter setting.


# We apply the `tune_sim_anneal()` function to the cell segmentation data:
library(finetune)

ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 10L)


set.seed(1404)

# Caution: This will take a while to run.
svm_sa <- svm_wflow |> 
  tune_sim_anneal(
    resamples = cell_folds,
    metrics = roc_res,
    initial = svm_initial,
    param_info = svm_param,
    iter = 50,
    control = ctrl_sa
  )


# The simulated annealing process discovers a new global optimum after 4 different iterations.

# Plot the progress of the simulated annealing process:
autoplot(svm_sa, type = "performance") +
  theme_bw()

graphics.off()


# Performance versus tuning parameters:
autoplot(svm_sa, type = "parameters") +
  theme_bw()

graphics.off()


# END
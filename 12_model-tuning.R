# Chapter 12 - Model tuning and the dangers of overfitting ----

# To use a model for prediction, the model parameters need to be estimated.

# Some of these model parameters can be estimated directly from the training data.

# But other model parameters, called *tuning parameters* or *hyperparameters*,
# must be specified ahead of time and can't be directly found from the training data.


# These are unknown structural or other kind of values that have a significant impact 
# on the model but cannot be directly estimated from the data.


# We will use this chapter to demonstrate how poor choices of these values lead to 
# overfitting.


### Previous analysis
library(tidymodels)
data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

ames_rec <- recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) |>
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |> 
  step_ns(Latitude, Longitude, deg_free = 20)

lm_model <- linear_reg() |> set_engine("lm")

lm_wflow <- workflow() |> 
  add_model(lm_model) |> 
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)

rf_model <- rand_forest(trees = 1000) |> 
  set_engine("ranger") |> 
  set_mode("regression")

rf_wflow <- workflow() |> 
  add_formula(
    Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
      Latitude + Longitude) |> 
  add_model(rf_model) 

set.seed(1001)
ames_folds <- vfold_cv(ames_train, v = 10)

keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

set.seed(1003)
rf_res <- rf_wflow |> fit_resamples(resamples = ames_folds, control = keep_pred)

## 12.1 Model parameters ----


# In ordinary linear regression, or OLS, there are model parameters b0 and b1 of the model
# y = b0 + b1*x1 + e

# When we have the outcome y nad the predictor x, we can estimate the two parameters
# b0 and b1 directly from the data, using the formula

# b1 = Cov(X,Y)/Var(X)

# b0 = E[Y] - b1*E[X]


# We can directly estimate these values from the data because they are analytically tractable;
# if we have the data, then we can estimate these model parameters.


# For the K-nearest neighbor, or KNN model, the prediction equation for a new value 
# x0 is
# y = \frac{1}{K} \sum_{l=1}^{K} x^{*}_{l}

# where K is the number of neighbors and the x^{*}_{l} are the K closest values to x0
# in the training set.


# The model itself is not defined by a model equation;
# the previous prediction equation instead defines it.

# This characteristic, along with the possible intractability of the distance measure,
# making it impossible to create a set of equations that can be solved for K
# (iteratively or otherwise).

# For small values of K, the boundary is very elaborate while for large values, it might be smooth.


# The number fo nearest neighbors is a good example of a tuning parameter or hyperparameter
# that cannot be directly estimated from the data.


## 12.2 Tuning parameters for different types of models ----

# There are many examples of tuning parameters / hyperparameters in statistical learning models:

# - Boosting is an ensemble method that combines a series of base models, each of which is 
#   created sequentially and depends on the previous model.
#   The number of boosting iterations is an important tuning parameter that requires optimization.

# - In the classic single-layer artificial neural network, a.k.a.
#   the multilayer perceptron,
#   the predictors are combined using two or more hidden units.
#   The hidden units are linear combinations of the predictors that are captured in an
#   *activation function*,
#   typically a nonlinear function such as a sigmoid or ReLu.
#   The hidden units are then connected to the outcome units;
#   one outcome unit is used for regression models,
#   and multiple outcome units are needed for classification.
#   The number of hidden units and the type of activation function are important
#   structural tuning parameters.

# - Modern gradient descent methods are improved by finding the right optimization parameters.
#   Examples are learning rates, momentum, and the number of iterations/epochs.
#   Gradient descent is used to estimate the model parameters in some ensemble models
#   and neural networks.
#   While the tuning parameters associated with gradient descent are not structural parameters,
#   they often require tuning.


# In some cases, preprocessing techniques require tuning:

# - In principal component analysis, or its supervised cousin called 
#   **partial least squares**, the predictors are replaced with new, artificial 
#   features that have better properties related to collinearity.
#   The number of extracted components can be tuned.

# - Imputation methods estimate missing predictor values using the complete values of
#   one or more predictors.
#   One effective imputation tool uses K-nearest neighbors of the complete columns
#   to predict the missing value.
#   The number of neighbors modulates the amount of averaging and can be tuned.


# Some classical statistical models also have structural parameters:

# - In binary regression, the logit link is commonly used, i.e.,
#   logistic regression.
#   Other link functions, such as the probit and complementary log-log, are also available.

# - Non-Bayesian longitudinal and repeated measures models require a specification for the
#   covariance or correlation structure of the data.
#   Options include compound symmetric, a.k.a. exchangeable, autoregressive, Toeplitz, and others.


# A counterexample where it is inappropriate to tune a parameter is the prior distribution
# required for Bayesian analysis.

# The prior distribution encapsulates the analyst's belief about the distribution of a
# quantity before evidence or data are taken into account.


# Another counterexample of a parameter that does *not* need to be tuned is the number
# of trees in a random forest model or in a bagging model.

# This value should instead be chosen to be large enough to ensure numerical stability
# in the results;

# tuning it cannot improve performance as long as the value is large enough to produce reliable results.


# For random forests, this value is typically in the thousands while the number
# of trees needed for bagging is around 50 or 100.


## 12.3 What do we optimize? ----

# For cases where the statistical properties of the tuning parameter are tractable, common
# statistical properties can be used as the objective function.

# In the case of binary logistic regression, the link function can be chosen by
# maximizing the likelihood or information criteria.

# However, these statistical properties may not align with the results achieved 
# using accuracy-oriented properties.


# The reason is that likelihood and error rate measure different aspects of fit quality.


# To demonstrate, consider the classification data with two predictors, A and B,
# and a training set of 539 data points.

# We want to fit a linear class boundary to these data.

# The most common method is the generalized linear model in the form of
# *logistic regression*.

# This model relates the log-odds log(pi/1-pi) of a sample being in Class 1,
# using the logit-transformation:
# log(pi/1-pi) = b0 + b1*x1 + ... + bp*xp

# In the context of generalized linear models, the logit function is the
# *link function* between the outcome pi (the probability of an observation
# being in Class 1) and the predictors x1.


# Another link function is the *probit* model,
# Pi^(-1)(pi) = b0 + b1*x1 + ... + bp*xp

# Where Pi is the cumulative standard normal function.


# A third link function is the
# *complementary log-log model*:
# log(-log(1-pi)) = b0 + b1*x1 + ... + bp*xp


# Each of these three models results in linear class boundaries.

# But which link function should we use?

# Because for these data, the number of model parameters to be estimated is constant,
# we can compute the (log-) likelihood for each model and determine the model with
# the highest value.


# traditionally, the likelihood is computed using the same data that were used
# to estimate the parameters, not using approaches like data splitting or resampling.


# For a data frame `training_set`, we can create a function to compute the
# different models and extract the likelihood statistics for the training set,
# using `broom::tidy()`:
library(tidymodels)
tidymodels_prefer()


llhood <- function(...) {
  logistic_reg() |> 
    set_engine("glm", ...) |> 
    fit(Class ~ ., data = training_set) |> 
    glance() |> 
    select(logLik)
}


bind_rows(
  llhood(),
  llhood(family = binomial(link = "probit")),
  llhood(family = binomial(link = "cloglog"))
  ) |> 
  mutate(
    link = c("logit", "probit", "c-log-log")
  ) |> 
  arrange(desc(logLik))


# According to these results, the logistic model has the best statistical properties.


# From the scale of the log-likelihood values, it is difficult to understand if these
# differences are important or negligible.

# One way of improving this analysis is to resample the statistics and separate the
# modeling data from the data used for performance estimation.

# With this small data set, repeated 10-fold cross-validation is a good choice for
# resampling.


# In the "yardstick" package, the `mn_log_loss()` function estimates the negative
# log-likelihood:
set.seed(1201)
rs <- vfold_cv(training_set, repeates = 10)


# Return the individual resampled performance estimates:
lloss <- function(...) {
  perf_meas <- metric_set(roc_auc, mn_log_loss)
  
  
  logistic_reg() |> 
    set_engine("glm", ...) |> 
    fit_resamples(Class ~ A + B, rs, metrics = perf_meas) |> 
    collect_metrics(summarize = FALSE) |> 
    select(id, id2, .metric, .estimate)
}


resampled_res <- 
  bind_rows(
    lloss() |> mutate(model = "logistic"),
    lloss(family = binomial(link = "probit")) |> mutate(model = "probit"),
    lloss(family = binomial(link = "cloglog")) |> mutate(model = "c-log-log")
  ) |> 
  # Convert log-loss to log-likelihood:
  mutate(
    .estimate = ifelse(.metric == "mn_log_loss", -.estimate, .estimate)
  ) |> 
  group_by(model, .metric) |> 
  summarise(
    mean = mean(.estimate, na.rm = TRUE),
    std_err = sd(.estimate, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )


resampled_res |> 
  filter(.metric == "mn_log_loss") |> 
  ggplot(mapping = aes(x = mean, y = model)) +
  geom_point() +
  geom_errorbar(mapping = aes(xmin = mean - 1.64 * std_err, xmax = mean + 1.64 * std_err), width = 0.1) +
  labs(
    title = "Means and approximate 90% confidence intervals for the resampled \n
    binomial log-likelihood with three different link functions",
    x = "log-likelihood",
    y = NULL
  ) +
  theme_bw()

# Note that the scale of these values is different than the previous values since
# they are computed on a smaller data set;
# the value produced by `broom::glance()` is a sum while
# `yardstick::mn_log_loss()` is an average.


# These results exhibit evidence that the choice of the link function does somewhat matter.

# Although there is an overlap in the confidence intervals, the logistic model has the best results.


# What about a different metric?
# We have calculated the area under the ROC curve for each resample.

# These areas reflect the discriminative ability of the models across all possible
# probability thresholds.

# The ROC values are similar for all four models, and when we plot the class boundaries
# produced by the four models we see that they are close to one another.


## 12.5 Two general strategies for optimization ----

# Tuning parameter optimization falls in two one of two categories: grid search and iterative search.

# - *Grid search* is when we predefine a set of parameter values to evaluate.
#   The main choices are how to produce the grid and how many parameter combinations to evaluate.
#   The number of grid points required to cover the parameter space can become unmanageable
#   with the curse of dimensionality.
#   This is mostly true when the process is not optimized.

# - *Iterative search* or sequential search is when we sequentially discover new parameter
#   combinations based on previous results.
#   In some cases, an initial set of results for one or more parameter combinations is required
#   to start the optimization process.

# Hybrid strategies can work as well.


## 12.6 Tuning parameters in tidymodels ----

# A number of tuning parameters for recipe and model specifications are:

# - The `threshold` for combining neighbors into an "other" category.

# - The number of degrees of freedom in a natural spline, `deg_free`.

# - The amount of regularization in penalized regression models, `penalty`.


# For "parsnip" model specifications, there are two kinds of parameter arguments.

# *Main arguments* are those that are most often optimized for performance
# and are available in multiple engines.

# The main tuning parameters are top-level arguments to the model specification function.

# Consider the `rand_forest()` function with the main arguments
# `trees`, `min_n`, and `mtry`.


# *Engine specific* tuning parameters are either less frequently optimized or
# are engine-specific.

# Consider the "ranger" package that includes gain penalization,
# which regularizes the predictor selection in the tree induction process.

# This parameter helps modulate the trade-off between the number of predictors used
# in the ensemble and performance.

# The name for this argument in `ranger()` is `regularization.factor`.

# Specify a value via a "parsnip" model specification by adding the
# supplemental argument to `set_engine()`:
rand_forest(trees = 2000, min_n = 10) |> # <- main arguments
  set_engine("ranger", regularization_factor(0.5)) # <- engine-specific argument


# Note that the main arguments use a harmonized naming system to remove inconsistencies
# across engines while the engine-specific arguments do not.


# How can we signal to "tidymodels" functions which arguments should be optimized?

# Parameters are marked for tuning by assigning them a value of `tune()`.

# For a single layer neural network (multilayer perceptron),
# the number of hidden units is designated for tuning with:
neural_net_spec <- mlp(hidden_units = tune()) |> 
  set_mode("regression") |> 
  set_engine("keras")

# The `tune()` function does not execute any parameter value;
# it only returns an expression:
tune()
# tune()


# Embedding this `tune()` value in an argument will tag the parameter for optimization.

# The model tuning functions shown in the next two chapters parse the model specification
# and/or recipe to discover the tagged parameters.


# These functions can automatically configure and process these parameters since they
# understand their characteristics like their range of possible values.


# To enumerate the tuning parameters for an object, call
# `extract_parameter_set_dials()`:
extract_parameter_set_dials(neural_net_spec)
# Collection of 1 paramters for tuning
# identifier    type          object
# hidden_units  hidden_units  nparam[+]


# The results show a value of `nparam[+]`, indicating that the number of hidden units
# is a numeric parameter.


# An optional identification argument associates a name with the parameters.

# this is useful when the same kind of parameter is being tuned in different places.

# Using the Ames housing data from Section 10.6, a "recipe" encodes both longitude
# and latitude with spline functions.


# If we want to `tune()` the two spline functions to potentially have different
# levels of smoothness, we call `steps_ns()` twice, once for each predictor.

# To make the parameters identifiable, the identification argument can take any character
# string:
ames_rec <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train
  ) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = tune()) |> # We tune the threshold!
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |> 
  step_ns(Longitude, deg_free = tune("longitude df")) |> # Separate into two calls
  step_ns(Latitude, deg_free = tune("latitude df")) # and add a name


recipes_param <- extract_parameter_set_dials(ames_rec)
print(recipes_param)
# Collection of 3 parameters for tuning

# identifier  type          object
# threshold   threshold     nparam[+]
# longitude   df deg_free   nparam[+]
# latitude    df deg_free   nparam[+]


# Note that the `identifier` and `type` columns are not the same for both of the spline parameters.


# When a recipe and model specification are combined using a workflow, 
# both sets of parameters are shown:
wflow_param <- workflow() |> 
  add_recipe(ames_rec) |> 
  add_model(neural_net_spec) |> 
  extract_parameter_set_dials()

print(wflow_param)
# Collection of 4 parameters for tuning

# identifier      type          object
# hiddeen_units   hidden_units  nparam[+]
# threshold       threshold     nparam[+]
# longitude       df deg_free   nparam[+]
# latitude        df deg_free   nparam[+]


# Note that neural networks are exquisitely capable of emulating nonlinear patterns.
# Adding spline terms to this type of model is unnecessary.


# Each tuning parameter argument has a corresponding function in the "dials" package.
# Usually, the funciton has the same name as the parameter argument:
hidden_units()
# Hidden Units (quantitative)
# Range: [1, 10]

threshold()
# Threshold (quantitative)
# Range: [0, 1]


# The `deg_free` parameter is a counterexample;
# the notion of degrees of freedom comes up in a variety of contexts.

# When used with splines, there is a "dials" function called `spline_degree()`:
spline_degree()
# Spline Degrees of Freedom (quantitative)
# Range: [1, 10]


# The "dials" package also has a convenience function to extract a particular parameter object:

# identify the parameter with the id value:
wflow_param |> 
  extract_parameter_dials("threshold")
# Threshold (quantitative)
# Range: [0, 0.1]


# Inside the parameter set, the range of the parameters can be updated in place:
extract_parameter_set_dials(ames_rec) |> 
  update(threshold = threshold(c(0.8, 1.0)))


# The *parameter sets* created by `extract_parameter_set_dials()` are consumed
# by the tidymodels tuning functions when needed.


# Sometimes the parameter range is critical and cannot be assumed with a default value.

# The number of predictor columns that are randomly sampled for each split in 
# a tree, `mtry()` are the primary tuning parameter in a random forest model.

# Without knowing the number of predictors in the data set,
# this parameter range cannot be assumed:
rf_spec <- rand_forest(mtry = tune()) |> 
  set_engine("ranger", regularization.factor = tune("regularization")) |> 
  set_mode("regression")


rf_param <- extract_parameter_set_dials(rf_spec)
print(rf_param)
# Collection of 2 parameters for tuning

# identifier      type                    object
# mtry            mtry                    nparam[?]
# regularization  regularization.factor   nparam[+]


# Model parameters needing finalization:
#   # Randomly Selected Predictors ('mtry')

# See `?dials::finalize` or `?dials::update.parameters` for more information.


# Note that a complete parameter object would have a `[+]` in their summary.

# The value of `[?]` in the `mtry` parameter indicates that at least one end of its
# range is missing.


# You can use `update()` to add a range:
rf_param |> 
  update(mtry = mtry(c(1, 70)))
# Collection of 2 parameters for tuning

# identifier      type                    object
# mtry            mtry                    nparam[+]
# regularization  regularization.factor   nparam[+]


# But this approach will not work if your workflow uses steps that add or subtract
# columns, lke PCA or best subset selection or LASSO

# Use `finalize()` to execute the recipe once to obtain the dimensions:
pca_rec <- recipe(Sale_Price ~ ., data = ames_train) |> 
  # Select the square-footage predictors and extract their PCA components:
  step_normalize(contains("SF")) |> 
  # Select the number of components needed to capture 95% of the variance in the predictors:
  step_pca(contains("SF"), threshold = 0.95)


updated_param <- workflow() |> 
  add_model(rf_spec) |> 
  add_recipe(pca_rec) |> 
  extract_parameter_set_dials() |> 
  finalize(ames_train)


print(updated_param)
# Collection of 2 parameters for tuning

# identifier      type                    object
# mtry            mtry                    nparam[+]
# regularization  regularization.factor   nparam[+]


updated_param |> 
  extract_parameter_dials("mtry")
# Randomly Selected Predictors (quantitative)
# Range: [1, 74]


# When the recipe is prepared, the `finalize()` function learns to set the upper range
# of `mtry` to 74 predictors.


# Additioanlly, the results of `extract_parameter_set_dials()` includes
# engine-specific parameters (if any).

# The "dials" package contains parameter functions for all potentially tunable
# engine-specific parameters:
print(rf_param)


regularization_factor()
# Gain Penalization (quantitative)
# Range: [0, 1]


# Some tuning parameters are associated with transformations.

# The penalty parameter for regularized regression is an example.

# It is nonnegative and is commonly stated in log units.

# The primary "dials" parameter object indicates that a transformation is used
# by default:
penalty()
# Amount of Regularization (quantitative)
# Transformer: log-10 [1e-100, Inf]
# Range (transformed scale): [-10, 0]


# This is important to know when altering the range.
# New range values must be in the transformed units:

# correct method to have penalty values between 0.1 and 1.0:
penalty(c(-1, 0)) |> 
  value_sample(1000) |> 
  summary()
# Min: 0.1001
# Max: 0.9991


# incorrect:
penalty(c(0.1, 1.0)) |> 
  value_sample(1000) |> 
  summary()
# Min: 1.261
# Max: 9.972


# The scale can be changed with the `trans` argument:
penalty(trans = NULL, range = 10^c(-10, 0))
# Range: [1e-10, 1]


# END
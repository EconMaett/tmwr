# Chapter 06 - Fitting models with parsnip ----

# The "parsnip" package includes the `parsnip::fit()` and
# `parsnip::predict()` functions.

# They can be applied to any object of the class "parsnip".


## 6.1 Create a model ----

# Suppose you want to fit a lienar model to the Ames housing data.

# There are two distinct methods:

# - Ordinary linear regression uses least squares to solve for the model parameters.

# - Regularized linear regression adds a penalty to the least squares method to encourage
#   simplicity by shrinking and/or eliminating some of the coefficients.
#   The parameters can be estimated using Bayesian methods or cross-validation.


# In R, the "stats" package that is included with base R provides the `lm()` function
# for linear regression.
# The basic syntax is:
# `model <- lm(formula, data, ...)`

# The ellipsis (...) is used to pass additional arguments to the function.

# The formula is a symbolic representation of the model.


# To estimate with regularization techniques, a Bayesian model can be fit
# using the "rstanarm" package:
# `model <- rstanarm::stan_glm(formula, data, family = "gaussian", ...)`

# In this case, the other options passed via `...` would include arguments
# for the prior distribution as well as specifics about the numberical aspects
# of the model.

# As with `stats::lm()`, only the `formula` interface is available.


# A popular non-Bayesian estimation approach for regularized regression comes
# with the "glmnet" package by Friedman, Hastie, and Tibshirani (2010):
# `model <- glmnet::glmnet(x = matrix, y = vector, family = "gaussian", ...)`

# In this case, the predictor data must already be formatted into a numeric matrix.

# There is only an `x/y` method and no `formula` method.


# For the "tidymodels" framework, the model specification follows a unified approach:

# 1. Specify the type of model based on its mathematical structure.

# 2. Specify the engine of the model.
#    This reflects the software to be used, like Stan or glmnet.
#    These are models in their own right, and "parsnip" provides consistent
#    interfaces to them.

# 3. When required, declare the mode of the model.
#    The mode reflects the type of prediciton outcome.
#    For numerical outcomes, the mode is "regression".
#    For qualitative outcomes, the mode is "classification".

library(tidymodels)
tidymodels_prefer()

linear_reg() |> set_engine("lm")
# Lienar Regression Model Specification (regression)

# Computational engine: lm


linear_reg() |> set_engine("glmnet")
# Linear Regression Model Specification (regression)

# Computational engine: glmnet


linear_reg() |> set_engine("stan")
# Linear Regression Model Specification (regression)

# Computational engine: stan


# Once the details of the model are specified, the estimation
# can be done with either `fit()` or `fit_xy()`.

# The "parsnip" package allows the user to choose between the 
# `formula` or the `x/y` interface.


# The `translate()` function provides details on how "parsnip" converts
# the user's code to the underlying package's syntax:
linear_reg() |> set_engine("lm") |> translate()
# Lienar Regression Model Specification (regression)

# Computational engine: lm

# Model fit template:
# stats::lm(formula = missing_arg(), data = missing_arg(), weights = missing_arg())


linear_reg(penalty = 1) |> set_engine("glmnet") |> translate()
# Linear Regression Model Specification (regression)

# Main Arguments:
#   penalty = 1

# Computational engine: glmnet

# Model fit template:
# glmnet::glmnet(x = missing_arg(), y = missing_arg(), weights = missing_arg(), family = "gaussian")


linear_reg() |> set_engine("stan") |> translate()
# Linear Regression Model Specification (regression)

# Computational engine: stan

# Model fit template:
# rstanarm::stan_glm(formual = missing_arg(), data = missing_arg(), weights = missing_arg(), family = stats::gaussian, refresh = 0)


# `missing_arg()` is just a placeholder for data that has yet to be provided.


# Lets predict the sale price of houses in the Ames data as a function
# of only longitude and latitude.
library(tidymodels)
data(ames)
ames <- ames |> 
  mutate(
    Sale_Price = log10(Sale_Price)
  )

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)


lm_model <- linear_reg() |> 
  set_engine("lm")

lm_form_fit <- lm_model |> 
  fit(formula = Sale_Price ~ Longitude + Latitude, data = ames_train)

lm_xy_fit <- lm_model |> 
  fit_xy(
    x = ames_train |> select(Longitude, Latitude),
    y = ames_train |> pull(Sale_Price)
  )

lm_form_fit

lm_xy_fit

# "parsnip" enables a consistent model interface for models from different
# packages.

# Random forest model functions typically need three arguments:

# - The number of trees in the ensemble
# - The number of predictors to randomly sample with each split within a tree
# - The number of data points required to make a split

# For three different R packages implementing the same model, the arguments are:

# Argument Type         ranger          randomForest  sparklyr
# sampled predictors    mtry            mtry          feature_subset_strategy
# trees                 num.trees       ntree         num_trees
# data points to split  min.node.size   nodesize      min_instances per node


# In an effort to make argument specification less painful, "parsnip" uses common
# argument names withina nd between packages.

# For random foreests, "parsnip" models use:

# Argument type         parsnip
# sampled predictors    mtry
# trees                 trees
# data points to split  min_n


# To specify the amount of regularization to use in a "glmnet" model,
# `lambda` is used.

# "parsnip" simply uses the term `penalty` instead.

# The number of neighbors in a KNN model is called `neighbors` instead of `k`.

# While `lambda` and `k` are used in the statistics literature,
# the more verbose names are used in "parsnip".

# To understand how "parsnip" argument names map to the original names,
# see
?rand_forest


# The `translate()` function also shows how "parsnip" converts the user's
# code to the underlying package's syntax:
rand_forest(trees = 1000, min_n = 5) |> 
  set_engine("ranger") |> 
  set_mode("regression") |> 
  translate()

# Random Forest Model Specification (regression)

# Main Arguments:
#  trees = 1000
# mtry = 0.5

# Computational engine: ranger

# Model fit template:
# ranger::ranger(x = missing_arg(), y = missing_arg(), weights = missing_arg(),
#   num.trees = 1000, min.node.size = min_rows(~5, x), num.threads = 1,
#   verbose = FALSE, seed = sample.int(10^5, 1))


# Modeling functions in "parsnip" separate model arguments into two categories:

# - *Main arguments* are commonly used and tend to be available across engines.

# - *Engine arguments* are specific to a particular engine and used more rarely.


# In the translation of the previous random forest model, the arguments
# `num.threads`, `verbose`, and `seed` were added by default.

# These arguments are specific to the "ranger" implementation of random forest models.

# Engine-specific arguments can be specified in `set_engine()`.

# To have the `ranger::ranger()` function print out more information,
# set `verbose = TRUE`:
rand_forest(trees = 1000, min_n = 5) |> 
  set_engine("ranger", verbose = TRUE) |> 
  set_mode("regression")
# Random Forest Model Specification (regression)

# Main Arguments:
#  trees = 1000
#  min_n = 5

# Engine-Specific Arguments:
#   verbose = TRUE

# Computational engine: ranger


## 6.2 Use the model results ----

# Once a model is specified and fit to the data, we may want to
# extract information from the model.

# A "parsnip" model object stores several quantities.

# The fitted model can be found in the `$fit` slot
# that can be accessed with `extract_fit_engine()`
lm_form_fit$fit

lm_form_fit |> extract_fit_engine()


# Normal methods can be applied to this object:
lm_form_fit |> extract_fit_engine() |> vcov()


# Note that it is discouraged to directly pass the `$fit` slot to
# `predict()`.

# Call `predict(lm_form_fit)`, NOT `predict(lm_form_fit$fit)`.


# Some existing methods in base R store the model results in a manner
# that is not useful.

# The `summary()` method for `"lm"` objects can be used to print the results
# of the model fit, including a table with parameter values, their uncertainty estimates,
# and p-values.

# These particularities can also be saved:
model_res <- lm_form_fit |> 
  extract_fit_engine() |> 
  summary()


# The model coefficient table is accessible via the `coef()` method:
coef(model_res)
coefficients(model_res)

param_est <- coef(model_res)
class(param_est)
# "matrix" "array"

print(param_est)

# The object `param_est` is a numeric matrix.
# The choice to use a matrix was made back in the late 1970s when computational
# efficiency was extremely critical.

# The non-numeric data are contained in the row names.
# Keeping the parameter labels as row names is a convention in the original S language.


# Use the "broom" package to convert many types of model objects to a tidy structure:
broom::tidy(lm_form_fit)

# The column names are standardized across models and do not contain any additional data.

# The data previously contained in the row names are now in a column called `term`..

# An important principle in the tidymodels ecosystem is that functions should
# return values that are
# - predictable
# - consistent
# - unsurprising


## 6.3 Make predictions ----

# For predictions, "parsnip" conforms to the following rules:

# 1. The results are always a tibble.
# 2. The column names of the tibble are always predictable.
# 3. There are always as many rows in the tibble as there are in the input data set.

ames_test_small <- ames_test |> 
  slice(1:5)

predict(object = lm_form_fit, new_data = ames_test_small)

# The row order of the predictions is always the same as the original data.

# Note that the column name contains a leading dot `.` and is called
# `.pred`. This is to avoid confusion should the data contain a column
# called `pred`.

# The three rules make it easier to merge predictions with the original data:
ames_test_small |> 
  select(Sale_Price) |> 
  bind_cols(predict(object = lm_form_fit, new_data = ames_test_small)) |> 
  bind_cols(predict(object = lm_form_fit, new_data = ames_test_small, type = "pred_int"))

# Note that in the last call to `predict()`, the `type` argument was used to
# to add 95% prediction intervals to the output.


# The motivation for the first rule, that tidymodels functions should return
# *predictable* output, stems from the fact that some R packages produce
# dissimilar data types from prediction functions.

# The "ranger" package does not return a data frame or a vector as output,
# but special object with multiple values embedded within, including the 
# predicted values.

# The native "glmnet" package can return at least four different output types
# for predictions, depending on the model specifics and the characteristics of the data.

# Different return values for "glmnet" prediction types:

# Type of Prediction        Returns a:
# numeric                   numeric matrix
# class                     character matrix
# probability (2 classes)   numeric matrix (2nd level only)


# Additionally, the column names of the results contain coded values that map
# to a vector called `lambda` within the `glmnet` model object.


# The second tidymodels prediciton rule, that column names should be consistent
# across models, is also violated by the "glmnet" package.

# type value  column name(s)
# numeric     .pred
# class       .pred_class
# prob        .pred_{class levels}
# conf_int    .pred_lower, .pred_upper
# pred_int    .pred_lower, .pred_upper


# The third rule, that the number of rows in the output should match the number
# of rows in the input, is critical.

# If any rows of the new data contain missing values, the output will be
# padded with missing results for those rows.


# Supposed we used a decision tree to model the Ames housing data.
# Ouside of the model specification, there are no significant differences
# in the code pipeline:
tree_model <- decision_tree(min_n = 2) |> 
  set_engine("rpart") |> 
  set_mode("regression")

tree_fit <- tree_model |> 
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

ames_test_small |> 
  select(Sale_Price) |> 
  bind_cols(predict(object = tree_fit, new_data = ames_test_small))


## 6.4 Parsnip-extension packages ----

# The "parsnip" package contains interfaces to a number of modeling packages.

# The "discrim" package has model definitions for the set of classification analysis
# methods such as linear and quadratic discriminant analysis.


# A list of all the models that can be used with "parsnip" can be found at
# https://www.tidymodels.org/find/



## 6.5 Creating model specifications -----

# The "parsnip" package contains an RStudio addin that can be chosen from
# the "Addins" toolbar menu or by calling
parsnip_addin()

# This opens a window in the Viewer panel of the RStudio IDE
# with a list of possible models for each mode.

# These can be written to the source panel code.

# The model list includes models from "parsnip"
# or "parsnip"-extension packages that are on CRAN.

# END
# Chapter 07 - A model workflow ----

## 7.2 Workflow basics ----

library(tidymodels) # Includes the "workflows" package
tidymodels_prefer()

data(ames)

ames <- ames |> 
  mutate(
    Sale_Price = log10(Sale_Price)
    )

set.seed(502)
ames_split <- initial_split(data = ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

lm_model <- linear_reg() |> 
  set_engine("lm")


# A workflow always requires a `"parsnip"` model object:
lm_wflow <- workflow() |> 
  add_model(lm_model)

lm_wflow
# == Workflow ====================================================================
# Preprocessor: None
# Model: linear_reg()

# --- Model ---------------------------------------------------------------------
# Lienar Regression Model Specification (regression)

# Computational engine: lm


# Since we have not yet specified how this workflow should preprocess the data,
# we have `Preprocessor: None` in the console output.


# If our model is very simple, a standard R formula can be used as a preprocessor:
lm_wflow <- lm_wflow |> 
  add_formula(Sale_Price ~ Longitude + Latitude)

lm_wflow

# Now we see
# --- Preprocessor --------------------------------------------------------------
# Sale_Price ~ Longitude + Latitude

# in the console output


# Workflows have a `fit()` method that can be used to create the model:
lm_fit <- fit(lm_wflow, ames_train)
lm_fit

# We can also `predict()` the fitted workflow:
predict(lm_fit, ames_test |> slice(1:3))


# Bot the model and preprocessor can be removed or updated:
lm_fit |> 
  update_formula(Sale_Price ~ Longitude)

# Note that in this new object, the output shows that the previous
# fitted model was removed since the new formula is inconsistent with the 
# previous model fit.


## 7.3 Adding raw variables to the `workflow()` ----

# There is another interface for passing data to the model, the 
# `add_variables()` function that uses "dplyr"-like syntax.

# The function's two primary arguments are `outcomes` and `predictors`.

# These arguments use a selection approach that is similar to the
# "tidyselect" backend used in "tidyverse" packages to capture
# multiple selectors using `c()`.
lm_wflow <- lm_wflow |> 
  remove_formula() |> 
  add_variables(outcomes = Sale_Price, predictors = c(Longitude, Latitude))

print(lm_wflow)


# The predictors could also have been specified as
# `predictors = c(ends_with("tude"))`

# Any outcome columns accidentally specified in the `predictors`
# argument will be quietly removed, so
# `predictors = everything()` is a viable choice.


# When the model is fit, the specification assembles these data, unaltered,
# into a data frame and passes it to the underlying function:
fit(lm_wflow, ames_train)


## 7.4 How does a `workflow()` use the formula? ----

# R possesses a formula method that lets you encode the original data into
# an analysis-ready format. This can involve executing inline transformations
# such as `log(x)`, `log10(x)`, or `poly(x)`.
# You can also crate dummy variables from `factor()` variables,
# and interaction terms like `x1 * x2`.

# However, some statistical methods require different types of encoding:

# - Tree-based models use the formula interface but *do not* encode
#   the categorical predictors as dummy variables.

# - Some models require that the outcome be encoded as a factor.
#   In survival analysis models, a formula term such `strata(site)`
#   would indicate that the column `site` is a stratification variable.
#   This means it should not be treated as a regular predictor and does not
#   have a corresponding location parameter estimate.

# - Some packages extend the formula in ways that base R functions cannot parse.
#   In multilevel models, such as mixed models or hierarchical Bayesian models,
#   a model term such as `(week | subject)` indicates that the column `week` is a
#   random effect that has a different slope parameter for each value of `subject`.


### Tree-based models ----

# When we fit a tree to the data, the "parsnip" package understands what the
# modeling function would do.

# For example, if a random forest model is fit using the "ranger" or
# "randomForest" pacakges, the workflow knows predictor columns that are
# factors should be left as is.

# As a counterexample, a boosted tree created with the "xgboost" package
# requires the user to create dummy variables from factor predictors,
# since `xgboost::xgb.train()` will not.

# This requirement is embedded into the model specification object and
# a workflow using "xgboost" will create the indicator columns for this engine.

# Note that a different engine for boosted trees, "C5.0" does not require
# dummy variables so none are created by the workflow.

# This determination is made for each model and engine combination.


### 7.4.1 Special formulas and inline functions ----

# A number of multilevel models have standardized on a formula specification
# devidsed in the "lme4" package.

# To fit a regression model that has random effects for subjects, we would use
# the formula:
library(lme4)
data(Orthodont, package = "nlme")

lmer(distance ~ Sex + (age | Subject), data = Orthodont)

# The effect of this is that each subject will have an estimated intercept and
# slope parameter for `age`.

# The problem is that standard R methods can't properly process this formula:
stats::model.matrix(distance ~ Sex + (age | Subject), data = Orthodont)

# The result is a zero row data frame.


# Even if this formula could be used with `stats::model.matrix()`, it would
# still present a problem since the formula also specifies the statistical attributes 
# of the model.


# The solution in "workflows" is an optional supplementary model formula that can be
# passed to `add_model()`.

# The `add_variables()` specification provides the bare column names,
# and then the actual formula given to the model is set with `add_model()`:
library(multilevelmod)

multilevel_spec <- linear_reg() |> 
  set_engine("lmer")

multilevel_workflow <- workflow() |> 
  add_variables(outcomes = distance, predictors = c(Sex, age, Subject)) |> 
  add_model(multilevel_spec, formula = distance ~ Sex + (age | Subject))

multilevel_fit <- fit(multilevel_workflow, data = Orthodont)
print(multilevel_fit)


# We can use the previously mentioned `strata()` function from the
# "survival" package for survival analysis:
library(censored)

parametric_spec <- survival_reg()

parametric_workflow <- workflow() |> 
  add_variables(outcomes = c(fustat, futime), predictors = c(age, rx)) |> 
  add_model(parametric_spec, formula = Surv(futime, fustat) ~ age + strata(rx))

parametric_fit <- fit(parametric_workflow, data = ovarian)
print(parametric_fit)

# Notice how in both of these calls the model-specific formula is used.


## 7.5 Creating multiple workflows at once ----

# Sometimes the data require numerous attempts until you find an appropriate model.

# - For predictive models, you can try a variety of models.

# - For sequential testing of hypotheses, you may start with an expanded set of predictors.
#   This "full model" is compared to a sequence of the same model that removes
#   a predictor in each turn.

# The "workflows" package creates combinations of workflow components.
# A list of preprocessors, e.g. formulas, "dplyr" selectors, or
# feature engineering recipe objects can be combined with a list of
# model specifications, resulting in a set of workflows.


# We can create a set of formulas with different predictors for the Ames housing data:
location <- list(
  longitude = Sale_Price ~ Longitude,
  latitude = Sale_Price ~ Latitude,
  coords = Sale_Price ~ Longitude + Latitude,
  neighborhood = Sale_Price ~ Neighborhood
)

# These representations can be corssed with one or more models using
# the `workflow_set()` function.
library(workflowsets)

location_models <- workflow_set(preproc = location, models = list(lm = lm_model))
print(location_models)

print(location_models$info[[1]])

extract_workflow(location_models, id = "coords_lm")

# Workflow sets are mostly designed to work with resampling.
# The columns `option` and `result` must be populated with specific types
# of objects that result from resampling.

# We can create model fits for each formula and save them in a new
# column called `fit`:
location_models <- location_models |> 
  mutate(
    fit = map(.x = info, .f = ~ fit(.x$workflow[[1]], ames_train))
  )

print(location_models)

location_models$fit[[1]]


## 7.6 Evaluating the test ----

# Assuming that we have concluded our model development and settled on a
# final model, there exists a convenience function called `last_fit()`
# that will *fit* the model to the entire training set and *evaluate*
# it with the testing set.

final_lm_res <- last_fit(lm_wflow, ames_split)
print(final_lm_res)


# The `.workflow` column contains the fitted workflow and can be pulled out of the 
# results using:
fitted_lm_wflow <- extract_workflow(final_lm_res)

# Similarly, `collect_metrics()` and `collect_predictions()` provide
# access to the performance metrixs and predictions, respectively.
collect_metrics(final_lm_res)

collect_predictions(final_lm_res) |> 
  slice(1:5)

# END
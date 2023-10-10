# Chapter 08 - Feature engineering with recipes ----

library(tidymodels)
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

lm_wflow <- workflow() |> 
  add_model(lm_model) |> 
  add_variables(outcomes = Sale_Price, predictors = c(Longitude, Latitude))

lm_fit <- fit(lm_wflow, ames_train)


# Feature engineering entails reformatting predictors to make them easier for
# a model to use.

# This includes transformations and encoding of the data to best represent
# their important characteristics.

# Imagine you have two predictors in a data set that can be more effectively
# represented in your model as a ratio; 
# creating a new predictor from the ratio of the original is an example of feature engineering.


# Take the location of a house in Ames. 
# There are several ways that such spatial information can be exposed to a model,
# including neighborhood (a qualitative measure), longitude/latitude,
# distance to the nearest school or to Iowa State University, and so on.

# When choosing which predictors to use in a model, we might choose an option
# we believe is most associated with the outcome.


# Other examples of preprocessing to build better features include:

# - Correlation between predictors can be reduced via feature extraction or
#   the removal of some predictors.

# - When some predictor have missing values, they can be imputed with a
#   sub-model.

# - Models that use variance-type measures may benefit from coercing the distribution
#   of some skewed predictors to be symmetric by estimating a transformation.


# Note that some models use geometric distance metrics, and consequently, 
# numeric predictors need to be centered and scaled so that they all have
# the same units (normalized).

# Otherwise, the distance values are biased by the scale of each column.


# The "recipes" package provides a framework for feature engineering.


## 8.1 A simple `recipe()` for the Ames housing data ----

# We focus on a subset of the predictors available in the Ames housing data:

# - The neighborhood (qualitative, with 29 neighborhoods in the training set)

# - The gross above-grade living area (continuous, `Gr_Liv_Area`)

# - The year built (`Year_Built`)

# - The type of building (`Bldg_Type` with values `OneFam`, `TwoFmCon`, `Duplex`, "Twnhs, and `TwnhsE`)


# Suppose the initial ordinary linear regression model were fit to these data.

# A standard call to `stats::lm()` is:
lm(formula = Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + Year_Built + Bldg_Type, data = ames)

# When this function is executed, the data are converted from a data frame to 
# a numeric *design matrix*, also called *model matrix*, with a call
# to `stats::model.matrix()`.

# Then the least squares method is used to estimate the parameters.

# What the formula does can be decomposed in a series of three steps:

# 1. Sale price is defined as the outcome, while neighborhood, gross living area,
#    the year built, and the building type variables are defined as predictors.

# 2. A log transformation is applied to the gross living area predictor.

# 3. The neighborhood and building type columns are converted from a 
#    non-numeric format to a numeric format, since least squares requires numeric predictors.


# A "recipe" object defines a series of steps for data processing.

# Unlike the formula method inside a modeling function, the "recipe" defines
# the steps via the `step_*()` family of functions without immediately executing them;
# it is only a specification of what should be done.

# Here is a "recipe" equivalent to the previous formula call:
library(tidymodels)
tidymodels_prefer()

simple_ames <- recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type, data = ames_train) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_dummy(all_nominal_predictors())

print(simple_ames)


# Let's break this down:

# 1. The call to `recipe()` with a formula tells the "recipe" the *roles*
#    of the "ingredients" or variables (e.g. predictor, outcome).
#    It only uses the data `ames_train` to determine the data types for the columns.

# 2. `step_log()` declares that `Gr_Liv_Area` should be log transformed.

# 3. `step_dummy()` specifies which variables should be converted from a qualitative
#     to a quantitative format, in this case, using dummy or indicator variables.


# The function `all_nominal_predictors()` captures the names of any predictor columns
# that are currently factor or character columns.

# This is a "dplyr"-like selector function similar to `starts_with()` or `matches()`
# but can be used inside a "recipe".

# Other selectors specific to the "recipes" package are:
# - `all_numeric_predictors()`
# - `all_numeric()`
# - `all_predictors()`
# - `all_outcomes()`

# As with "dplyr", one or more unquoted expressions, separated by commas,
# can be used to select which columns are affected by each step.


# The advantage to using a "recipe" over the formula or raw predictors includes:

# - These computations can be recycled across models since they are not tightly coupled
#   to the modeling function.

# - A "recipe" enables a broader set of data processing choices than formulas offer.

# - The syntax is very compact. `all_nominal_predictors()` captures many variables
#   while a formula would require each variable to be explicitly listed.

# - All data processing is captured in a single R object instead of in scripts that
#   are repeated, or even spread across different files.


## 8.2 Using recipes ----

# Preprocessing choices and feature engineering should be considered part of a 
# modeling workflow, not a separate task.

# The "workflows" package contains high level functions to handle different types
# of preprocessors.

# We attach the `simple_ames` "recipe" to the workflow:
lm_wflow |> 
  add_recipe(simple_ames)
# Error in `add_recipe()`:
# ! A recipe cannot be added when variables already exist.

# This does not work!

# We can have only one preprocessing method at a time, so we need to remove
# the existing preprocessor before adding the "recipe":
lm_wflow <- lm_wflow |> 
  remove_variables() |> 
  add_recipe(simple_ames)

print(lm_wflow)


# Let's estimate both the recipe and model using a simple call to `fit()`:
lm_fit <- fit(lm_wflow, ames_train)

# The `predict()` method applies the same preprocessing that was used on the training
# set to the new data before passing them along to the model's `predict()` emthod:
predict(lm_fit, ames_test |> slice(1:3))


# If we need the bar model object or recipe, there is an `extract_*()` family
# of functions to retrieve them:

# Get the recipe after it has been estimated:
lm_fit |> 
  extract_recipe(estimated = TRUE)

# To tidy the model fit use `broom::tidy()`:
lm_fit |> 
  extract_fit_parsnip() |> 
  broom::tidy() |> 
  slice(1:5)


## 8.3 How data are used by the `recipe()` ----

# Data are passed to recipes at different stages.

# First, when calling `recipe(..., data)`, the data set is used to determine
# the data types of each column,
# so that selectors such as `all_numeric()` or
# `all_numeric_predictors()` can be used.


# Second, when preparing the data using `fit(workflow, data)`, the
# training data are used for all estimation operations including a recipe that
# may be part of the `workflow`, from determining factor levels to computing
# PCA components and everything in between.

# Note that all preprocessing and feature engineering steps may
# *only* use the training data!
# Otherwise, information leakage will negatively impact the model's performance
# when applied to new data.


# Finally, when using `predict(workflow, new_data)`, no model or preprocessor paarameters
# like those from recipes are re-estimated using the values in `new_data`.

# Take centering and scaling using `step_normalize()` as an example.

# Using this step, the means and standard deviations from the appropriate columsn
# are determined from the training set;
# new samples at prediction time are standardized using these values
# from training when `predict()` was invoked.


## 8.4 Examples of recipe steps ----

# WE take an extended tour of the capabilities of the "recipes" package
# and explore the most important `step_*()` functions.

# These recipe step functions each specify a specific possible step in a
# feature engineering process.


### 8.4.1 Encoding qualitative data in numeric format ----

# Transforming a nominal or qualitative variable such as a factor or a character variable
# into numeric format is a common task.

# `step_unknown()` can be used to change missing values to a dedicated factor level.

# `step_novel()` can allot a new level for this purpose.

# `step_other()` can be used to analyze the frequencies of the factor levels in the training set
# and convert infrequently occurring values to a catch-all level called "other",
# with a threshold that can be specified.

# Take the `Neighborhood` variable:
ggplot(data = ames_train, mapping = aes(y = Neighborhood)) +
  geom_bar() +
  labs(y = NULL) +
  theme_bw()

graphics.off()
# We see that two neighborhoods have less than five properties in the training
# data, `Landmark` and `Green Hills`; in this case, no houses at the `Landmark`
# neighborhoods were included in the testing set.

# For some models, it may be problematic to have dummy variables with a single
# nonzero entry in the column.

# At a minimum, it is highly improbably that these features would be important to a model.

# If we add `step_other(Neighborhood, threshold = 0.01)` to the recipe,
# the bottom 1% of the neighborhoods will be lumped into a new level called "other".
# In this training set, this will catch seven neighborhoods.

simple_ames <- recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type, data = ames_train) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors())


# Many, but not all, underlying model calculations require predictor values to
# be encoded as numbers.
# Notable exceptions include:
# - tree-based models
# - rule-based models
# - naive Bayes models


# The most common method to convert a factor predictor to a numeric format
# is to create a dummy or indicator variable.

# The building type predictor in the Ames housing data is an example.
levels(ames_train$Bldg_Type)
# OneFam, TwoFmCon, Duplex, Twnhs, TwnhsE

length(levels(ames_train$Bldg_Type))
# 5

# For dummy variables, the factor variable with five levels would be replaced
# with four numeric columns whose values are either 0 or 1.

# In R, the convention is to exclude a column for the first factor level,
# here `OneFam`, which serves as the base level.

# You need not include a dummy variable for the first levels as this
# would introduce perfect multicolinearity to the model which would
# render estimation with OLS impossible.


# The full set of encodings can be used for some models.
# This is called the one-hot encoding and can be achieved by using the
# `one_hot` argument in `step_dummy()`.


# The `step_dummy()` function gives you more control over how the resulting
# dummy variables are named.

# In base R, dummy variable names mash the variable name with the level,
# resulting in names like 
# `NeighborhoodVeenker`.

# Recipes, by default, use an underscore as the separator between the name and level,
# e.g. `Neighborhood_Veenker`.


# The default naming convention in "recipes" makes it easier to capture
# those new columns in future steps using a selector such as
# `starts_with("Neighborhood_")`.


# Traditional dummy variables require that all of the possible categories be known
# to create a full set of numeric features.

# There are other methods for doing this transformation to a numeric format:


# - feature hashing: only consider the value of the category to
#   assign it to a predefined pool of dummy variables

# - effect/likelihood encodings: replace the original data with a single numeric
#   column that measures the effect of those data.


### 8.4.2 Interaction terms ----

# We might find that the regression slopes for the gross living area differ for
# different building types.
ggplot(data = ames_train, mapping = aes(x = Gr_Liv_Area, y = 10^Sale_Price)) +
  geom_point(alpha = 0.2) +
  facet_wrap(~ Bldg_Type) +
  geom_smooth(method = lm, formula = y ~ x, se = FALSE, color = "lightblue") +
  scale_x_log10() +
  scale_y_log10() +
  labs(
    x = "Gross Living Area", y = "Sale Price (USD)"
  ) +
  theme_bw()

graphics.off()


# A base R formula would take an interaction term with a colon `:` between the:
# `Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + Bldg_Type + lg10(Gr_Liv_Area):Bldg_Type`
# or
# `Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) * Bldg_Type`

# Where the multiplication sign `*` expands those columns to the main effects 
# and interaction term.


# Recipes are more explicit and sequential and give the user more control.

# With the current recipe, `step_dummy()` has already created dummy variables

# To include an interaction, we use `step_interact(~ interaction_terms)`

# where the terms on the right-hand side of the tilde `~` are the interactions.

simple_ames <- recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type, data = ames_train) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_"))


# Additional interactions can be specified in this formula by separating them
# with a plus sign `+`.

# Note that the recipe will only use interactions between different variables.
# If the formula uses `var_1:var_1`, this is ignored.


# Suppose in a recipe we had not yet made dummy variables for building types.
# It would be inappropriate to include a factor column in this step, like
# `step_interact(~ Gr_Liv_Area:Bldg_Type)`

# This is telling the underlying base R code used by `step_interact()` to
# make dummy variables and then form the interactions.

# If this occurs, a warning is issued that unexpected results may be generated.


# As with naming dummy variables, "recipes" provides more coherent names for interaction terms.

# The interaction here is named `Gr_Liv_Area_x_Bldg_type_Duplex` instead of
# `Gr_Liv_Area:Bld_TypeDuplex`, which is not a valid column name for a data frame.


### 8.4.3 Spline functions ----

# When a predictor has a nonlinear relationship with the outcome, 
# some predictive models can adaptively approximate this relationship during training.

# But since simpler is usually better (Occam's Razor), you could include
# *spline* functions to represent the data.

# Splines replace the existing numeric predictor with a set of columns
# that emulate a flexible, nonlinear relationship.

# As more spline terms are added to the data, the capacity
# to nonlinearly represent the relationship increases.

# However, this also increases the likelihood of picking up on data trends that
# only occur by chance in the training data (Overfitting).


# If you have used `geom_smooth()` within `ggplot2`, you have already used
# a spline representation of the data.

# Let's create a different number of smooth splines for the latitude predictor:
library(patchwork)
library(splines)

plot_smoother <- function(deg_free) {
  ggplot(data = ames_train, mapping = aes(x = Latitude, y = 10^Sale_Price)) +
    geom_point(alpha = 0.20) +
    scale_y_log10() +
    geom_smooth(
      method = lm,
      formula = y ~ ns(x, df = deg_free),
      color = "lightblue",
      se = FALSE
    ) +
    labs(
      title = paste(deg_free, "Spline Terms"),
      y = "Sale Price (USD)"
    ) +
    theme_bw()
}


(plot_smoother(2) + plot_smoother(5)) / (plot_smoother(20) + plot_smoother(100))

graphics.off()

# The `splines::ns()` function generates feature columns using functions
# called *natural splines*.


# The plots from above show that two terms *underfit* the data while
# 100 terms *overfit* the data.

# The panels with five and twenty terms seem to be the best fit.


# In "recipes", multiple steps can create these types of terms.
# To add a natural spline representation for this predictor:
recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude, data = ames_train) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) |> 
  step_ns(Latitude, deg_free = 20)

# The user would need to determine if both neighborhood and latitude should be in
# the model since they both represent the same underlying data in different ways.


### 8.4.4 Feature extraction ----

# Feature extraction is the process of creating new features from existing data.
# Principal Component Analysis (PCA) tries to extract as much of the original information
# in the predictor set as possible using a smaller number of features.

# PCA is a linear extraction method, meaning that each new feature is a linear
# combination of the original predictors.

# The new features, called the principal components, or PCA scores,
# are uncorrelated, or orthogonal to each other.

# Therefore PCA is useful when the predictors in the data set are highly correlated
# with each other.



# In the Ames housing data, seve3ral predictors measure the size of the property:
# - `Total_Bsmt_SF`: Total basement size
# - `First_Flr_SF`: Size of the first floor
# - `Gr_Liv_Area`: Gross living area

# PCA might be an option ot represent these potentially redundant variables
# in a smaller feature set.

# Aaprt from the gross living area `Gr_Liv_Area`, all these predicors have
# the suffix `SF` in their names (for square feet), so a recipe step for PCA may be:
# `step_pca(matches("(SF$)|(Gr_Liv)"))`


# Note that all of these columns are measured in square feet.
# PCA assumes that all predictors are on the same scale.
# While this is true in this case, you will usually need to call
# `step_normalize()` to center and scale each column before applying PCA.


# Additional recipe steps for feature extraction are:
# - ICA: Independent Component Analysis
# - NNMF: Non-Negative Matrix Factorization
# - MDS: MultiDimensional Scaling
# - UMAP: Uniform Manifold Approximation


### 8.4.5 Row sampling steps ----

# Recipe steps can affect the rows of a data set.

# Subsampling techniques for class imbalances change the class proportions
# in the data to generate better behaved distributions fo the predicted class probabilities.

# Approaches to subsampling data with class imbalances include:

# - Downsampling: Keep the minority class and take a random sample of the majority class
#   so that class frequencies are balanced.

# - Upsampling: Replicate samples form the minority class to balance classes.
#   Synthesize new samples that resemble the minority class or simply add the same
#   minority samples repeatedly.

# - Hybrid methods: Do a combination of the above.


# The "themis" package has recipe steps to address class imbalances with
# subsampling, like `step_downsample(outcome_column_name)`.


# Note that you may only affect the training data with subsampling techniques.
# All of the subsampling steps default the `skip` argument to have a value
# of `TRUE`.


# Other step functions that are row-based are:
# - `step_filter()`
# - `step_sample()`
# - `step_slice()`
# - `step_arrange()`
# In almost all use cases, the `skip` argument should be set to `TRUE`.


### 8.4.6 General transformations ----

# Mirroring the original "dplyr" operation, `step_mutate()` can be sued
# to compute the ration of two variables, such as 
# `Bedroom_AbvGr / Full_Bath`, the ratio of bedrooms to bathrooms.


# Use extra care to avoid data leakage in your preprocessing.



### 8.4.7 Natural language processing ----

# The "textrecipes" package can apply natural language processing methods to the data.

# The input column is a string of text, and different steps can be used
# to tokenize the data, e.g. split the text into separate words,
# filter out tokens,
# and create new features appropriate for modeling.


## 8.5 Skipping steps for new data ----

# The sale price data are already log-transformed in the `ames` data frame.

# Why not use: `step_log(Sale_Price, base = 10)` ?

# This will cause a failure when the recipe is applied to new data
# with an unknown sale price.

# To avoid information leakage, many tidymodels packages isolate the data being
# used when making any predictions.

# this means the training set and any outcome columns are not available for use
# at prediction time.


# Any transformations of the outcome columns should be conducted outside of the recipe.


# There are circumstances where this is not an adequate solution.
# In classification models where there is a severe class imbalance,
# it is common to conduct *subsampling* of the data that are given to the
# modeling function.

# Suppose there are two classes and a 10% event rate.
# A controversial approach would be to *downsample* the data so the model is
# provided with all of the events and a random 10% of the nonevent samples.

# The subsampling process should not be applied to the data being predicted.

# Each `step_*()` function in the "recipes" package has a `skip` argument that,
# when set to `TRUE`, will be ignored by the `predict()` function.

# The steps will still be applied when you call `fit()`.


# The step functions in the "recipes" and "themis" packages that are *only*
# applied to the training data are:
# - `step_adasyn()`
# - `step_bsmote()`
# - `step_downsample()`
# - `step_filter()`
# - `step_naomit()`
# - `step_nearmiss()`
# - `step_rose()`
# - `step_sample()`
# - `step_slice()`
# - `step_smote()`
# - `step_smotenc()`
# - `step_tomek()`
# - `step_upsample()`


## 8.6 Tidy a `recipe()` ----

# `broom::tidy()` can tidy statistical objects.

# First, we create an extended recipe for the Ames housing data:
ames_rec <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
    Latitude + Longitude,
  data = ames_train
  ) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |>
  step_ns(Latitude, Longitude, deg_free = 20)

# The `broom::tidy()` method, when called with the "recipe" object,
# gives a summary of the recipe steps.
class(ames_rec)
# "recipe"

broom::tidy(ames_rec)


# This result can be helpful for identifying individual steps, perhaps to then
# be able to execute the `broom::tidy()` method in one specific step.


# We can specify the `id` argument in any function call; otherwise it is generated
# using a random suffix.

# Setting this value is helpful if the same type of step is added to the recipe several times.

# Let's specify the `id` ahead of time for `step_other()` since we'll want to
# `broom::tidy()` it:
ames_rec <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
    Latitude + Longitude, data = ames_train
  ) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01, id = "my_id") |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |> 
  step_ns(Latitude, Longitude, deg_free = 20)

# We'll refit the workflow with this new recipe:
lm_wflow <- workflow() |> 
  add_model(lm_model) |> 
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)


# The `broom::tidy()` method can be called again along with the `id` identifier
# we specified to get our results for applying `step_other()`:

estimated_recipe <- lm_fit |> 
  extract_recipe(estimated = TRUE)

broom::tidy(estimated_recipe, id = "my_id")

# The `broom::tidy()` results we see here for using `step_other()` show
# which factor levels were retained, i.e., not added to the new "other" category.


# The `broom::tidy()` method can be called with the `number` identifier as well,
# if we know which step in the recipe we need:
broom::tidy(estimated_recipe, number = 2)

# Each `broom::tidy()` method returns the relevant information about that step.


## 8.7 Column roles ----

# When a formula is sued with the initial call to `recipe()`, the *roles*
# "predictor" or "outcome" are assigned to each of the columns,
# depending on which side of the tilde `~` they are on.


# In the Ames housing data, the original raw data contained a column for
# address.

# It may be useful to keep that column in the data so that, after 
# predictions are made, problematic results can be investigated in detail.

# In other words, the column could be important even though it is neither
# a predictor nor an outcome.


# The `add_role()`, `remove_role()`, and `update_role()` functions can be used

# The role of the street address could be modified using:
ames_rec |> 
  update_role(address, new_role = "street address")


# After this change, the `address` column in the data frame will no longer
# be a predictor but instead will be a `"street address"` according to the recipe.

# Any character string can be used a s a role.

# Columns can also have multiple roles added with `add_role()`.


# This step is helpful when the data are *resampled*.
# It helps to keep the columns that are not involved with the model fit in the
# same data frame, rather than in an external vector.


# Finally, all step functions have a `role` field that can assign roles to the
# results of the step.

# In many cases, columns affected by a step retain their existing role.

# For example, the `step_log()` calls to our `ames_rec` object affected
# the `Gr_Liv_Area` column.

# For that step, the default behavior is to keep the existing role for this column 
# sinc eno new column is created.

# A counter-example would be the step to produce splines which defaults
# new columsn to a new role of `"predictor"`.


# END
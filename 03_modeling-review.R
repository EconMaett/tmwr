# Chapter 03 - A review of R modeling fundamentals ----

# Before introducing the tidymodels framework, we review
# some of the fundamentals of modeling in base R.


# The S language, on which R is based, had a rich data analysis environment
# ever since the publication of The White Book by Chambers and Hastie in 1992.

# This version of S introduced some of the standard infrastructure components
# that are still used today, including:
# - Symbolic model formula (e.g., `y ~ x1 + x2`)
# - Model matrices (e.g., `model.matrix()`)
# - Data frames (e.g., `data.frame()`)

# The S language was also the first to introduce the concept of a generic function,


## 3.1 An example ----

# We use experimental data from McDonald (2009) by way of Mangiafico (2015),
# on the relationship between the ambient temperature and the rate of cricked chirps per minute.

# The data was collected for two species of crickets, the snowy tree cricket and the common field cricket.

# The data are contained in a data frame called `crickets` with a total of 31 data points.
library(tidyverse)

data(crickets, package = "modeldata")
names(crickets)
# "species" "temp" "rate"

# Plot the temperature on the x-axes, the chirp rate on the y-axis.
# The plot elements will be colored for each species:
ggplot(data = crickets, mapping = aes(x = temp, y = rate, color = species, pch = species, lty = species)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = FALSE, alpha = 0.5) +
  scale_color_brewer(palette = "Paired") +
  labs(x = "Temperature (C)", y = "Chirp Rate (per minute)")

graphics.off()

# The data exhibits linear trends for both species.

# For an inferential model, one might be tempted to formulate the following
# null hypotheses prior to seeing the data:

# - Temperature has no effect on the chirp rate.
# - There are no differences between the species' chirp rate.

# To fit an ordinary linear model in base R, use the `lm()` function.
# The formula is *symbolic* `rate ~ temp` or `rate ~ temp + time`.

# The model formula `rate ~ temp + species` creates a model with different
# y-intercepts for each species.

# Should the slopes for each species be different, the formula would be
# `rate ~ temp + species + temp:species`.

# A shortcut can be used to expand all interactions containing interactions 
# with two variables:
# `rate ~ (temp + species)^2`

# Another shortcut to expand factors to include all possible interactions:
# `rate ~ temp * species`


# Besides automatically creating dummy variables and interaction terms 
# for factor variables, the `lm()` function as additional features:

# - *In-line* functions can be used in the formula.
#   For example, `rate ~ log(temp)` or `rate ~ I((temp * 9/5) + 32)`.

# - Other functions that can be used inside of formulas include `poly(x, 3)`,
#   which adds linear, quadratic, and cubic terms for `x`.
#   The "splines" package provides additional functions for creating spline terms.

# - For data sets with many predictors, the period `.` shortcut
#   represents all other variables in the data frame.
#   Using `rate ~ (.)^3` would add all possible three-way interactions
#   for all variables in the data frame except `rate`.


# We use the suffix `_fit` to store teh two-way interaction model:
interaction_fit <- lm(formula = rate ~ (temp + species)^2, data = crickets)

print(interaction_fit)

# We can use the `plot()` method for `"lm"` objects
class(interaction_fit)
# "lm"

# Place two plots next to one another:
par(mfrow = c(1, 2))

# Show residuals vs predicted values
plot(interaction_fit, which = 1)

# A normal quantile plot on the residuals
plot(interaction_fit, which = 2)

graphics.off()


# We should assess if the inclusion of the interaction term is warranted.
# The `anova()` method for `"lm"` objects can be used to compare models.

# We fit a reduced model:
main_effect_fit <- lm(formula = rate ~ temp + species, data = crickets)

# Compare the two models:
anova(main_effect_fit, interaction_fit)

# This statistical test generates a p-value, Pr(>F) of 0.2542.
# This implies that there is a lack of evidence against the null hypothesis
# that the interaction term is not needed by the model.

# For this reason, we conduct further analysis on the model without the interaction.


# The `summary()` method for `"lm"` objects provides a summary of the model fit:
summary(main_effect_fit)


# If we needed to estimate the chirp rate at a temperature that was not observed
# in the experiment, we could use the `predict()` method.

# It takes the model object and a data frame of new values for  a prediction.

new_values <- data.frame(species = "O. exclamationis", temp = 15:20)

predict(object = main_effect_fit, newdata = new_values)


## 3.2 What does the R formula do? ----

# - The formula defines the columns that the model uses.
# - The formula encodes the columns in an appropriate format.
# - The roles of the columns are defined by the formula.


## 3.3 Why tidiness is important for modeling ----

# Three common methods for crating a scatter plot of two numeric variables
# in a data frame `plot_data` are:

# `plot(plot_data$x, plot_data$y)`

# library(lattice)
# xyplot(y ~ x, data = plot_data)

# library(ggplot2)
# ggplot(data = plot_data, mapping = aes(x = x, y = y)) + geom_point()


# In R, there is often more than one way to do something.
# But the syntax is often similar:

# Function        Package      Code
# `lda()`         MASS        `predict(object)`
# `glm()`         stats       `predict(object, type = "response")`
# `gbm()`         gbm         `predict(object, type = "response", n.trees)`
# `mda()`         mda         `predict(object, type = "posterior")`
# `rpart()`       rpart       `predict(object, type = "prob")`
#  various        RWeka       `predict(object, type = "probability")`
# `logitboost()`  LogitBoost  `predict(object, type = "raw", nIter)`
# `pamr.train()`  pamr        `pamr.predict(object, type = "posterior")`


# The last example is the only one that uses a custom function `pamr.predict()`
# instead of the generic `predict()` function.


# When models make predictions, the vast majority require all of the predictors
# to have complete values.

# Several options are baked into base R with the generic
# `na.action()` function.

# This sets the policy for how a function should behave if there are missing
# values, `NA`s.

# The two most common policies are `na.fail()` and `na.omit()`.


# Add a missing value to the prediction set
new_values$temp[1] <- NA

# The `predict()` method for `"lm"` defaults to `na.pass()`:
predict(object = main_effect_fit, newdata = new_values)


# Alternatively
predict(object = main_effect_fit, newdata = new_values, na.action = na.fail)
# Error

predict(object = main_effect_fit, newdata = new_values, na.action = na.omit)


# The tidymodels packages use  a set of design goals.

# Examples include:

# - Use generic functions like `predict()` instead of new custom `predict_*()` functions.

# - Sensible defaults are important. 
#   Also, functions should not have a default argument for the first, 
#   most important argument to be defined by the user.

# - Argument values whose default can be derived from the data should be.
#   For example, for `glm()`, the `family` argument could check the type of the data
#   in the outcome variable, and if no `family` is given, the default
#   is determined from the data.

# - Functions should take the data structures that *users* want,
#   not the ones that *developers* want.
#   A developer would prefer matrices, but a user prefers data frames.


# The `broom::tidy()` function can return many R objects in a more
# usable format.
# Suppose that predictors are being screened based on their correlation
# to the output column.

# Using `purrr::map(.x, .f)`, the results from `cor.test()` can be returned
# in a "list" for each predictor:
corr_res <- purrr::map(.x = mtcars |> select(-mpg), .f = cor.test, y = mtcars$mpg)

# The first of the ten results is:
corr_res[[1]]


# If we want to use this information in a plot, `broom::tidy()`
# returns a tibble with standardized names:
broom::tidy(corr_res[[1]])

# These results can be stacked and added to a `ggplot()`:
corr_res |> 
  map_dfr(.f = broom::tidy, .id = "predictor") |> 
  ggplot(mapping = aes(x = fct_reorder(predictor, estimate))) +
  geom_point(mapping = aes(y = estimate)) +
  geom_errorbar(mapping = aes(ymin = conf.low, ymax = conf.high), width = 0.1) +
  labs(x = NULL, y = "Correlation with mpg") +
  theme_bw()


## 3.4 Combining base R models and the tidyverse ----

# Suppose we want to fit separate models for each cricket species.
# We first break out the cricket data by the `species` column using
# `dplyr::group_nest()`:
split_by_species <- crickets |> 
  group_nest(species)

split_by_species
# A tibble: 2 * 2
# species                         data
# <fct>             <list<tibble[,2]>>
# O. exclamationis            [14 * 2]
# O. niveus                   [17 * 2]

# The `data` column contains the `rate` and `temp` columns from `crickets`
# in a list-column.

# From this, the `purrr::map()` functionc an create individual models
# for each species:
model_by_species <- split_by_species |> 
  mutate(
    model = map(.x = data, .f = ~ lm(formula = rate ~ temp, data = .x))
  )

model_by_species
# A tibble: 2 * 3
# species                         data model
# <fct>             <list<tibble[,2]>> <list>
# O. exclamationis            [14 * 2] <lm>
# O. niveus                   [17 * 2] <lm>

# To collect the coefficients of thse models, use `broom::tidy()`
# to convert them to a data frame format so that they can be unnested:
model_by_species |> 
  mutate(
    coef = map(.x = model, .f = broom::tidy)
  ) |> 
  select(species, coef) |> 
  unnest(cols = c(coef))


## 3.5 The tidymodels metapacakge ----

# The tidyverse is designed as a set of modular R packages, each with
# a fairly narrow scope.

# The tidymodels framework follows a similar design.

# For example, the "rsample" package focuses on data splitting and resampling.

# The performance metrics are contained in the "yardstick" package.


# The "tidymodels" metapackage loads the core packages and a few
# other packages that are useful for modeling.
library(tidymodels)

# Note that this usually leads to a few conflicts,
# so it can be useful to apply the `package::function()` syntax to avoid
# confusion.


# Another option is to use the "conflicted" package.
# It allows you to set a rule that remains in effect until the
# end of the R session to ensure that one specific function
# will always run if no namespace is given in the code.

# For example, if we prefer `dplyr::filter()` over `stats::filter()`,
# and want to ensure that `filter()` always calls `dplyr::filter()`,
# we call:
library(conflicted)
conflicted::conflict_prefer(name = "filter", winner = "dplyr", loser = "stats")
# [conflicted] Will prefer dplyr::fitler over stats::filter.


# For convenience, tidymodels contains a function that captures most of the
# common naming conflicts we might encounter:
tidymodels::tidymodels_prefer(quiet = FALSE)

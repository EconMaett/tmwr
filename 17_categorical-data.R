# Chapter 17 - Encoding categorical data ----

# For statistical modeling in R, the preferred representation for
# categorical data is a *factor", which is a variable type that
# can take on a limited number of different values.

# Internally, factors are stored as a vector of integer values with a
# set of text labels.


# Many models require transformations to a numeric representation for
# categorical data.

# But for some data sets, dummy variables are not a good fit.

# This often happens because there are *too many* categories
# or where there are *new* categories at prediction time.

# These options are available as tidymodels recipe steps in the
# "embed" and "textrecipes" packages.


## 17.1 Is an encoding necessary? ----

# A minority of models like trees or rules can handle categorical data natively
# and do not require encoding or transformation of these features.

# Naive Bayes models are another example where the structure of the model
# can deal with categorical variables natively.


# These models can not only handle categorical features but also deal with
# numeric, continuous features.


# It is advised to start with untransformed categorical variables when a model allows it.
# Complex encodings often do not result in better performance for such models.


## 17.2 Encoding ordinal predictors ----

# Qualitative columns may have a natural order, like "low", "medium", and "high".

# In base R, the default encoding strategy is to make new columns that
# are polynomial expansions of the data.


# Instead, recipe steps for ordered factors such as `step_unorder()`,
# to convert to unordered factors, or `step_ordinalscore()`, which
# maps specific numeric values for each factor levels may be used.


## 17.4 Using the outcome for encoding predictors ----

# One method for encoding factor variables that is more complex than dummy
# or indicator variables is called *effect* or *likelihood encodings*.

# These replace the original categorical variables with a single numeric
# column that measures the effect of those data.

library(tidymodels)
tidymodels_prefer()

data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

# From the Ames housing data, we can compute the mean or meadian
# sale price for each neighborhood and substitute these means
# for the original values:
ames_train |> 
  group_by(Neighborhood) |> 
  summarise(
    mean = mean(Sale_Price),
    std_err = sd(Sale_Price) / sqrt(length(Sale_Price))
  ) |> 
  ggplot(mapping = aes(y = reorder(Neighborhood, mean), x = mean)) +
  geom_point() +
  geom_errorbar(mapping = aes(xmin = mean - 1.64 * std_err, xmax = mean + 1.64 * std_err)) +
  labs(
    x = "Price (nean, log scale)",
    y = NULL
  ) +
  theme_bw()

graphics.off()

# This kind of effect encoding works well when your categorical variable has
# many levels.

# In "tidymodels", the "embed" package includes several "recipe" `step_*()` functions
# for different types of effect encodings such as
# `step_lencode_glm()`, `step_lencode_mixed()`, and `step_lencode_bayes()`.

# These steps use a generalized linear model to estimate the effect of each level
# in a categorical predictor on the outcome.

# When using a recipe step like `step_lencode_glm()`, specify the variable
# being encoded first and then the outcome using `vars()`:
library(embed)

ames_glm <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train
  ) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_lencode_glm(Neighborhood, outcome = vars(Sale_Price)) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |> 
  step_ns(Latitude, Longitude, deg_free = 20)

print(ames_glm)


# We can `prep()` our "recipe" to fit or estimate parameters for the
# preprocessing transformations using training data.

# We can then `broom::tidy()` this prepared recipe to see the results:
glm_estimates <- prep(ames_glm) |> 
  broom::tidy(number = 2)

print(glm_estimates)


# When we use the newly encoded `Neighborhood` numeric variable created via
# this method, we substitute the original level (such as `"North_Ames"`)
# with the estimate for `Sale_Price` from the GLM.

# Effect encoding methods like this one can seamlessly handle situations
# where a novel factor level is encountered in the data.

# This `value` is the predicted price from the GLM when we don't have
# any specific neighborhood information:
glm_estimates |> 
  filter(level == "..new")


# Note that the effects need to be computed from the training set,
# after data splitting.

# This type of supervised preprocessing should be rigorously 
# resampled to avoid overfitting.


# When you create an effect encoding for your categorical variable,
# you are effectively layering a mini-model inside your actual model.

# The possibility of overfitting with effect encodings is
# an example of why feature engineering *must* be considered part of 
# the model process, and why feature engineering must be estimated
# together with model parameters inside resampling.


### 17.3.1 Effect encodings with partial pooling ----

# Creating an effect encoding with `step_lencode_glm()` estimates the effect
# separately for each factor level, in this example, for each neighbhorhood.


# However, some of these neighborhoods have many houses in them, while
# others only have a few.

# We can use *partial pooling* to adjust these estimates so that levels
# with small sample sizes are shrunken toward the overall mean.

# The effects for each level are modeled all at once using a mixed or 
# hierarchical generalized linear model:
ames_mixed <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train
) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_lencode_mixed(Neighborhood, outcome = vars(Sale_Price)) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |> 
  step_ns(Latitude, Longitude, deg_free = 20)

print(ames_mixed)


# Let's `prep()` and `broom::tidy()` this recipe to see the results:
mixed_estimates <- prep(ames_mixed) |> 
  broom::tidy(number = 2)

print(mixed_estimates)


# New levels are then encoded at close to the same value as with the GLM:
mixed_estimates |> 
  filter(level == "..new")


# You can use a fully Bayesian hierarchical model for the effects in the
# same way with `step_lencode_bayes()`:

# Let's visually compare the effects using partial pooling vs. no pooling:
glm_estimates |> 
  rename(`no pooling` = value) |> 
  left_join(
    mixed_estimates |> 
      rename(`partial pooling` = value),
    by = "level"
  ) |> 
  left_join(
    ames_train |> 
      count(Neighborhood) |> 
      mutate(level = as.character(Neighborhood))
  ) |> 
  ggplot(mapping = aes(x = `no pooling`, y = `partial pooling`, size = sqrt(n))) +
  geom_abline(color = "gray50", lty = 2) +
  geom_point(alpha = 0.7) +
  coord_fixed() +
  theme_bw()

graphics.off()

# Most estimates for neighborhood effects are about the same when we
# compare pooling to no pooling.

# However, the neighborhoods with the fewest homes in them have been pulled
# (either up or down) toward the mean effect.

# When we use pooling, we shrink the effect estimates toward the mean because
# we don't have as much evidence about the price in those neighborhoods.


## 17.4 Feature hashing ----

# Traditional dummy variables require that all of the possible categories
# be known to create a full set of numeric features.


# *Feature hashing* methods also create dummy variables, but only consider
# the value of the category to assign it to a predefined pool of dummy variables.

# We apply `rlang::hash()` to the `Neighborhood` values in the Ames housing data:
library(rlang)

ames_hashed <- ames_train |> 
  mutate(
    Hash = map_chr(.x = Neighborhood, .f = rlang::hash)
  )


ames_hashed |> 
  select(Neighborhood, Hash)

# If we input `"Briardale"` to this hashing function, we will always
# get the same output.

# The neighborhoods in this case are called the "keys*
# while the outputs are the "hashes".


# A hashing function takes an input of variable size and maps it to
# an output of fixed size.

# They are commonly used in cryptography and databases.


# The `rlang::hash()` function generates a 128-bit hash, which means there
# are `2^128` possible hash values.

# This is great for some applications, but does not help with feature
# hashing of *high cardinality* variables (variables with many levels).


# In feature hashing, the number of possible hashes is a hyperparameter and
# is set by the model developer through computing the modulo
# of the integer hashes.

# We can get sixteen possible hash values by using `Hash %% 16`:
ames_hashed |> 
  ## first make a smaller hash for integers that R can handle
  mutate(
    Hash = strtoi(substr(x = Hash, start = 26, stop = 32), base = 16L),
    ## now take the modulo
    Hash = Hash %% 16
  ) |> 
  select(Neighborhood, Hash)


# Now instead of 28 neighborhoods in our original data or an incredibly
# huge number of the original hashes, we have sixteen hash values.

# This method is very fast and memory efficient,
# and it can be a good strategy when there are a large number of possible
# categories.


# Feature hashing is useful for text data as well as high cardinality
# categorical data.


# We implement feature hashing using a "tidymodels" "recipe" `step_*()`
# function from the "textrecipes" package:
library(textrecipes)

ames_hash <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train
  ) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_dummy_hash(Neighborhood, signed = FALSE, num_terms = 16L) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |> 
  step_ns(Latitude, Longitude, deg_free = 20)


print(ames_hash)


# While fast and efficient, feature hashing has some downsides.

# Different category values often map to the same hash value.

# This is called a *collision* or *aliasing*.


# The number of neighborhoods mapped to each hash value varies
# between zero and four.

# all hash values greater than one are hash collisions.


# Note that feature hashing is not interpretable because hashing
# functions cannot be reversed.

# We can't determine what the input category levels were from the hash value,
# or if a collision occurred.


# The number of hash values is a *tuning parameter* of this preprocessing 
# technique, and you need to try several values to determine the best one.

# A lower number of hash values results in more collisions, but 
# a high number is no improvement over the original high cardinality variable.


# Feature hashing can handle new category levels at prediction time, since it 
# does not rely on pre-determined dummy variables.


# You can reduce hash collisions with a *signed* has by using
# `signed = TRUE`.

# This expands the values from only 1 to either +1 or -1, 
# depending on the sign of the hash.


# It is likely that some hash columns will contain all zeros.
# We recommend a zero-variance filter via `step_zv()` to filter out
# such columns.


## 17.5 Model encoding options ----

# We can build a full set of *entity embeddings* to transform a 
# categorical variable with many levels to a set of lower-dimensional vectors.

# This approach is best suited to a nominal variable with many category levels,
# many more than the example we have used with neighborhoods in AMES.


# END
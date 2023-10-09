# Chapter 05 - Spending our data ----

## 5.1 Common methods for splitting data ----

# The training set is usually the majority of the data.
# These data are a sandbox for model building,
# where different models can be fit,
# feature engineering strategies investigated, and so on.


# The test set is held in reserve until one or two models are chosen
# as the methods most likely to succeed.

# The test set is then used as the final arbiter to determine the efficacy of the model.


# The "rsample" package has tools for making data splits such as
# `rsample::initial_split()`.

# It takes the data frame as its argument as well as the proportion
# to be placed into training.
library(tidymodels)
tidymodels_prefer()

# Set the random number stream using `set.seed()` to make the results reproducible
set.seed(501)

# Save the split information for an 80/20 split of the data
ames_split <- initial_split(data = ames, prop = 0.80)
ames_split
# <Training/Testing/Total>
# <2344/596/2930>

class(ames_split)
# "initial_split" "mc_split" "rsplit"

# To get the resulting data sets, we apply two more functions:
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

dim(ames_train)
# 2344 74


# These objects are data frames with the same columns as the original data
# but only the appropriate rows for each set.


# In most cases, such "simple random sampling" is appropriate.

# When there is a dramatic *class imbalance* in classification problems,
# simple random sampling is insufficient.

# To avoid haphazardly allocating the infrequent class disproportionately to
# either the test or the training set, *stratified sampling* is used.


# For regression problems, the outcome data can be artificially binned into
# quartiles and then stratified sampling can be conducted four separate times.

# This is an effective method for keeping the distributions of the outcomes 
# similar between the training and test set.


# In the "rsample" package, stratified sampling is done with the `strata`
# argument inside `rsample::initial_split(data, prop, strata)`.
set.seed(502)
ames_split <- initial_split(data = ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

dim(ames_train)
# 2342 74

# We have seen that the price distribution is right-skewed, with proportionately
# more inexpensive houses than expensive houses on either side of the center of the distribution.

# The worry is that with simple random sampling, we accidentally split the data
# into training and test sets where the training set has a much larger proportion of 
# expensive houses.


# Other situations in which simple random sampling is not an appropriate choice
# for data splitting includes time series and spatial data,
# where we would neglect the time or spatial component.

# In time series data, it is more common to use the most recent data as the test set.
# `rsample::initial_time_split(data, prop)` splits the data in a way that
# the `prop` argument denotes what proportion of the first part of the data
# should be used as the training set.

# Note that the function assumes that the data are sorted by time.


## 5.2 What about a validation set? ----

# If you use a validation set, start with a different splitting function:
set.seed(52)

# To put 60% into training, 20% into validation, and 20% into testing:
ames_val_split <- initial_validation_split(data = ames, prop = c(0.6, 0.2))
ames_val_split
# <Training/Validation/Testing/Total>
# <1758/586/586/2930>


# To get the training, validation, and testing data, use the same syntax:
ames_train <- training(ames_val_split)
ames_test <- testing(ames_val_split)
ames_val <- validation(ames_val_split)


## 5.3 Multilevel data ----

# With the Ames housing data, a property is considered to be the
# *independent experimental unit*.

# It is safe to assume that, statistically, the data from a property are independent
# of other properties.

# For other applications, that may not be the case:

# For longitudinal data, the same independent experimental unit can
# be measured over multiple time points.
# This is typically the case for a subject in a medical trial.

# END
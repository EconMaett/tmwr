# Chapter 19 - When should you trust your predictions? ----

# When a new data point is well outside of the range of the data used to
# create a model, this prediction will be an inappropriate *extrapolation*.


# A more qualitative example of an inappropriate prediction would be
# when the model is used in a completely different context.


# The cell segmentation data from Chapter 14 flags when human breast cancer
# cells can or cannot be accurately isolated inside an image.


# A model built from these data could be inappropriately applied to stomach cells
# for the same purpose.


# There are two methods for quantifying the potential quality of a prediction:

# - *Equivocal zones* use the predicted values to alert the user that results may be suspect.

# - *Applicability* uses the predictors to measure the amount of extrapolation (if any) for new samples.


## 19.1 Equivocal results ----

# In some cases, the amount of uncertainty associated with a prediction is too high to be trusted.


# Regulatory bodies often require any medical diagnostics to have an *equivocal zone*.

# This zone is a range of results in which the prediction should not be reported to pations.


# The same notion can be applied to models created outside of medical diagnostics.


# We use a function that simulates classification data with two classes
# and two predictors, `x` and `y`.


# The true model is a logistic regression model with the equation:

# logit(p) = -1 - 2x - \frac{x^2}{5} + 2y^2


# The two predictors follow a bivariate normal distribution
# with a correlation of 0.7.
# This means E[x, y] = (0, 0)^T and COV(x, y) = (1, 0.7; 0.7, 1).

# We create a training set of 200 samples and a test set of 50:
library(tidymodels)
tidymodels_prefer()


simulate_two_classes <- function(n, error = 0.1, eqn = quote(-1 - 2*x - 0.2*x^2 + 2*y^2)) {
  
  # Slightly correlated predictors
  sigma <- matrix(data = c(1, 0.7, 0.7, 1), nrow = 2, ncol = 2)
  
  dat <- MASS::mvrnorm(n = n, mu = c(0, 0), Sigma = sigma)
  
  colnames(dat) <- c("x", "y")
  
  cls <- paste0("class_", 1:2)
  
  dat <- as_tibble(dat) |> 
    mutate(
      linear_pred = !!eqn,
      
      # Add some misclassification noise
      linear_pred = linear_pred + rnorm(n, sd = error),
      prob = binomial()$linkinv(linear_pred),
      class = ifelse(test = prob > runif(n), yes = cls[1], no = cls[2]),
      class = factor(class, levels = cls)
    )
  
  return(dplyr::select(dat, x, y, class))
}


set.seed(1901)

training_set <- simulate_two_classes(n = 200)
testing_set <- simulate_two_classes(n = 50)


# Estimate a logistic regression model using Bayesian methods,
# using the default Gaussian prior distribution for the parameters:
two_class_mod <- logistic_reg() |> 
  set_engine("stan", seed = 1902) |> 
  fit(class ~ . + I(x^2) + I(y^2), data = training_set)


print(two_class_mod, digits = 3)


# The fitted class boundary is overlaid onto the test set in Figure 19.1.

# The data point closest to the class boundary are the most uncertain.

# If their values change slightly, their predicted class might change.


# One simple method for disqualifying some results is to class them
# "equivocal" fi the values are within some range around 50%
# (or the appropriate cutoff for a certain situation).

# Depending on the problem the model is being applied to, this might indicate
# we should collect another measurement or we require more information
# before a trustworthy prediction is possible.


# We could base the width of the band around the cutoff on how performance
# improves when the uncertain results are removed.

# However, we should also estimate the reportable rate
# (the expected proportion of usable results).


# We use the test set to determine the balance between improving performance
# and having enough reportable results.

# The predictions are created using:
test_pred <- augment(two_class_mod, testing_set)

test_pred |> 
  head()


# With "tidymodels", the "probably" package contains functions for equivocal
# zones.

# For cases with two classes, the `make_two_class_pred()` function creates
# a factor-like column that has the predicted classes with an equivocal zone:
library(probably)

lvls <- levels(training_set$class)

test_pred <- test_pred |> 
  mutate(
    .pred_with_eqz = make_two_class_pred(
      estimate = .pred_class_1,
      levels = lvls,
      buffer = 0.15
    )
  )

test_pred |> 
  count(.pred_with_eqz)

# Rows that are within 0.50 +/1 0.15 are given a value of `[EQ]`.


# Since the factor levels are the same as the original data, confusion matrices
# and other statistics can be computed without error.


# When using standard functions from the "yardstick" package,
# the equivocal results are converted to `NA` and are not used
# in calculations that use the hard class predictions.

# Notice the differences in these two confusion matrices:

# All data
test_pred |> 
  conf_mat(class, .pred_class)


# Reportable results only:
test_pred |> 
  conf_mat(class, .pred_with_eqz)


# An `is_equivocal()` function can filter these two rows from the data.


# Does the equivocal zone improve accuracy?
# We examine different buffer sizes:

# A function to change the buffer size and then compute the performance
eq_zone_results <- function(buffer) {
  
  test_pred <- test_pred |> 
    mutate(
      .pred_with_eqz = make_two_class_pred(
        estimate = .pred_class_1,
        levels = lvls,
        buffer = buffer
      )
    )
  
  acc <- test_pred |> 
    accuracy(class, .pred_with_eqz)
  
  rep_rate <- reportable_rate(test_pred$.pred_with_eqz)
  
  return(tibble(
    accuracy = acc$.estimate,
    reportable = rep_rate,
    buffer = buffer
  ))
}


# Evaluate a sequence of buffers and plot the results
map(.x = seq(from = 0, to = 0.1, length.out = 40), .f = eq_zone_results) |> 
  list_rbind() |> 
  pivot_longer(c(-buffer), names_to = "statistic", values_to = "value") |> 
  ggplot(mapping = aes(x = buffer, y = value, lty = statistic)) +
  geom_step(linewidth = 1.2, alpha = 0.8) +
  labs(
    title = "The effect of equivocal zones on model performance",
    y = NULL,
    lty = NULL
  ) +
  theme_bw()


graphics.off()

# The figure shows that accuracy improves by a few percentage points but
# at the cost of nearly 10% of predictions being unusable.

# The value of such a compromise depends on how the model predictions are used.


# This analysis focused on using the predicted class probability to disqualify points.

# A better approach is to use the standard error of the class probability.

# Since we used a Bayesian model, the probability estimates we found
# are actually the mean of the posterior predictive distribution.

# The Bayesian model gives us a distribution for the class probability.

# Measuring the standard deviation of this distribution gives us a
# *standard error of prediction* of the probability.


# In most cases, this value is directly related to the mean class probability.

# You might recall that for a Bernoulli random variable with probability p,
# the variance is p*(1-p).

# Because of this relationship, the standard error is largest when the probability is 0.5.


# The standard error of prediction takes into account more than just the class probability.

# When there are aberrant predictor values or a significant extrapolation,
# the standard error increases.


# This helps us flag predictions that are uncertain.


# The advantage of a Bayesian model is that it naturally estimates
# the standard error of prediction.


# Using `type = "pred_int"` produces upper and lower limits and
# the `std_error` adds a column for that quantity.

# For 80% prediction intervals:
test_pred <- test_pred |> 
  bind_cols(
    predict(two_class_mod, testing_set, type = "pred_int", std_error = TRUE)
  )



## 19.2 Determining model applicability ----

# It may be that model statistics like the standard error of prediction
# cannot measure the impact of extrapolation.

# Is our model applicable for predicting a specific data point?

# The goal of the Chicago train data set is to predict the number
# of customers entering the Clark and Lake train station each day.


# Teh data set "Chicago" is in the "modeldata" package.
# It has daily values between January 22, 2001 and August 28, 2016.

# We create a small test set using two weeks of the data:

## Loads both `Chicago` data set as well as `stations`
data("Chicago", package = "modeldata")

Chicago <- Chicago |> 
  select(ridership, date, one_of(stations))

n <- nrow(Chicago)


Chicago_train <- Chicago |> 
  slice(1:(n-14))

Chicago_test <- Chicago |> 
  slice((n-13):n)


# The main predictors are lagged ridership data at different train stations,
# including Clark and Lake, as well as the date.


# The ridership predictors are highly correlated with each other.

# In the following "recipe", the date column is expanded into several new features,
# and the ridership predictors are represented using partial least squares (PLS) components.

# Using the preprocessed data, we fit a standard linear model:
base_recipe <- recipe(ridership ~ ., data = Chicago_train) |> 
  # Create date features
  step_date(date) |> 
  step_holiday(date, keep_original_cols = FALSE) |> 
  # Create dummy variables from factor columns
  step_dummy(all_nominal()) |> 
  # Remove any columns with a single unique value
  step_zv(all_predictors()) |> 
  step_normalize(!!!stations) |> 
  step_pls(!!!stations, num_comp = 10, outcome = vars(ridership))


lm_spec <- linear_reg() |> 
  set_engine("lm")


lm_wflow <- workflow() |> 
  add_recipe(base_recipe) |> 
  add_model(lm_spec)


set.seed(1902)

lm_fit <- fit(lm_wflow, data = Chicago_train)


# How well do the data fit the test set?
# We `predict()` for the test set to find both predictors and prediction intervals:
res_test <- predict(lm_fit, Chicago_test) |> 
  bind_cols(
    predict(lm_fit, Chicago_test, type = "pred_int"),
    Chicago_test
  )

res_test |> 
  select(date, ridership, starts_with(".pred"))

res_test |> 
  rmse(ridership, .pred)
# rmse: 0.865

# These are fairly good results.



# How well would the model work a few years later, in June 2020?
res_2020 <- predict(lm_fit, Chicago_2020) |> 
  bind_cols(
    predict(lm_fit, Chicago_2020, type = "pred_int"),
    Chicago_2020
  )

res_2020 |> 
  select(date, contains(".pred"))


# Given the global pandemic in 2020, the performance on these data
# are abysmal:
res_2020 |> 
  select(date, ridership, starts_with(".pred"))


res_2020 |> 
  rmse(ridership, .pred)
# rmse: 17.2


# This situation can be avoided by having a secondary methodology that 
# quantifies how applicable the model is for the new prediction,
# i.e. the model's *applicability domain*.


# The approach used here is a simple unsupervised method that
# attempts to measure how much a new data point is beyond the
# training data.


# The idea is to accompany a prediction with a score that measures
# how similar the new point is to the training set.


# We apply PCA on the numeric predictor values.

# We first conduct PCA on the training data.

# Next, using these results, we measure the distance of each training set
# point to the center of the PCA data.

# We can then use this *reference distribution* to 
# estimate how far a data point is from the mainstream of the training data.


# For a new sample, the PCA scores are computed along with the distance
# to the center of the training set.


# We can compute a percentile for new samples that reflect how much
# the training set is less extreme than the new samples.

# A percentile of 90% would then mean that most of the training set
# is closer to the data center than the new sample.


# The "applicable" package can develop an applicability model using PCA.

# We use the 20 lagged station ridership predictors as inputs into the PCA.

# The `threshold` argument determines how many components are used in the
# distance calculation.


# We use a large value to indicate that enough components to account for
# 99% of the variation in the ridership predictors should be explained.
library(applicable)

pca_stat <- apd_pca(~ ., data = Chicago_train |> select(one_of(stations)), threshold = 0.99)


print(pca_stat)


# The `autoplot()` method plots the reference distribution.

# We add `distance` to only plot the training set distance distribution:
autoplot(pca_stat, distance) +
  labs(
    title = "The results of using the autoplot() method on an applicable object",
    x = "distance"
  ) +
  theme_bw()

graphics.off()

# The x-axis shows the values of the distance and the y-axis displays the
# distribution's percentiles.

# For example, half of the training set samples had distances less than 3.7.


# Compute the percentiles for new data with the `score()` function:
score(pca_stat, Chicago_test) |> 
  select(starts_with("distance"))

# These seem reasonable.

# For the 2020 data:
score(pca_stat, Chicago_2020) |> 
  select(starts_with("distance"))

# END
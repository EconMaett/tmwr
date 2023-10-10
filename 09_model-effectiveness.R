# Chapter 09 - Judging model effectiveness ----

library(tidymodels)
tidymodels_prefer()

data(ames)
ames <- ames |> 
  mutate(Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(data = ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

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

lm_model <- linear_reg() |> 
  set_engine("lm")

lm_wflow <- workflow() |> 
  add_model(lm_model) |> 
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)


# The best approach to empirical validation involves using *resampling* methods.

# Keep in mind that the test set can only be used once.

# Two common metrics for regression models are the Root Mean Squared Error (RMSE)
# and the coefficient of determination (R-squared).

# The RMSE measures *accuracy* and the R-squared measures *precision*.

# These need not be the same thing!


# A model optimized for RMSE has more variability but has relatively uniform
# accuracy across the range of the outcome.


# The "yardstick" package provides a convenient way to calculate these metrics


## 9.2 Regression metrics ----

# tidymodels prediction functions return tibbles with columns for the predicted values
# called `.pred`.

# Functions from the "yardstick" package that produce performance metrics also have
# consistent interfaces.

# They are data frame-based, as opposed to vector-based, and follow the syntax:
# `function(data, truth, ...)`


# Where `data` is a data frame or tibble and `truth` is the column with the actually observed outcomes.

# The ellipsis `...` or other arguments are used to specify the columns containing the predictions.


# The model `lm_wflow_fit` combines a linear regression model with a predictor set
# supplemented with an interaction and spline function for longitude and latitude.

# It was created from the training set `ames_train`.

# Although we do not advice using the test set at this juncture of the modeling process,
# it will be used here to illustrate functionality and syntax.

# The data frame `ames_test` consists of 588 properties.
dim(ames_test)
# 588 74

# To start, let's produce predictions:
ames_test_res <- predict(lm_fit, new_data = ames_test |> select(-Sale_Price))
print(ames_test_res)

# We combine the predicted and the actual values
ames_test_res <- ames_test_res |> 
  bind_cols(ames_test |> select(Sale_Price))
print(ames_test_res)

ggplot(data = ames_test_res, mapping = aes(x = Sale_Price, y = .pred)) +
  geom_abline(lty = 2) + # Add a diagonal line
  geom_point(alpha = 0.5) +
  labs(
    y = "Predicted Sale Price (log10)",
    x = "Sale Price (log10)"
  ) +
  coord_obs_pred() + # Scale and size the x- and y-axis uniformly
  theme_bw()

graphics.off()

# There is one low-price property that is substantially over-predicted, i.e., high
# above the dashed line.

# We compute the root mean squared error with the `rmse()` function:
rmse(ames_test_res, truth = Sale_Price, estimate = .pred)

# This returns the standard "yardstick" output.

# To get multiple metrics in one go, we create a *metric set*.
# We add the R-squared and the mean absolute error:

ames_metrics <- metric_set(rmse, rsq, mae)

ames_metrics(ames_test_res, truth = Sale_Price, estimate = .pred)

# Note that the RMSE and MAE are both on the scale of the outcome, so `log10(Sale_Price)`.

# While for the RMSE and MAE, values closer to zero are better, for the R-squared,
# values closer to one are better.


## 9.3 binary classification metrics ----

# The "modeldata" package contains the data set "two_class_example"
# with predictions from a test set with two classes, "Class1" and "Class2".

data("two_class_example")
tibble(two_class_example)

# Columns "Class1" and "Class2" contain the predicted class probabilities for the test set
# while the column "predicted" contains the discrete predictions.

# For the hard class predictions, a variety of "yardstick" functions are helpful:

# A confusion matrix:
conf_mat(two_class_example, truth = truth, estimate = predicted)

# Accuracy:
accuracy(two_class_example, truth = truth, estimate = predicted)
# 0.838

# Matthews correlation coefficient:
mcc(two_class_example, truth = truth, estimate = predicted)
# 0.677

# F1 metric:
f_meas(two_class_example, truth = truth, estimate = predicted)
# 0.849


# Combining these three classification metrics together:
classification_metrics <- metric_set(accuracy, mcc, f_meas)

classification_metrics(two_class_example, truth = truth, estimate = predicted)

# The Matthews correlation coefficient and the F1 score both summarize the confusion matrix,
# but compared to `mcc()`, which measures the quality of both positive and negative examples,
# the `f_meas()` metric emphasizes the positive class, i.e., the event of interest.

# For binary classification data sets like this example, "yardstick" functions have a standard
# argument called `event_level` to distinguish positive and negative levels.

# The default (which we used in this code) is that the *first* level of the outcome factor
# is the event of interest.


# An example where the second level is the event of interest:
f_meas(two_class_example, truth = truth, estimate = predicted, event_level = "second")
# 0.826

# In the output, the value of the `.estimator` column is `"binary"`, indicating that the
# standard formula for binary classes is used.


# Other classification metrics use the predicted probabilities as inputs rather than
# the hard class predictions.

# The Receiver Operating Characteristic (ROC) curve computes the sensitivity and specificity
# over a continuum of different event thresholds.

# The rpedicted class column is not used.

# There are two "yardstick" functions for thsi method:
# - `roc_curve()` computes the data points that make up the ROC curve
# - `roc_auc()` computes the Area Under the (ROC) Curve


# The `...` placeholder is used to pass the class probability column.
# For a two-class problem, the probability column for the event of interest is
# passed into the function:
two_class_curve <- roc_curve(two_class_example, truth, Class1)

print(two_class_curve)

roc_auc(two_class_example, truth, Class1)
# 0.939


# The `two_class_curve` object can be used in a `ggplot()` call or with `autoplot()`:
autoplot(two_class_curve)

graphics.off()

# If the curve was close to the diagonal line,
# the model's predictions would be no better than
# random guessing or a coin toss.


# Other functions that use probability estimates are:
# - `gain_curve()`
# - `lift_curve()`
# - `pr_curve()`


## 9.4 Multiclass classification metrics ----

# We explore the `hpc_cv` data which has four classes:
data("hpc_cv")
tibble(hpc_cv)

# The data includes a `Resample` column because these results are for out-of-sample
# predictions associated with 10-fold cross validation.

# The functions for metrics that use the discrete class predictions are identical
# to their binary counterparts:

accuracy(hpc_cv, truth = obs, estimate = pred)
# 0.709

mcc(hpc_cv, truth = obs, estimate = pred)
# 0.515


# The `.estimator` column now reads `"multiclass"`.


# There are wrapper methods that apply sensitivity to multi-class outcomes.

# - Macro-averaging: Compute a set of one-versus all metrics using the standard two-class
#   statistics and take their average.

# - Macro-weighted averaging: Do the same as above but weight the average by the number of
#   samples in each class.

# - Micro-averaging: Compute the contribution for each class, aggregate these contributions,
#   then compute a single metric from the aggregates.


# The usual two-class calculation for the sensitivity is the ratio of the number of 
# correctly predicted events divided by the number of true events.

# The manual calculations for these averaging methods are:

class_totals <- hpc_cv |> 
  count(obs, name = "totals") |> 
  mutate(
    class_wts = totals / sum(totals)
  )

print(class_totals)

cell_counts <- hpc_cv |> 
  group_by(obs, pred) |> 
  count() |> 
  ungroup()


# Compute the four sensitivites using 1-vs-all
one_versus_all <- cell_counts |> 
  filter(obs == pred) |> 
  full_join(y = class_totals, by = "obs") |> 
  mutate(
    sens = n / totals
  )

print(one_versus_all)


# Three different estimates:
one_versus_all |> 
  summarize(
    macro = mean(sens),
    macro_wts = weighted.mean(sens, class_wts),
    micro = sum(n) / sum(totals)
  )


# Thankfully, "yardstick" functions automatically apply these methods when
# the `estimator` argument is specified:
sensitivity(hpc_cv, obs, pred, estimator = "macro")
# 0.560

sensitivity(hpc_cv, obs, pred, estimator = "macro_weighted")
# 0.709

sensitivity(hpc_cv, obs, pred, estimator = "micro")
# 0.709


# An ROC curve for multi-class outcomes can be determined when all of the class
# probability columns are given to the function:
roc_auc(hpc_cv, obs, VF, F, M, L)
# 0.829

# Macro-weighted averaging is also available as an option for the ROC AUC:
roc_auc(hpc_cv, obs, VF, F, M, L, estimator = "macro_weighted")
# 0.868


# All of these performance methods can be computed with "dplyr" groupings.

# We can pass the data grouped by the values of the `Resample` column to the
# performance functions:
hpc_cv |> 
  group_by(Resample) |> 
  accuracy(truth = obs, estimate = pred)

# We have calculated the accuracy for all 10 folds.


# The groupings translate to the `autoplot()` methods:

# Four 1-vs-all ROC curves for each fold
hpc_cv |> 
  group_by(Resample) |> 
  roc_curve(obs, VF, F, M, L) |> 
  autoplot()

graphics.off()

# The plots show that the different groups perform similarly, but that the `VF`
# class is predicted better than the `F` or `M` classes, since the ROC curves for the
# `VF` class are closer to the top-left corner.


# END
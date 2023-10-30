# Chapter 20 - Ensembles of models ----

# A model ensemble aggregates the predictions of multiple learners 
# into one prediction.


# Popular methods to create ensemble models are bagging,
# random forests, and boosting.


# One of the earliest methods to create ensemble models is *model stacking*.


# Model stacking combines the predictions for multiple models of any type.


# The "stacks" package helps stack predictive models:

# 1. Assemble the training set of hold-out predictions (produced via resampling).
# 2. Create a model to blend these predictions.
# 3. For each member of the ensemble, fit the model on the original training set.


# Consider the multilayer perceptron (MLP), a.k.a. neural network.

# Use the term *candidate members* to describe the possible model
# configurations (of all model types) that might be included
# in the stacking ensemble.


## 20.1 Creating the training set for stacking ----

# The first step relies on the assessment set predictions from a resampling
# scheme with multiple splits.

# For each data point in the training set, stacking requires an out-of-sample
# prediction.


# For regression models, this is the predicted outcome.

# For classification models, the predicted classes or the probabilities.


# For a set of models, a data set is assembled where rows are the training
# set and columns are the out-of-sample predictions from the set of multiple models.


# In Chapter 15, we used five repeats of 10-fold cross-validation to resample the data.

# This resampling scheme generates five assessment set predictions for each
# training set sample.

# Multiple out-of-sample predictions can occur in several other resampling techniques,
# e.g., bootstrapping.


# For the purpose of stacking, any replicate predictions for a data point in the
# training set are averaged so that there is a single prediction per training set
# sample per candidate member.


# Simple validation sets can also be used with stacking since "tidymodels"
# considers this to be a single resample.


# For the concrete example, the training set used for model stacking has columns
# for all the candidate tuning parameter results.


# There is a single column for the bagged tree model since it has no tuning parameters.

# Recall that MARS was tuned over a single parameter (the product degree)
# with two possible configurations, so this model is represented by two columns.


# For classification models, the candidate prediction columns would be predicted
# class probabilities.

# Since these columns add to one for each model, the probabilities for one 
# of the classes can be left out.


# The first step to stacking is to assemble the assessment set predictions
# for the training set from each candidate model.


# We use the "stacks" package and create an empty data stack with `stacks()`
# and then add the candidate models.

# We use the racing results:
print(race_results)
# A workflow set/tibble: 12 * 4
# wflow_id info option result


# The syntax is:
library(tidymodels)
tidymodels_prefer()
library(stacks)

concrete_stack <- stacks() |> 
  add_candidates(race_results)


print(concrete_stack)
# A data stack with 12 model definitions and 18 candidate members:


# Recall that racing methods are more efficient since they do not evaluate
# all possible configurations on all resamples.

# Stacking requires all candidate members have the complete set of resamples.

# `add_candidates()` includes only the model configurations that have complete results.


## 20.2 Blend the predictions ----

# The training set predictions and the corresponding observed outcome data
# are used to create a *meta-learning model* where the assessment set predictions
# are the predictors of the observed outcome data.


# Meta-learning can be accomplished using any model.

# The most commonly used model is a regularized generalized linear model,
# which encompasses linear, logistic, and multinomial models.


# Regularization via the lasso penalty (Tibshirani 1996), which uses shrinkage
# to pull points toward a central value, has several advantages:

# - Using the lasso penalty can remove candidates (and sometimes whole model types)
#   from the ensemble.

# - The correlation between ensemble candidates tends to be very high, and regularization alleviates this.


# When a linear model is sued to blend predictions, it can be helpful to constrain
# the blending coefficients to be nonnegative.


# This is the default for the "stacks" package.


# Since our outcome is numeric, linear regression is used for the metamodel.

# We fit the metamodel with:
set.seed(2001)
ens <- blend_predictions(concrete_stack)

# This evaluates the meta-learning model over a predefined grid of lasso penalty
# values and uses an internal resampling method to determine the best value.


# The `autoplot()` method visualizes the penalization:
autoplot(ens) +
  theme_bw() +
  labs(
    title = " Results of using the autoplot() method on the blended stacks object"
  )

graphics.off()

# the top panel shows the average number of candidate ensemble members retained
# by the meta-learning model.

# The number of members is fairly constant, and, as it increases, the RMSE also increases.


# The default range may not have been the appropriate choice here.
# We pass an additional option to evaluate the meta-learning model with
# larger penalties:
set.seed(2002)
ens <- blend_predictions(
  concrete_stack,
  penalty = 10^seq(from = -2, to = -0.5, length = 20)
)


# Now we see a range where the ensemble model becomes worse than with the
# first blend (but not by much).
# The R-squared values increase with more members and larger penalties.
autoplot(ens) +
  theme_bw() +
  labs(
    title = "The results of using the autoplot() method on the updated blended stacks object"
  )

graphics.off()


# When blending predictions using a regression model, it is common to constrain
# the blending parameters to be nonnegative.

# For these data, this constraining has the effect of eliminating many
# of the potential ensemble members.

# Even at fairly low penalties, the ensemble is limited to a fraction of the
# original eighteen.


# The penalty value associated with the smallest RMSE was 0.051.

# Printing the object shows the details of the meta-learning model:
print(ens)


# The regularized linear regression meta-learning model contained seven blending
# coefficients across four types of models.

# The `autoplot()` emthod can be used to show the contributions of each model type:
theme_set(theme_bw())

autoplot(ens, "weights") +
  geom_text(mapping = aes(x = weight + 0.01, label = model), hjust = 0) +
  theme(legend.position = "none") +
  lims(x = c(-0.01, 0.8)) +
  labs(
    title = "Blending coefficients for the stacking ensemble"
  )

graphics.off()

# The boosted tree and neural network models have the largest contributions
# to the ensemble.


## 20.3 Fit the member models ----

# The ensemble contains seven candidate members, and we now know how their
# predictions can be blended into a final prediction ensemble.

# However, these individual model fits have not yet been created.

# To use the stacking model, seven additional model fits are required.

# These use the entire training set with the original predictors.

# The "stacks" package has the function `fit_members()` that trains and
# returns these models:
ens <- fit_members(ens)

# This updates the stacking object with the fitted workflow object for
# each member.

# At this point, the stacking model can be used for prediction.


## 20.4 Test set results ---

# Since the blending process used resampling, we can estimate that the 
# ensemble with seven members has an estimated RSME of 4.12.

# Recall from Chapter 15 that the best boosted tree had a test set RMSE
# of 3.41.

# We use `predict()` to find out how the ensemble model compares to the test set:
reg_metrics <- metric_set(rmse, rsq)

ens_test_pred <- predict(ens, concrete_test) |> 
  bind_cols(concrete_test)

ens_test_pred |> 
  reg_metrics(compressive_strength, .pred)
# rmse: 3.37
# rsq: 0.956

# This is moderately better than our best single model.

# It is fairly common for stacking to produce incremental benefits
# when compared to the best single model.


# END
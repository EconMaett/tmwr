# Chapter 15 - Screening many models ----

library(tidymodels)
tidymodels_prefer()

# For projects with new data sets that have not yet been well understood, 
# a data practitioner may need to screen many combinations of models and preprocessors.

# It is common to have little or no *a priori* knowledge about which method will work best
# with a novel data set.


# A good strategy is to spend some initial effort trying a variety of modeling approaches,
# determine what works best, then invest additional time tweaking/optimizing a small set of models.


# Workflow sets provide a user interface to create and manage this process.


## 15.1 Modeling concrete mixture strength ---

# To demonstrate how to screen multiple model workflows, we use the concrete mixture
# data from *Applied Predictive Modeling* (Kuhn & Johnson, 2013).

# Chapter 10 of this book demonstrates models to predict the *compressive strength*
# of concrete mixtures using the ingredients as predictors.

# A wide variety of models will be evaluated with different predictor sets and
# preprocessing needs.

# Workflow sets make such a process of large scale testing for models easier.


# First, we define the data splitting and resampling schemes.
data("concrete", package = "modeldata")

glimpse(concrete)
# 1,030 rows, 9 columns
# cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age, compressive_strength

# The outcome is `compressive_strength`.

# The `age` predictor is the age of the concrete sample at testing in time,
# since concrete strengthens over time.

# The rest of the predictors are concrete components like `cement` or `water`
# in units of *kilograms per cubic meter*.


# Note that for some cases in this data set, the same concrete formula was
# tested multiple times.
# We want to exclude these replicate mixtures as individual data points might be distributed
# across both the training and test set and might artificially inflate our performance estimates.


# We use the mean compressive strength per concrete mixture for modeling:

concrete <- concrete |> 
  group_by(across(-compressive_strength)) |> 
  summarise(
    compressive_strength = mean(compressive_strength),
    .groups = "drop"
  )

nrow(concrete)  
# 992


# We split the data using the default 3:1 ration of training-to-test
# and resample the training set using five repeats of 10-fold cross-validation:
set.seed(1501)

concrete_split <- initial_split(concrete, strata = compressive_strength)
concrete_train <- training(concrete_split)
concrete_test <- testing(concrete_split)

set.seed(1502)
concrete_folds <- vfold_cv(concrete_train, strata = compressive_strength, repeats = 5)


# Models like neural networks, KNN, and SVMs require predictors to be normalized,
# that is centered and scaled.

# Other models require preprocessing steps such as a traditional 
# response surface design model expansion such as
# quadratic and two-way interactions

# For these purposes, we create two recipes:
normalized_rec <- recipe(compressive_strength ~ ., data = concrete_train) |> 
  step_normalize(all_predictors())


poly_recipe <- normalized_rec |> 
  step_poly(all_predictors()) |> 
  step_interact(~ all_predictors():all_predictors())


# For the models, we use the "parsnip" addin to create a set of model specifications:
library(rules)
library(baguette)

linear_reg_spec <- linear_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet")

nnet_spec <- mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |> 
  set_engine("nnet", MaxNWts = 2600) |> 
  set_mode("regression")

mars_spec <- mars(prod_degree = tune()) |> # <- use GCV to choose terms
  set_engine("earth") |> 
  set_mode("regression")

svm_r_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) |> 
  set_engine("kernlab") |> 
  set_mode("regression")

svm_p_spec <- svm_poly(cost = tune(), degree = tune()) |> 
  set_engine("kernlab") |> 
  set_mode("regression")

knn_spec <- nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) |> 
  set_engine("kknn") |> 
  set_mode("regression")

cart_spec <- decision_tree(cost_complexity = tune(), min_n = tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")

bag_cart_spec <- bag_tree() |> 
  set_engine("rpart", times = 50L) |> 
  set_mode("regression")

rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |> 
  set_engine("ranger") |> 
  set_mode("regression")

xgb_spec <- boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(),
                       min_n = tune(), sample_size = tune(), trees = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

cubist_spec <- cubist_rules(committees = tune(), neighbors = tune()) |> 
  set_engine("Cubist")


# The analysis in M. Kuhn and Johnson (2013) specifies that the neural network should have
# up to 27 hidden units in the layer.
# `extract_parameter_set_dials()` extracts the parameter set, 
# which we modify to have the correct parameter range:
nnet_param <- nnet_spec |> 
  extract_parameter_set_dials() |> 
  update(hidden_units = hidden_units(c(1, 27)))


# A workflow set helps us to match these models to their recipes, tune them, and
# evaluate their performance.


## 15.2 Crating the workflow set ----

# A workflow set take a named list of preprocessors and model specifications and
# combines them into an object with multiple workflows.

# There are three kinds of preprocessors:

# - A standard R formula
# - A recipe object (prior to estimation/prepping)
# - A "dplyr"-style selector to choose outcomes and predictors.


# As an example we create a workflow set that combines the recipe that only
# standardizes the predictors to the nonlinear models that require the predictors
# to be in the same units:
normalized <- workflow_set(
  
  preproc = list(
    normalized = normalized_rec
    ),
  
  models = list(
    SVM_radial = svm_r_spec,
    SVM_poly = svm_p_spec,
    KNN = knn_spec,
    neural_network = nnet_spec
  )
)

print(normalized)

# Because there is only a single preprocessor, `worklfow_set()` created a set of workflows
# where all `wflow_id`s have the prefix `normalized_`.

# We can call `mutate()` to change the `wflow_id`.

# The `info` column contains a tibble with some identifiers and the workflow object.
# The workflow can be extracted:
normalized |> 
  extract_workflow(id = "normalized_KNN")


# The `option` column is a placeholder for any arguments to use when we evaluate
# the workflow.

# To add the neural network parameter object:
normalized <- normalized |> 
  option_add(
    param_info = nnet_param, 
    id = "normalized_neural_network"
    )

print(normalized)

# When a function from the "tune" or "finetune" package is used to tune or resample the
# workflows, the argument in the `option` column will be used.


# The `result` column is a placeholder for the output of the tuning or resampling functions.


# For the other nonlinear models, we create another workflow set that uses
# "dplyr" selectors for the outcome and predictors:
model_vars <- workflow_variables(
  outcomes = compressive_strength,
  predictors = everything()
)

no_pre_proc <- workflow_set(
  
  preproc = list(
    simple = model_vars
  ),
  
  models = list(
    MARS = mars_spec,
    CART = cart_spec,
    CART_bagged = bag_cart_spec,
    RF = rf_spec,
    boosting = xgb_spec,
    Cubist = cubist_spec
  )
)


# Finally, we assemble the set that uses nonlinear terms and interactions
# with the appropriate models:
with_features <- workflow_set(
  
  preproc = list(
    full_quad = poly_recipe
  ),
  
  models = list(
    linear_reg = linear_reg_spec,
    KNN = knn_spec
  )
)


# These objects are tibbles with the extrac class `"workflow set"`
class(with_features)
# "workflow_set" "tbl_df" "tbl" "data.frame"

# Row binding does not affect the state of the sets and the result is again a workflow set:
all_workflows <- bind_rows(
  no_pre_proc, normalized, with_features
  ) |> 
  mutate(
    wflow_id = gsub("(simple_)|(normalized_)", "", wflow_id)
  )

print(all_workflows)


## 15.3 Tuning and evaluating the models ----

# Almost all members of `all_workflows` contain tuning parameters.

# To evaluate their performance, we use the standard tuning or resampling functions
# like `tune_grid()`.

# `workflow_map()` applies the same function to all workflows in a workflow set.
# The default is `tune_grid()`.


# We apply grid search to each workflow using up to 25 different parameter candidates.

# We use the same resampling and control objects for each workflow, 
# along with a grid size of 25.

# `worklfow_map()` has a `seed` argument that ensures reproducibility:
grid_ctrl <- control_grid(
  save_pred = TRUE,
  parallel_over = "everything",
  save_workflow = TRUE
)

# Caution: This takes a two hours to run!
grid_results <- all_workflows |> 
  workflow_map(
    seed = 1503,
    resamples = concrete_folds,
    grid = 25,
    control = grid_ctrl
  )

# The results show that the `option` and `result` columns are updated.
print(grid_results)


# The `option` column contains all options used in `worklfow_map()`.
# The `result` column contains `tune[+]` and  `rsmp[+]` indicating that the
# object had no issues.

# A value such as `tune[x]` indicates that all of the models failed.


# `rank_results()` will order the models in `grid_results` by a performance metric.
# The default is the first metric in the metric set, in our instance the RMSE.
grid_results |> 
  rank_results() |> 
  filter(.metric == "rmse") |> 
  select(model, .config, rmse = mean, rank)


# By default, the function ranks all candidate sets.
# The `select_best` option can be used to rank the models using their best tuning 
# parameter combination.

# Visualize the best results for each model with the `autoplot()` method,
# using the `select_best` argument.
autoplot(
  grid_results,
  rank_metric = "rmse", # <- how to order models
  metric = "rmse", # <- which metric to visualize
  select_best = TRUE # <- one point per workflow
  ) +
  geom_text(mapping = aes(y = mean - 1/2, label = wflow_id), angle = 90, hjust = 1) +
  lims(y = c(3.5, 9.5)) +
  theme(legend.position = "none") +
  theme_bw()

graphics.off()


# To see the tuning parameter results for a specific model, 
# use the assign a `wflow_id` to the `id` argument:
autoplot(grid_results, id = "Cubist", metric = "rmse") +
  theme_bw()

graphics.off()


# The workflow sets defined in this exercise to the concrete mixture data
# fit a total of 12,600 models.

# Using two workers in parallel, the estimation process takes 1.9 ours to complete.


## 15.4 Efficiently screening models ----

# Because the workflows from before take almost two hours to run, we want to use
# a racing approach to speed things up.

# `workflow_map()` is used to apply a racing approach.

# After we pipe the workflow set, 
# the argument we use inside `workflow_map()` is applied to all workflows.
# We can use `"tune_race_anova"`.

# We also pass an appropriate control object.
library(finetune)

race_ctrl <- control_race(
  save_pred = TRUE,
  parallel_over = "everything",
  save_workflow = TRUE
)

race_results <- all_workflows |> 
  workflow_map(
    fn = "tune_race_anova",
    seed = 1503,
    resamples = concrete_folds,
    grid = 25,
    control = race_ctrl
  )


# The elements of the `result` column now show a value of `race[+]`,
# indicating a different type of object:
print(race_results)


autoplot(
  object = race_results,
  rank_metric = "rmse",
  metric = "rmse",
  select_best = TRUE
  ) +
  geom_text(mapping = aes(y = mean - 1/2, label = wflow_id), angle = 90, hjust = 1) +
  lims(y = c(3.0, 9.5)) +
  theme(legend.position = "none") +
  theme_bw()


graphics.off()


# Overall, the racing approach estimated a total of 1,050 models, just 8.33% of the
# full set of 12,600 models in the full grid.

# As a result, the racing approach was almost five times faster.

# We can compare the results if we first rank them, merge them, and plot 
# them against one another:
matched_results <- rank_results(race_results, select_best = TRUE) |> 
  select(wflow_id, .metric, race = mean, config_race = .config) |> 
  inner_join(
    rank_results(grid_results, select_best = TRUE) |> 
      select(wflow_id, .metric, complete = mean, config_complete = .config, model),
    by = c("wflow_id", ".metric")
  ) |> 
  filter(.metric == "rmse")


library(ggrepel)

matched_results |> 
  ggplot(mapping = aes(x = complete, y = race)) +
  geom_abline(lty = 3) +
  geom_point() +
  geom_text_repel(mapping = aes(label = model)) +
  coord_obs_pred() +
  labs(
    title = "",
    x = "Complete Grid RMSE",
    y = "Racing RMSE"
  ) +
  theme_bw()

graphics.off()


# The racing approach selected the same candidate parameters as the complete grid for
# only 41.67% of the models, but the performance metrics of both approaches are nearly equal.

# The correlation of RMSE values is 0.968 and the rank correlation is 0.951.

# This indicates that, within a model, there are multiple tuning parameter combinations
# with similar results.


## 15.5 Finalizing a model ----

# We finalize the process by fitting the chosen models on the training set.

# Since the boosted tree model worked well, we extract it from the set,
# update the parameters with the best settings, and fit it to the training set:
best_results <- race_results |> 
  extract_workflow_set_result("boosting") |> 
  select_best(metric = "rmse")

print(best_results)


boosting_test_results <- race_results |> 
  extract_workflow("boosting") |> 
  finalize_workflow(best_results) |> 
  last_fit(split = concrete_split)


# We can see the test set metrics results and visualize the predictions:
collect_metrics(boosting_test_results)


boosting_test_results |> 
  collect_predictions() |> 
  ggplot(mapping = aes(x = compressive_strength, y = .pred)) +
  geom_abline(color = "gray50", lty = 2) +
  geom_point(alpha = 0.5) +
  coord_obs_pred() +
  labs(
    title = "",
    x = "observed", 
    y = "predicted"
  ) +
  theme_bw()

graphics.off()

# END
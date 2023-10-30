# Chapter 18 - Explaining models and predictions ----

# Some models, like linear regression, provide coefficients that
# are easy to interpret.

# Other models, like random forests, can capture highly nonlinear
# behavior, and are less easy to interpret.


# There are two types of model explanations: global and local.

# Global model explanations provide an overall understanding aggregated
# over a whole set of observations;

# Local model explanations provide information about a prediction for 
# a signle observation.


## 18.1 Software for model explanations ----

# The "tidymodels" framework does not intself ocntain software for model
# explanations.

# Instead, models trained and evaluated with tidymodels can be explained with other,
# supplementary software in R packages such as "lime", "vip", and "DALEX".

# We often choose:

# - "vip" functions when we want to use *model-based* methods to take advantage
#    of model structure (and are often faster).

# - "DALEX" functions when we want to use *model-agnostic* methods that
#    can be applied to any model.


# In chapters 10 and 11, we compared predictions for the Ames housing data
# with a linear model with interactions and with a random forest model.
library(tidymodels)
tidymodels_prefer()

data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

ames_rec <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude, 
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

rf_model <- rand_forest(trees = 1000) |> 
  set_engine("ranger") |> 
  set_mode("regression")

rf_wflow <- workflow() |> 
  add_formula(
    Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude
    ) |> 
  add_model(rf_model)

rf_fit <- rf_wflow |> 
  fit(data = ames_train)

set.seed(1001)
ames_folds <- vfold_cv(ames_train, v = 10)

keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

set.seed(1003)

rf_res <- rf_wflow |> 
  fit_resamples(resamples = ames_folds, control = keep_pred)


# We want to build model-agnostic explainers for both models to find out
# why they make these predictions.

# We can use the add-on package for "DALEX", "DALEXtra" that supports "tidymodels".

# We first prepare the appropriate data and then create an *explainer*
# for each model:
library(DALEXtra)

vip_features <- c("Neighborhood", "Gr_Liv_Area", "Year_Built", 
                  "Bldg_Type", "Latitude", "Longitude")

vip_train <- ames_train |> 
  select(all_of(vip_features))

explainer_lm <- explain_tidymodels(
  model = lm_fit,
  data = vip_train,
  y = ames_train$Sale_Price,
  label = "lm + interactions",
  verbose = FALSE
)

explainer_rf <- explain_tidymodels(
  model = rf_fit,
  data = vip_train,
  y = ames_train$Sale_Price,
  label = "random forest",
  verbose = FALSE
)


# Using a separate model explanation algorithm for a linear model
# may make sense if it includes splines and interaction terms.


# We can quantify global or local model explanations either in terms of:

# - original, basic predictors as they existed without significant feature engineering trasformations

# - derived features, such as those created via dimensionality reduction
#   or interaction and spline terms.


## 18.2 Local explanations ----

# Local explanations provide information about a prediction for a single
# observation.

# For example, let's consider an older duplex in the North Ames neighborhood:
duplex <- vip_train[120, ]

print(duplex)


# There are multiple approaches to understand why a model predicts a given
# price for this duplex.

# One is a break-down explanation, implemented with the "DALEX" function
# `predict_parts()`;
# It computes how contributions attributed to individual features change 
# the mean model's prediction for a particular observation.

# For the linear model, the duplex status, `Bldg_Type = 3`,
# size, longitude, and age all contribute the most to the price being
# driven down from the intercept:
lm_breakdown <- predict_parts(explainer = explainer_lm, new_observation = duplex)

print(lm_breakdown)


# Since this linear model was trained with spline terms for latitude and longitude,
# the contribution to price for `Longitude` shown here combines the effects
# of all of its individual spline terms.

# The contribution is in terms of the original `Longitude` feature,
# not the derived spline features.


# The most important features are slightly different for the random
# forest model, with the size, age, and duplex status being most important:
rf_breakdown <- predict_parts(explainer = explainer_rf, new_observation = duplex)

print(rf_breakdown)


# Model break-down explanations like these depend on the *order* of the features.


# If we choose the *order* for the random forest model explanation
# to be the same as the default for the linear model, chosen via a heuristic,
# we can change the relative importance of the features:
predict_parts(
  explainer = explainer_rf,
  new_observation = duplex,
  order = lm_breakdown$variable_name
)


# We can use the fact that these break-down explanations change based on
# order to compute the most important features over all (or many) possible orderings.

# This is the idea behind Shapely Additive Explanations (SHAP), 
# where the average contributions of features are computed
# under different combinations or "coalitions" of feature orderings.


# Compute the SHAP attributions for our duplex, using `B = 20` random orderings:
set.seed(1801)

shap_duplex <- predict_parts(
  explainer = explainer_rf,
  new_observation = duplex,
  type = "shap",
  B = 20
)


# We can use the default plot method from "DALEX" by calling
# `plot(shap_duplex)`:
plot(shap_duplex)

graphics.off()


# or we can access the underlying data and create a custom plot.

# Create box plots that display the distributions across all the orderings
# we tried, and the bars that display the average attribution for each
# feature:
library(forcats)

theme_set(theme_bw())

shap_duplex |> 
  group_by(variable) |> 
  mutate(
    mean_val = mean(contribution)
  ) |> 
  ungroup() |> 
  mutate(
    variable = forcats::fct_reorder(variable, abs(mean_val))
  ) |> 
  ggplot(mapping = aes(x = contribution, y = variable, fill = mean_val > 0)) +
  geom_col(
    data = ~ distinct(., variable, mean_val),
    mapping = aes(x = mean_val, y = variable),
    alpha = 0.5
  ) +
  geom_boxplot(width = 0.5) +
  theme(legend.position = "none") +
  scale_fill_viridis_d() +
  labs(
    title = "Shapley additive explanations from the random forest model for a duplex property",
    y = NULL
  )

graphics.off()


# What about a different observation in our data set?

# Let's look at a larger, newer one-family home in the Gilbert neighborhood:
big_house <- vip_train[1269, ]

print(big_house)


# We can compute SHAP average attributions for this house in the same way:
set.seed(1802)

shap_house <- predict_parts(
  explainer = explainer_rf,
  new_observation = big_house,
  type = "shap",
  B = 20
)


# We can use the default `plot()` method:
plot(shap_house)

graphics.off()


# Or use `ggplot()`:
library(forcats)

theme_set(theme_bw())

shap_house |> 
  group_by(variable) |> 
  mutate(
    mean_val = mean(contribution)
  ) |> 
  ungroup() |> 
  mutate(
    variable = forcats::fct_reorder(variable, abs(mean_val))
  ) |> 
  ggplot(mapping = aes(x = contribution, y = variable, fill = mean_val > 0)) +
  geom_col(
    data = ~ distinct(., variable, mean_val),
    mapping = aes(x = mean_val, y = variable),
    alpha = 0.5
  ) +
  geom_boxplot(width = 0.5) +
  theme(legend.position = "none") +
  scale_fill_viridis_d() +
  labs(
    title = "Shapley additive explanations from the random forest model for a one-family home in Gilbert",
    y = NULL
  )

graphics.off()



## 18.3 Global explanations ----

# Global model explanations, also called global feature importance or
# variable importance, help us understand which features are most important
# in driving predictions of the linear and random forest models overall,
# aggregated over the whole training set.


# While the previous section addressed which variables or features are 
# most important in predicting sale price for an individual home,
# global feature importance addresses the most important variables
# for a model in aggregate.


# One way to compute variable importance is to *permute* the features
# (Breiman, 2001a). We can permute or shuffle the values of a feature,
# predict from the model, and then measure how much worse the model
# fits the data compared to before.


# If shuffling a column causes a large degradation in model performance,
# this is an important feature.

# If shuffling a column's values does not make much difference how the model
# performs, this is an unimportant variable.


# This approach can be applied to any kind of model, so this is
# a *model agnostic* approach.

# The results are easy to interpret.


# Using "DALEX", we compute this kind of variable importance via the
# `model_parts()` function.
set.seed(1803)

vip_lm <- model_parts(
  explainer = explainer_lm, 
  loss_function = loss_root_mean_square
  )


set.seed(1804)

vip_rf <- model_parts(
  explainer = explainer_rf, 
  loss_function = loss_root_mean_square
  )

# Again, we could use the default `plot()` method from "DALEX"
# by calling
plot(vip_lm, vip_rf)

graphics.off()


# or we can access the underlying data
# and use `ggplot()`:

# We create a custom function for plotting:
ggplot_imp <- function(...) {
  obj <- list(...)
  metric_name <- attr(x = obj[[1]], which = "loss_name")
  metric_lab <- paste(metric_name, "after permutations\n(higher indicates more important)")
  
  full_vip <- bind_rows(obj) |> 
    filter(variable != "_baseline_")
  
  
  perm_vals <- full_vip |> 
    filter(variable == "_full_model_") |> 
    group_by(label) |> 
    summarise(
      dropout_loss = mean(dropout_loss)
    )
  
  
  p <- full_vip |> 
    filter(variable != "_full_model_") |> 
    mutate(
      variable = forcats::fct_reorder(variable, dropout_loss)
    ) |> 
    ggplot(mapping = aes(x = dropout_loss, y = variable))
  
  if (length(obj) > 1) {
    p <- p +
      facet_wrap(vars(label)) +
      geom_vline(
        data = perm_vals,
        mapping = aes(xintercept = dropout_loss, color = label),
        linewidth = 1.4,
        lty = 2,
        alpha = 0.7
      ) +
      geom_boxplot(mapping = aes(color = label, fill = label), alpha = 0.2)
  } else {
    p <- p +
      geom_vline(
        data = perm_vals,
        mapping = aes(xintercept = dropout_loss),
        linewidth = 1.4,
        lty = 2,
        alpha = 0.7
      ) +
      geom_boxplot(fill = "#91CBD765", alpha = 0.4)
  }
  
  p + 
    theme(legend.position = "none") +
    labs(
      x = metric_lab,
      y = NULL,
      fill = NULL,
      color = NULL
    )
}


# Call `ggplot_imp(vip_lm, vip_rf)`:
ggplot_imp(vip_lm, vip_rf) +
  ggtitle("Global explainer for the random forest and linear regression models")

graphics.off()

# The dashed line in each panel shows the RSME for the full model,
# either the linear model or the random forest model.

# Features to the right are more important,
# because permuting them results in higher RMSE.


# There is a lot of interesting information to learn form this plot;
# for example, neighborhood is important in the linear model with interactions/splines
# but the second least important feature for the random forest model.

# Note: This may be because the random forest is able to
# predict the neighborhood from other features of the houses,
# like size etc.



## 18.4 Building global explanations from local explanations ----

# We have previously created local model explanations for a single observation,
# with Shapely Additive Explanations (SHAE).


# We have also created global model explanations for whole data sets,
# with permuting features.


# It is possible to build global model explanations by aggregating
# local model explanations, aw with *partial dependence profiles*.


# Partial dependence profiles show how the expected value of a model prediction,
# like the prediction of a home in Ames, changes as a function of a feature,
# like the age or gross living area.


# We can build such a profile by aggregating profiles for individual observations.

# A profile showing how an individual observation's prediction changes as a
# function of a given feature is called an
# Individual Conditional Expectation (ICE) profile or a
# *ceteris paribus* (CP) profile.


# We can compute indivdual such profiles (for 500 of the observations
# in our training set) and then aggregate them using the "DALEX"
# function `model_profile()`:
set.seed(1805)

pdp_age <- model_profile(
  explainer = explainer_rf,
  N = 500,
  variables = "Year_Built"
)


# Create another custom plotting function:
ggplot_pdp <- function(obj, x) {
  
  p <- as_tibble(obj$agr_profiles) |> 
    mutate(
      `_label_` = stringr::str_remove(`_label_`, "^[^_]*_")
    ) |> 
    ggplot(mapping = aes(x = `_x_`, y = `_yhat_`)) +
    geom_line(
      data = as_tibble(obj$cp_profiles),
      mapping = aes(x = {{ x }}, group = `_ids_`),
      linewidth = 0.5,
      alpha = 0.05,
      color = "gray50"
    )
  
  
  num_colors <- n_distinct(obj$agr_profiles$`_label_`)
  
  
  if (num_colors > 1) {
    p <- p +
      geom_line(mapping = aes(color = `_label_`), linewidth = 1.2, alpha = 0.8)
  } else {
    p <- p +
      geom_line(color = "midnightblue", linewidth = 1.2, alpha = 0.8)
  }
  
  
  return(p)
}


# Call `ggplot_pdp(pdp_age, Year_Built)` to see the nonlinear behavior
# of the random forest model:
ggplot_pdp(pdp_age, Year_Built) +
  labs(
    title = "Partial dependence profiles for the random forest model focusing on the year built predictor",
    x = "Year built",
    y = "Sale Price (log)",
    color = NULL
  )

graphics.off()

# Sale price for houses built in different years is mostly flat,
# with a modest rise after about 1960.

# Partial dependence profiles can be computed for any other feature in the model,
# and also for groups in the data such as `Bldg_Type`.


# Let's use 1,000 observations for these profiles:
set.seed(1806)


# Caution: This takes a munute to run
pdp_liv <- model_profile(
  explainer = explainer_rf,
  N = 1000,
  variables = "Gr_Liv_Area",
  groups = "Bldg_Type"
)


ggplot_pdp(obj = pdp_liv, x = Gr_Liv_Area) +
  scale_x_log10() +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Partial dependence profiles for the random forest model focusing on building types and gross living area",
    x = "Gross living area",
    y = "Sale price (log)",
    color = NULL
  )

graphics.off()


# We have the option of using the default `plot()` method for "DALEX" plots
plot(pdp_liv)

graphics.off()

# But since we are making plots with the underlying data, we can even
# facet by one of the features to visualize if the predictions change
# differently and highlight the imbalance in these subgroups:
as_tibble(pdp_liv$agr_profiles) |> 
  mutate(
    Bldg_Type = stringr::str_remove(`_label_`, "random forest_")
  ) |> 
  ggplot(mapping = aes(x = `_x_`, y = `_yhat_`, color = Bldg_Type)) +
  geom_line(
    data = as_tibble(pdp_liv$cp_profiles),
    mapping = aes(x = Gr_Liv_Area, group = `_ids_`),
    linewidth = 0.5, alpha = 0.1, color = "gray50"
  ) +
  geom_line(linewidth = 1.2, alpha = 0.8, show.legend = FALSE) +
  scale_x_log10() +
  facet_wrap(~ Bldg_Type) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Partial dependence profiles for the random forest model focusing on building types and gross living area using facets",
    x = "Gross living area",
    y = "Sale price (log)",
    color = NULL
  )


graphics.off()


# Note that there is no one correct approach for building model explanations.


## 18.5 Back to beans ----

# For the example data set of dry bean morphology measures predicting bean type,
# we saw that partial least squares (PLS) dimensionality reduction in combination
# with a regularized discriminant analysis model produced good results.

# We create a model-agnostic ex-plainer to compute global model explanations
# with `model_parts()`
library(tidymodels)
tidymodels_prefer()

library(beans)

set.seed(1601)

bean_split <- initial_validation_split(beans, strata = class, prop = c(0.75, 0.125))

print(bean_split)


# Return data frames:
bean_train <- training(bean_split)
bean_test <- testing(bean_split)
bean_validation <- validation(bean_split)


set.seed(1602)
# Return an 'rset' object to use with the tune functions:
bean_val <- validation_set(bean_split)
bean_val$splits[[1]]

library(baguette)
library(discrim)

mlp_spec <- mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |>
  set_engine('nnet') |>
  set_mode('classification')

bagging_spec <- bag_tree() |>
  set_engine('rpart') |>
  set_mode('classification')

fda_spec <- discrim_flexible(
    prod_degree = tune()
  ) |>
  set_engine('earth')

rda_spec <- discrim_regularized(frac_common_cov = tune(), frac_identity = tune()) |>
  set_engine('klaR')

bayes_spec <- naive_Bayes() |>
  set_engine('klaR')

bean_rec <- recipe(class ~ ., data = bean_train) |>
  step_zv(all_numeric_predictors()) |>
  step_orderNorm(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors())

pls_rec <- 
  bean_rec |> 
  step_pls(all_numeric_predictors(), outcome = "class", num_comp = tune())

umap_rec <-
  bean_rec |>
  step_umap(
    all_numeric_predictors(),
    outcome = "class",
    num_comp = tune(),
    neighbors = tune(),
    min_dist = tune()
  )

ctrl <- control_grid(parallel_over = "everything")

bean_res <- workflow_set(
    preproc = list(basic = class ~., pls = pls_rec, umap = umap_rec), 
    models = list(bayes = bayes_spec, fda = fda_spec,
                  rda = rda_spec, bag = bagging_spec,
                  mlp = mlp_spec)
  ) |> 
  workflow_map(
    verbose = TRUE,
    seed = 1603,
    resamples = bean_val,
    grid = 10,
    metrics = metric_set(roc_auc),
    control = ctrl
  )

rankings <- rank_results(bean_res, select_best = TRUE) |> 
  mutate(method = map_chr(wflow_id, ~ str_split(.x, "_", simplify = TRUE)[1])) 

rda_res <- bean_res |> 
  extract_workflow("pls_rda") |> 
  finalize_workflow(
    bean_res |> 
      extract_workflow_set_result("pls_rda") |> 
      select_best(metric = "roc_auc")
  ) |> 
  last_fit(split = bean_split, metrics = metric_set(roc_auc))

rda_wflow_fit <- extract_workflow(rda_res)


# Now create the model-agnostic explainer and compute the global
# model explanations with `model_parts()`:
set.seed(1807)

vip_beans <- explain_tidymodels(
  model = rda_wflow_fit,
  data = bean_train |> select(-class),
  y = bean_train$class,
  label = "RDA",
  verbose = FALSE
  ) |> 
  model_parts()


# Call `ggplot_imp(vip_beans)`:
ggplot_imp(vip_beans) +
  ggtitle(" Global explainer for the regularized discriminant analysis model on the beans data")


graphics.off()


# The figure shows that shape factors are among the most important features
# for predicting bean type, especially shape factor 4, a measure of
# solidity that takes into account the area *A*, the major axis *L*,
# and the minor axis *l*:

# SF4 = \frac{ A }{ \pi \left( L\/2 \right) \left( l\/2 \right) }

# The figure also shows that shape factor 1 (the ratio of the major axis to the area),
# the minor axis length, and roundness are the next most important characteristics
# for predicting bean variety.



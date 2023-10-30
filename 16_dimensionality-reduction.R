# Chapter 16 - Dimensionality reduction ----

## 16.1 What problems can dimensionality reduction solve? ----

# Dimensionality reduction can be used for two purposes:

# - Feature engineering

# - Exploratory analysis

# In exploratory analysis, dimensionality reduction can help
# detect unwanted trends in the data, e.g. effects not
# related to the question of interest, such as lab-to-lab differences.


# In OLS, dimensionality reduction helps reducing collinearity.


# PCA is the most straightforward method for dimensionality reduction.


# In addition to the tidymodels package, this chapter uses
# the baguette, beans, bestNormalize, corrplot, discrim, embed,
# ggforce, klaR, learntidymodels, mixOmics, and uwot packages.
{
  library(tidymodels)
  library(baguette)
  library(beans)
  library(bestNormalize)
  library(corrplot)
  library(discrim)
  library(embed)
  library(ggforce)
  library(klaR)
  library(learntidymodels) # Not available for this version of R
  library(mixOmics)
  library(uwot)
}


## 16.2 A picture is worth a thousand beans ----

# We use dimensionality reduction with "recipes" for an example data set.

# Ozkan (2020) published a data set of visual characteristics of dried beans 
# and described methods for determining their varieties.


# The process of determining which pixels correspond to a particular image is called
# *image segmentation*.

# The training data come from a set of manually labeled images that can be used to
# distinguish between seven varieties:
# Cali, Horoz, Dermason, Seker, Bombay, Barbunya, and Sira.


# There are numerous methods to quantify the shapes of objects, including the
# area, the perimeter, the major axis, the compactness, and the elongation (with
# eccentricity statistics).


# In the bean data, 16 morphology features were measured:
# area, perimeter, major axis length, minor axis length, aspect ratio, eccentricity,
# convex area, equiv diameter, extent, solidity, roundness, compactness,
# shape factor 1, shape factor 2, shape factor 3, and shape factor 4.


# We begin by loading the data:
library(tidymodels)
tidymodels_prefer()
library(beans)

# First we want to hold back a test set with `initial_split()`.
# The remaining data is split into training and validation sets:
set.seed(1601)

bean_split <- initial_validation_split(data = beans, strata = class, prop = c(0.75, 0.125))
# Warning:
# Too little data to stratify
# * Resampling will be unstratified.


print(bean_split)
# <Training/Validation/Testing/Total>
# <10206/1702/1703/13611>


# Return data frames:
bean_train <- training(bean_split)
bean_test <- testing(bean_split)
bean_validation <- validation(bean_split)


set.seed(1602)
# Return an "rset" object to use with the "tune" functions:
bean_val <- validation_set(bean_split)
bean_val$splits[[1]]
# <Training/Validation/Total>
# <10206/1702/11908>


# We want to see the correlation structure of the data:
library(corrplot)

tmwr_cols <- colorRampPalette(c("#91CBD765", "#CA225E"))

bean_train |> 
  select(-class) |> 
  cor() |> 
  corrplot(
    col = tmwr_cols(200),
    tl.col = "black",
    method = "ellipse"
  )

graphics.off()

# Many of the predictors are highly correlated, such as area and perimeter
# or shape factors 2 and 3.


## 16.3 A starter recipe ----

# We start with a basic recipe to preprocess the data prior to any dimensionality reduction.

# Several predictors are ratios and as such are likely to have skewed distributions.

# Such distributions have an increased variance, which means that PCA will
# assign thmen a higher importance.

# The "bestNormalize" package helps enforce symmetric distributions for the predictors:
library(bestNormalize)

bean_rec <- recipe(class ~ ., data = bean_train) |> 
  step_zv(all_numeric_predictors()) |> 
  step_orderNorm(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors())

# This recipe will be extended with additional steps to achieve dimensionality reduction.


## 16.4 Recipes in the wild ----

# A "workflow" object containing a "recipe" object uses `fit()` to estimate
# the "recipe" and "model", then `predict()` to make model predictions.

# Analogous functions in the "recipes" package can be used for the same purpose:

# - `prep(recipe, training)` fits the recipe to the training set.
# - `bake(recipe, new_data)` applies the recipe operations to `new_data`.

# - `recipe()` defines the preprocessing. Returns a "recipe".
# - `prep()` calculates statistics from the training set. Analogous to `fit()`. Returns a "recipe".
# - `bake()` applies the preprocessing to data sets. Analogous to `predict()`. Returns a "tibble".


### 16.4.1 Preparing a recipe ----

# We estimate `bean_rec` with the training data using `prep(bean_rec)`:
bean_rec_trained <- prep(bean_rec)
print(bean_rec_trained)


# `prep()` is for a "recipe" like `fit()` for a "model".


# When `retain = TRUE` in `prep()`, the estimated version of the training set is kept 
# within the "recipe" object.

# If the training data is large, you don't want to keep such a large amount of data in memory.
# Use `retain = FALSE` instead.


# Use `verbose = TRUE` inside of `prep()` to see where errors occur.
bean_rec_trained |> 
  step_dummy(conrbread) |> # <- not a real predictor
  prep(verbose = TRUE)
# Error in `step_dummy()`:
# Caused by error in `prep()`:
# ! Can't subset columns that don't exist.
# x Column `cornbread` doesn't exist.


# The `log_changes` option is helpful too:
show_variables <- bean_rec |> 
  prep(log_changes = TRUE)
# step_zv (zv_RLYwH): same number of columns

# step_orderNorm (orderNorm_Jx8oD): same number of columns

# step_normalize (normalize_GU75D): same number of columns


### 16.4.2 Baking the recipe ----

# Using `bake()` with a "recipe" is like using `predict()` with a "model";
# The operations estimated from the training set are applied to any data,
# like testing data or new data at prediction time.

# For example, the validation set samples can be processed:
bean_val_processed <- bake(bean_rec_trained, new_data = bean_validation)


# We plot histograms of the `area` predictor before and after the recipe was prepared:
library(patchwork)

p1 <- bean_validation |> 
  ggplot(mapping = aes(x = area)) +
  geom_histogram(bins = 30, color = "white", fill = "blue", alpha = 1/3) +
  ggtitle("Original validation set data") +
  theme_bw()

p2 <- bean_val_processed |> 
  ggplot(mapping = aes(x = area)) +
  geom_histogram(bins = 30, color = "white", fill = "red", alpha = 1/3) +
  ggtitle("Processed validation set data") +
  theme_bw()

p1 + p2

graphics.off()

# Two important aspects of `bake()` are worth noting:

# 1) Using `prep(recipe, retain = TRUE)` keeps the existing processed version of the training set in the recipe.
# This enables the user to use `bake(recipe, new_data = NULL)` to return that data for further use:
bake(bean_rec_trained, new_data = NULL) |> 
  nrow()
# 10206

bean_train |> 
  nrow()
# 10206


# If the training set is not large, using the value of `retain`
# can save computational time.


# 2) Additional selectors specify which columns to return.
# The default is `everything()`.


## 16.5 Feature extraction techniques ----

# "recipes" are the primary option in "tidymodels" for dimensionality reduction.

# We write a function to estimate the transformation and plot the data
# in a scatter plot matrix with the "ggforce" package:
library(ggforce)

plot_validation_results <- function(recipe, dat = bean_validation) {
  recipe |> 
    # Estimate any additional steps
    prep() |> 
    # Process the data (the validation set by default)
    bake(new_data = dat) |> 
    # Create the scatterplot matrix
    ggplot(mapping = aes(x = .panel_x, y = .panel_y, color = class, fill = class)) +
    geom_point(alpha = 0.4, size = 0.5) +
    geom_autodensity(alpha = 0.3) +
    facet_matrix(vars(-class), layer.diag = 2) +
    scale_color_brewer(palette = "Dark2") +
    scale_fill_brewer(palette = "Dark2")
}

# This function will be reused several times in this chapter.


### 16.5.1 Principal components analysis ----

# Principal components analysis (PCA) is an unsupervised method that uses
# linear combinations of the predictors to define new features.

# These features attempt to account for as much variation as possible 
# in the original data.

# We add `step_pca()` to the original "recipe" and use our custom
# function `plot_validation_results()` to visualize the results 
# on the validation set:
bean_rec_trained |> 
  step_pca(all_numeric_predictors(), num_comp = 4) |> 
  plot_validation_results() +
  ggtitle("Principal Component Analysis") +
  theme_bw()

graphics.off()

# We see that the first two components, `PC1` and `PC2`,
# especially when used together, can effectively distinguish between
# or separate the classes.


# Recall that PCA is unsupervised. For these data, PCA components that
# explain the most variation in the predictors also happen to be
# predictive of the classes.

# What features are driving the performance?

# The "learntidymodels" package has functions to visualize
# the top features for each component.

# We need the prepared "recipe"; The PCA step is added along with a call to `prep()`:
library(learntidymodels)

bean_rec_trained |> 
  step_pca(all_numeric_predictors(), num_comp = 4) |> 
  prep() |> 
  plot_top_loadings(component_number <= 4, n = 5) +
  scale_fill_brewer(palette = "Paired") +
  ggtitle("Principal Component Analysis") +
  theme_bw()

graphics.off()


# The top loadings are related to the cluster of correlated predictors
# shown in the to-left portion of the previous correlation plot:
# perimeter, area, major axis length, and convex area.
# These are all related to bean size.

# Shape factor 2, from Symons and Fulcher (1988), is the area over the
# cube of the major axis length and is also related to bean size.

# Measures of elongation appear to dominate the second PCA component.


### 16.5.2 Partial least squares ----

# Partial Least Squares (PLS), is a supervised version of PCA.

# It tries to find the components that simultaneously maximize the
# variation in the predictors while also maximizing the relationship 
# between those components and the outcome:
bean_rec_trained |> 
  step_pls(all_numeric_predictors(), outcome = "class", num_comp = 4) |> 
  plot_validation_results() +
  ggtitle("Partial Least Squares") +
  theme_bw()

graphics.off()

# The first two PLS components plotted are nearly identical to the
# first two PCA components.

# This is beccause tw first two PCA components are effective at
# separating the varieties of beans.

# The remaining components are different.

# We can visualize the loadings, the top features for each component:
bean_rec_trained |> 
  step_pls(all_numeric_predictors(), outcome = "class", num_comp = 4) |> 
  prep() |> 
  plot_top_loadings(component_number <= 4, n = 5, type = "pls") +
  scale_fill_brewer(palette = "Paired") +
  ggtitle("Partial Least Squares") +
  theme_bw()

graphics.off()

# Solidity, i.e. the density of the bean, drives the third PLS component,
# along with roundness.

# Solidity may be capturing bean features related to "bumpiness" of the
# bean surface since it can measure irregularity of the bean boundaries.


### 16.5.3 Independent component analysis ----

# Independent Component Analysis (ICA) aims to find components that
# are statistically as independent from one another as possible
# (as opposed to uncorrelated).

# It maximizes the "non-Gaussianity" of the ICA components, 
# i.e. it separates information instead of compressing information 
# like PCA does.

# We use `step_ica()`: 
bean_rec_trained |> 
  step_ica(all_numeric_predictors(), num_comp = 4) |> 
  plot_validation_results() +
  ggtitle("Independent Component Analysis") +
  theme_bw()

graphics.off()

# Inspecting this plot, there is not much separation between the
# classes in the first few components when using ICA.

# These independent components do not separate bean types.


### 16.5.4 Uniform manifold approximation and projection ----

# Uniform Manifold Approximation and Projection (UMAP) is similar to the
# popular t-SNF method for nonlinear dimension reduction.

# In the original high-dimensional space, UMAP uses a distance-based
# nearest neighbor method to find local areas of the data where the
# data points are more likely to be related.

# The relationship between data points is saved as a directed graph
# model where most points are not connected.


# From there, UMAP translates points in the graph to the reduced dimensional space.
# To do this, the algorithm has an optimization process that uses cross-entropy
# to map data points to the smaller set of features so that the graph is well approximated.


# To create the mapping, the "embed" pacakge contains a step function for this method:
library(embed)

bean_rec_trained |> 
  step_umap(all_numeric_predictors(), num_comp = 4) |> 
  plot_validation_results() +
  ggtitle("UMAP") +
  theme_bw()

graphics.off()

# While the between-cluster space is pronounced, the cluster can contain
# a heterogeneous mixture of classes.


# There is also a supervised version of UMAP:
bean_rec_trained |> 
  step_umap(all_numeric_predictors(), outcome = "class", num_comp = 4) |> 
  plot_validation_results() +
  ggtitle("UMPAP (supervised)") +
  theme_bw()

graphics.off()


# UMPAP is a powerful method to reduce the feature space.
# However, it is sensitive to the tuning parameters
# (e.g., the number of neighbors and so on).


## 16.6 Modeling ----

# We want to explore a variety of different models with the
# dimensionality reduction techniques PLS and UMAP,
# along with no transformation at all:

# - a single layer neural network
# - bagged trees
# - flexible discriminant analysis (FDA)
# - naive Bayes
# - regularized discriminant analysis (RDA)


# We create a series of model specifications and then use a workflow set
# to tune the models.

# Note that the model parameters are tuned in conjunction with the
# recipe parameters, e.g. the size of the reduced dimension or the UMAP parameters.
library(baguette)
library(discrim)

mlp_spec <- mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |> 
  set_engine("nnet") |> 
  set_mode("classification")

bagging_spec <- bag_tree() |> 
  set_engine("rpart") |> 
  set_mode("classification")

fda_spec <- discrim_flexible(prod_degree = tune()) |> 
  set_engine("earth")

rda_spec <- discrim_regularized(frac_common_cov = tune(), frac_identity = tune()) |> 
  set_engine("klaR")

bayes_spec <- naive_Bayes() |> 
  set_engine("klaR")


# We also need recipes for the dimensionality reduction methods that we want to try.
# We start with a base recipe `bean_rec` and extend it with different dimensionality
# reduction steps:
bean_rec <- recipe(class ~ ., data = bean_train) |> 
  step_zv(all_numeric_predictors()) |> 
  step_orderNorm(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors())

pls_rec <- bean_rec |> 
  step_pls(all_numeric_predictors(), outcome = "class", num_comp = tune())


umap_rec <- bean_rec |> 
  step_umap(
    all_numeric_predictors(), 
    outcome = "class",
    num_comp = tune(),
    neighbors = tune(),
    min_dist = tune()
  )


# The "workflwosets" package takes the preprocessors and models and crosses them.

# The `control` option `parallel_over` is set so that the parallel processing
# works simultaneously across tuning parameters.

# `worklfow_map()` applies grid search to optimize the model / preprocessing
# parameters (if any) across 10 parameter combinations.

# The multiclass area under the ROC curve is estimated on the validation set.
ctrl <- control_grid(parallel_over = "everything")


# Caution: This will take over an hour to run.
bean_res <- workflow_set(
  
  preproc = list(
    basic = class ~ .,
    pls = pls_rec,
    umap = umap_rec
    ),
  
  models = list(
    bayes = bayes_spec,
    fda = fda_spec,
    rda = rda_spec,
    bag = bagging_spec,
    mlp = mlp_spec
    )
  
  ) |> 
  workflow_map(
    verbose = TRUE,
    seed = 1603,
    resamples = bean_val,
    grid = 10, 
    metrics = metric_set(roc_auc),
    control = ctrl
  )


# We can rank the models by their validation set estimates of the
# area under the ROC curve:
rankings <- rank_results(bean_res, select_best = TRUE) |> 
  mutate(
    method = map_chr(.x = wflow_id, .f = ~ str_split(.x, "_", simplify = TRUE)[1])
  )

tidymodels_prefer()
filter(rankings, rank <= 5) |> 
  dplyr::select(rank, mean, model, method)


# Most models give a very good performance;
# there are few bad choices here.


# For demonstration, we will use the RDA model with PLS features as the final model.
# We will finalize the workflow with the numerically best parameters,
# fit it to the training set, then evaluate with the test set:
rda_res <- bean_res |> 
  extract_workflow("pls_rda") |> 
  finalize_workflow(
    bean_res |> 
      extract_workflow_result("pls_rda") |> 
      select_best(metric = "roc_auc")
  ) |> 
  last_fit(split = bean_split, metrics = metric_set(roc_auc))


rda_wflow_fit <- extract_workflow(rda_res)


# What are the results for our metric (multiclass ROC AUC) on the testing set?
collect_metrics(rda_res)


# This is a good result and we can use this model in the next chapter to
# demonstrate variable importance methods.

# END
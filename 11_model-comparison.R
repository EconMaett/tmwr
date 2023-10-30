# Chapter 11 Comparing models with resampling ----

## Previous analysis ----
library(tidymodels)
tidymodels_prefer()

# Access the data
data(ames)
ames <- ames |> 
  mutate(
    Sale_Price = log10(Sale_Price)
  )

set.seed(502)
# Initial split, 80% training set, 20% test set, stratified sampling
ames_split <- initial_split(data = ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

# recipe
ames_rec <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train
  ) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) |> 
  step_ns(Latitude, Longitude, deg_free = 20)

# model
lm_model <- linear_reg() |> 
  set_engine("lm")

# workflow
lm_wflow <- workflow() |> 
  add_model(lm_model) |> 
  add_recipe(ames_rec)

# fit
lm_fit <- fit(lm_wflow, ames_train)


# random forest model
rf_model <- rand_forest(trees = 1000) |> 
  set_engine("ranger") |> 
  set_mode("regression")

rf_wflow <- workflow() |> 
  add_formula(
    Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude
  ) |> 
  add_model(rf_model)


# 10-fold cross-validation
set.seed(1001)
ames_folds <- vfold_cv(ames_train, v = 10)

keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)


set.seed(1003)
rf_res <- rf_wflow |> 
  fit_resamples(resamples = ames_folds, control = keep_pred)


### 11.1 Creating multiple models with workflow sets ----

# We create three different linear models that incrementally add preprocessing steps.
basic_rec <- recipe(
  Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
  data = ames_train
  ) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors())


interaction_rec <- basic_rec |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") )


spline_rec <- interaction_rec |> 
  step_ns(Latitude, Longitude, deg_free = 50)


preproc <- list(
  basic    = basic_rec,
  interact = interaction_rec,
  splines  = spline_rec
)


lm_models <- workflow_set(
  preproc = preproc,
  models  = list(lm = linear_reg()),
  cross   = FALSE
)

print(lm_models)


# We would like to resample each of these models in turn.
# To do so, we use a "purrr"-like function called `workflow_map()`

# This function takes an initial argument of the function to apply to the workflows,
# followed by options to that function.

# We also set a `verbose` argument that will print the progress as well as 
# a `seed` argument that ensures that each model uses the same random number stream as the others.
lm_models <- lm_models |> 
  workflow_map(
    "fit_resamples",
    
    # Options to `workflow_map()`
    seed = 1101,
    verbose = TRUE,
    
    # Options to `fit_resamples()`:
    resamples = ames_folds,
    control   = keep_pred
  )

print(lm_models)

# The `option` and `result` columns are now populated.

# The `option` column includes the options to `fit_resamples()`
# that were given (for reproducibility)


# The `result` column contains the results produced by `fit_resamples()`.


# Use `collect_metrics()` to collate the performance statistics and
# `filter()` for the metric you are interested in:
collect_metrics(lm_models) |> 
  filter(.metric == "rmse")
# basic_lm:    0.0803
# interact_lm: 0.0799
# splines_lm:  0.0785

# We can add the random forest model from last chapter to the set by
# first converting it to its own workflow and then binding the rows.

# This requires that, ad the time the model was resampled,
# the option `save_workflow = TRUE` was set in the control function:
four_models <- as_workflow_set(random_forest = rf_res) |> 
  bind_rows(lm_models)

print(four_models)


# The `autoplot()` method shows confidence intervals for each model in order
# of best to worst.

# We focus on the coefficient of determination, the R-squared,
# and use `metric = "rsq"`:
library(ggrepel)

autoplot(four_models, metric = "rsq") +
  geom_text_repel(mapping = aes(label = wflow_id), nudge_x = 1/8, nudge_y = 1/100) +
  theme(legend.position = "none") +
  theme_bw()

graphics.off()

# From this pot of R-squared intervals, we can see that the random forest method
# is doing the best job and there are minor improvements in the linear models
# as we add more recipe steps.


## 11.2 Comparing resampled performance statistics ----

# The additional terms do not profoundly improve the mean RMSE or R-squared
# statistics for the linear models.

# While the difference is small, it might still be larger than the experimental
# noise in the system, i.e., considered statistically significant.

# We can formally test the hypothesis that the additional terms increase the R-squared.


# Note that before making between-model comparisons, it is important to discuss
# the within-resample correlation for resampling statistics.

# Each model was measured with the same cross-validation folds, and results
# for the resample tend to be similar.


# There are some resamples where performance across models tends to be low and others
# where it tends to be high. This is called a
# *resample-to-resample* component of variation.


# We gather the individual resampling statistics for the linear models and the random forest.
# We focus on the R-squared statistic which measures the correlation between the observed
# and predicted sale price for each house.

# We `filter()` to only keep the R-squared metrics, reshape the results,
# and compute how the metrics are correlated with each other:
rsq_indv_estimates <- collect_metrics(four_models, summarize = FALSE) |> 
  filter(.metric == "rsq")

head(rsq_indv_estimates)


rsq_wider <- rsq_indv_estimates |> 
  select(wflow_id, .estimate, id) |> 
  pivot_wider(id_cols = "id", names_from = "wflow_id", values_from = ".estimate")

head(rsq_wider)


corrr::correlate(rsq_wider |> select(-id), quiet = TRUE)

# These correlations are high, and indicate that, across models, there are
# large within-resample correlations.

# We can visualize the R-squared statistics for each model with
# lines connecting the resamples:
rsq_indv_estimates |> 
  mutate(
    wflow_id = reorder(wflow_id, .estimate)
  ) |> 
  ggplot(mapping = aes(x = wflow_id, y = .estimate, group = id, color = id)) +
  geom_line(alpha = 0.5, linewidth = 1.25) +
  theme(legend.position = "none") +
  ylab(expression(R^{2})) +
  theme_bw()

graphics.off()


# If the resample-to-resample effect was not real, there would not be any parallel lines.

# A statistical test for the correlations evaluates whether the magnitudes of these
# correlations are not simply noise.

# For the linear models:
rsq_wider |> 
  with(cor.test(basic_lm, splines_lm)) |> 
  broom::tidy() |> 
  select(estimate, starts_with("conf"))
# estimate: 0.997
# conf.low: 0.987
# conf.high: 0.999

# The results of the correlation test show that the within-sample correlation
# appears to be real.


# What effect does the extra correlation have on the analysis?

# Consider the variance of two random variables X and Y:
# Var[X - Y] = Var[X] - 2*Cov[X,Y] + Var[Y]

# If the covariance between X and Y is positive, the overall variance decreases.

# This would mean that any statistical test of this difference is critically
# under-powered when comparing the difference in the two models.

# Ignoring the resample-to-resample effect would bias the model comparison
# towards finding no difference between models.


# It can be helpful to define a relevant **practical effect size**.

# In the example here, the relevant effect size would be the change in 
# the R-squared statistic that we would consider to be a realistic difference 
# that actually matters.

# Practical significance is subjective.
# We might think that two models are not practically different if their
# R-squared values are within +/- 2%.


## 11.3 Simple hypothesis testing methods ----

# We consider a linear model
# y = b_0 + b_1*x_1 + ... + b_p*x_p + e

# This model is used as a regression model and forms the basis of the
# analysis of variance (ANOVA) technique for comparing groups.

# With the ANOVA model, the predictors x_i are binary dummy variables for
# different groups.

# The b_i parameters then estimate whether two or more groups are different 
# from one another.


# For our example, we can define
# y = b_0 + b_1*x_1 + b_2*x_2 + b_3*x_3 + e

# Where y is the R-squared statistic, and x_1, x_2, x_3 are the model components.

# b_0 will be the estimate of the mean R-squared for the basic linear model
# (i.e. without splines or interaction terms, x_1 = x_2 = x_3 = 0).

# b_1 is the change in mean R-squared when the interaction terms are included,
# i.e. x_1 = 1, x_2 = x_3 = 0

# b_2 is the change in mean R-squared between the basic linear model and the random forest model,
# aka x_1 = 0, x_2 = 1, x_3 = 0

# b_3 is the change in mean R-squared between the basic linear model and the
# one with interaction terms and splines, 
# aka x_1 = x_2 = 0, x_3 = 1

compare_lm <- rsq_wider |> 
  mutate(
    difference = splines_lm - basic_lm
  )


lm(formula = difference ~ 1, data = compare_lm) |> 
  broom::tidy(conf.int = TRUE) |> 
  select(estimate, p.value, starts_with("conf"))
# estimate = 0.00913
# p.value = 0.0000256


# Alternatively, a paired t-test could be used:
rsq_wider |> 
  with(t.test(splines_lm, basic_lm, paired = TRUE)) |> 
  broom::tidy() |> 
  select(estimate, p.value, starts_with("conf"))
# estimate = 0.00913
# p.value = 0.000256


# What is a p-value?
# Informally, a p-value is the probability under the null hypothesis
# (a specified statistical model)
# that the difference between two compared groups would be
# equal to or more extreme than its observed value.


## 11.4 Bayesian methods ----

# We can take a more general approach to making these formal comparisons
# using random effects and Bayesian statistics, following Richard McElreath (2020).

#  McElreath, R. 2020. Statistical Rethinking: A Bayesian Course with Examples in R and Stan. CRC press. 

# The previous ANOVA model had the form
# y = b_0 + b_1*x_1 + b_2*x_2 + b_3*x_3 + e

# where the residuals e are assumed to be independent and follow a 
# Gaussian distribution with zero mean and constant standard deviation sigma.
# e ~ iid N(0, sigma)

# From this assumption follows that the estimated regression parameters
# b_0, b_1, b_2, b_3,
# follow a multivariate Gaussian distribution.

# A Bayesian linear model makes additional assumptions.

# In addition to specifying a distribution for the residuals, we require *prior
# distribution* specifications for the model parameters b_i and sigma.

# These are distributions for the parameters that the model assumes before being
# observed to the observed data.

# A simple set of prior distributions for our model might be:
# e ~ N(0, sigma)
# b ~ N(0, 10)
# sigma ~ exp(1)


# These prior distributions set the possible/probable ranges of the model parameters
# and have no unknown parameters.

# For example, the prior on sigma indicates that values must be larger than
# 0, are very right-skewed, and have values that are usually less than 3 or 4.


# Note that the regression parameters have a wide prior distribution,
# with a standard deviation of 10.

# In many cases, we may not have a strong opinion about the prior distribution
# beyond it being symmetric and bell shaped.

# The large standard deviation implies a fairly uninformative prior.

# It is not overly restrictive in terms of the possible values that the
# parameters might take on.

# It allows the data to have more of an influence during parameter estimation.


# Given the observed data and the prior distribution, the model parameters
# can be estimated.

# The final distributions of the model parameters are combinations of the priors
# and the likelihood estimates.

# These **posterior distributions** of the parameters are the key distribution
# of interest.

# They are a full probabilistic description of the model's estimated parameters.


## A random intercept model ----

# We consider a *random intercept* model, where we assume that the resamples
# impact the model only by changing the intercept.

# Note that this constrains the resamples from having a differential impact
# on the regression parameters b_j;

# these are assumed to have the same relationship across resamples.

# y = (b_0 + b_i) + b_1*x_1 + b_2*x_2 + b_3*x_3 + e

# This is a reasonable model for resampled statistics that when plotted
# across models have fairly parallel effects across models,
# i.e., little cross-over of lines.


# For this model, an additional assumption is made for the prior distribution
# of random effects.

# A reasonable assumption is another symmetric distribution, such as a
# bell-shaped curve.

# Given the effective sample size of 10 in our summary statistic data,
# we use a prior distribution that is wider than a standard normal distribution.

# We use a t-distribution with a single degree of freedom, i.e.
# b ~ t(1)
# which has heavier tails than the analogous Gaussian distribution.


# The "tidyposterior" package has functions to fit such Bayesian models for
# the purpose of comparing resampling methods.

# The main function is `perf_mod()`:

# - For workflow sets, it creates an ANOVA model where the groups correspond
#   to the workflows. 
#   If one of the workflows in the set had data on tuning parameters, the
#   best tuning parameters set for each workflow is used in the Bayesian analysis.
#   Thereby `perf_mod()` focuses on *between-workflow comparisons*.

# - For objects that contain a single model that has been tuned by resampling,
#   `perf_mod()` makes *within-model comparisons*.
#   The grouping variables tested in the Bayesian ANOVA are the submodels
#   defined by the tuning parameters.

# - The `perf_model()` function can take a data frame returned by "rsample"
#   with columns of performance metrics associated with two or more
#   model/workflow results.


# From any of these objects, the `tidyposterior::perf_mod()` function
# determines an appropriate Bayesian model and fits it with the resampling statistics.

# For our example, it will model the four sets of R-squared statistics
# associated with the four workflows.


# The "tidyposterior" package uses the **Stan** software and programming language
# to specify and fit the models, with the "rstanarm" package being the interface.
library(tidyposterior)
library(rstanarm)


# The functions within the "rstanarm" package have default priors
?rstanarm::priors


# The "rstanarm" package creates copious amounts of output;
# those results are not shown here but are worth inspecting for potential issues.
# The option `refresh = 0` can be used to eliminate the logging.
rsq_anova <- perf_mod(
  four_models,
  metric = "rsq",
  prior_intercept = rstanarm::student_t(df = 1),
  chains = 4,
  iter = 5000,
  seed = 1102
)


# the resulting object, `rsq_anova`, has information on the resampling process
# as well as the Stan object embedded within, an element called `stan`.
class(rsq_anova)
# perf_mod_workflow_set, perf_mod

# We are most interested in the posterior distributions of the regression parameters.

# The "tidyposterior" package has a `tidy()` method that extracts these
# posterior distributions into a tibble:
model_post <- rsq_anova |> 
  # Take a random sample form the posterior distribution so set the seed again for reproducibility
  tidy(seed = 1103)

glimpse(model_post)


# The four posterior distributions can be visualized:
model_post |> 
  mutate(
    model = forcats::fct_inorder(model)
  ) |> 
  ggplot(mapping = aes(x = posterior)) +
  geom_histogram(bins = 50, color = "white", fill = "blue", alpha = 0.4) +
  facet_wrap(~ model, ncol = 1) +
  labs(
    title = "",
    subtitle = "",
    y = "count",
    x = expression("Posterior for mean" ~ R^{2})
  ) +
  theme_bw()

graphics.off()


# These histograms describe the estimated probability distributions of the mean R-squared value
# for each of the four models.

# There is some overlap, especially for the three linear models.


# There is also a basic `autoplot()` method for the model results,
# as well as the tidied object that shows overlaid density plots:
autoplot(rsq_anova) +
  ggrepel::geom_text_repel(mapping = aes(label = workflow), nudge_x = 1/8, nudge_y = 1/100) +
  theme(lengend.position = "none") +
  theme_bw()

graphics.off()
# These are credible intervals derived from the model posterior distributions.


# The wonderful aspect of using resampling with Bayesian methods is that, once
# we have the posteriors for the parameters, it is trivial to get the posterior distributions
# for combinations of the parameters.

# For example, to compare the two linear regression models, we are interested in the
# difference in means.

# The posterior of this difference is computed by sampling from the individual posteriors
# and taking the differences.


# The `contrast_models()` function can do this.
# To specify the comparisons to make, the `list_1` and `list_2` parameters take character
# vectors and compute the differences between the models in those lists,
# parameterized as `list_1 - list_2`.


# Compare two of the linear models and visualize the results:
rsq_diff <- contrast_models(rsq_anova, list_1 = "splines_lm", list_2 = "basic_lm", seed = 1104)

rsq_diff |> 
  as_tibble() |> 
  ggplot(mapping = aes(x = difference)) +
  geom_vline(xintercept = 0, lty = 2) +
  geom_histogram(bins = 50, color = "white", fill = "red", alpha = 0.4) +
  xlab(expression("Posterior for mean difference in" ~ R^{2} ~ "(splines + interaction - basic)")) +
  theme_bw()

graphics.off()
# The graphic shows the posterior distribution for the difference in the coefficient 
# of determination (R-squared).


# The posterior shows that the center of the distribution is greater than zero,
# indicating that the model with splines typically had larger values,
# but there is no overlap with zero to a degree.

# The `summary()` method for this object computes the mean of the distribution
# as well as credible intervals, the Bayesian analog to confidence intervals:
summary(rsq_diff) |> 
  select(-starts_with("pract"))
# mean = 0.00910,
# probability = 1.00

# The `probability` column reflects the proportion of the posterior greater than zero.

# This is the probability that the positive difference is real.

# The value is not close to zero, providing a strong case for statistical significance,
# i.e., the idea that statistically the actual differenc eis not zero.


# However, the estimate of the mean difference is close to zero.
# Recall that we defined the practical effect size we as 2%.

# With a posterior distribution, we can also compute the probability of being practically significant.

# In Bayesian analysis, this is a *ROPE estimate*, where ROPE stands for
# Region Of Practical Equivalence.

# To estimate this, the `size` option in the `summary()` method is sued:
summary(rsq_diff, size = 0.02) |> 
  select(contrast, starts_with("pract"))
# pract_neg = 0
# pract_equiv = 1.00
# pract_pos = 0.0001

# The `pract_equiv` column is the proportion of the posterior that is within the
# range `[-size, size]`.

# The columns `pract_neg` and `pract_pos` are the proportions below and above this interval.

# This large value indicates that, for our effect size, there is an overwhelming probability
# that the two models are practically the same.

# Even though the previous plot showed that our difference is likely nonzero,
# the equivalence test suggests that it is small enough not to be of no practical use.


# The same process can be used to compare the random forest model to one of the linear regressions.

# `perf_mod()` can be applied to a workflow set and the `autoplot()` method will
# show the `pract_equiv` results that compare each workflow to the current best model:
autoplot(rsq_anova, type = "ROPE", size = 0.02) +
  ggrepel::geom_text_repel(mapping = aes(label = workflow)) +
  theme(legend.position = "none") +
  theme_bw()

graphics.off()
# The graphic shows the probability of practical equivalence
# for an effect size of 2%.

# The figure indicates that none of the linear models come close to the random
# forest model when a 2% practical effect size is used.


### The effect of the amount of resampling ----

# How does the number of resamples affect these types of formal Bayesian comparisons?

# More resamples increases the precision of the overall resampling estimate;
# that precision propagates to this type of analysis.


# For illustration, additional resamples were added using repeated cross-validation.

# How did the posterior distribution change?

# We plot the 90% credible intervals with up to 100 resamples, generated from
# 10 repeats of 10-fold cross-validation:

# The code to generate the `intervals` object is available here:
# https://github.com/tidymodels/TMwR/blob/main/extras/ames_posterior_intervals.R

library(tidymodels)
library(doMC) # not available for this version of R
library(tidyposterior)
library(workflowsets)
library(rstanarm)

theme_set(theme_bw())

data(ames, package = "modeldata")

ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(123)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

crs <- parallel::detectCores()

registerDoMC(cores = crs)

## -----------------------------------------------------------------------------

set.seed(55)
ames_folds <- vfold_cv(ames_train, v = 10, repeats = 10)

lm_model <- linear_reg() |> set_engine("lm")

rf_model <-
  rand_forest(trees = 1000) |>
  set_engine("ranger") |>
  set_mode("regression")

# ------------------------------------------------------------------------------

basic_rec <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) |>
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors())

interaction_rec <- 
  basic_rec |> 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) 

spline_rec <- 
  interaction_rec |> 
  step_ns(Latitude, Longitude, deg_free = 50)

preproc <- 
  list(basic = basic_rec, 
       interact = interaction_rec, 
       splines = spline_rec,
       formula = Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + 
         Bldg_Type + Latitude + Longitude
  )

models <- list(lm = lm_model, lm = lm_model, lm = lm_model, rf = rf_model)

four_models <- 
  workflow_set(preproc, models, cross = FALSE)
four_models

posteriors <- NULL

for(i in 11:100) {
  if (i %% 10 == 0) cat(i, "... ")
  
  tmp_rset <- rsample:::df_reconstruct(ames_folds |> slice(1:i), ames_folds)
  
  four_resamples <- 
    four_models |> 
    workflow_map("fit_resamples", seed = 1, resamples = tmp_rset)
  
  ## -----------------------------------------------------------------------------
  
  rsq_anova <-
    perf_mod(
      four_resamples,
      prior_intercept = student_t(df = 1),
      chains = crs - 2,
      iter = 5000,
      seed = 2,
      cores = crs - 2,
      refresh = 0
    )
  
  rqs_diff <-
    contrast_models(rsq_anova,
                    list_1 = "splines_lm",
                    list_2 = "basic_lm",
                    seed = 3) |>
    as_tibble() |>
    mutate(label = paste(format(1:100)[i], "resamples"), resamples = i)
  
  posteriors <- bind_rows(posteriors, rqs_diff)
  
  rm(rqs_diff)
  
}

## -----------------------------------------------------------------------------

ggplot(posteriors, aes(x = difference)) +
   geom_histogram(bins = 30) +
   facet_wrap(~label)

ggplot(posteriors, aes(x = difference)) +
   geom_line(stat = "density", trim = FALSE) +
  facet_wrap(~label)

intervals <-posteriors |>
  group_by(resamples) |>
  summarize(
    mean = mean(difference),
    lower = quantile(difference, prob = 0.05),
    upper = quantile(difference, prob = 0.95),
    .groups = "drop"
  ) |>
  ungroup() |>
  mutate(
    mean = predict(loess(mean ~ resamples, span = 0.15)),
    lower = predict(loess(lower ~ resamples, span = 0.15)),
    upper = predict(loess(upper ~ resamples, span = 0.15))
  )

save(intervals, file = "RData/post_intervals.RData")

ggplot(intervals,
       aes(x = resamples, y = mean)) +
  geom_path() +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "red", alpha = 0.1) +
  labs(
    title = "Probability of practical equivalence to random forest model",
    y = expression(paste("Mean difference in ", R^2)),
    x = "Number of Resamples (repeated 10-fold cross-validation)"
    )

# The width of the intervals decreases as more resamples are added.

# Clearly, going from ten resamples to thirty has a larger impact than going from
# eighty to 100.

# These are diminishing returns for using a "large" number of resamples.

# END
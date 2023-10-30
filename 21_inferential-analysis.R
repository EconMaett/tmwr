# Chapter 21 - Inferential analysis ----

# The previous chapters have focused on predictive modeling.


# Inferential models are used to make inferences or judgement about
# some component of the model, such as a coefficient value or another parameter.

# The results should answer e pre-defined question or hypothesis.


# Predictive models use predictions on hold-out data to validate or characterize
# the quality of the model.


# Inferential methods focus on validating the probabilistic or structural assumptions
# that are made prior to fitting the model.



# Ordinary linear regression can be used for both predictive and inferential purposes.

# It rests on the assumption that the residual values are independent and follow
# a Gaussian distribution with a constant variance.


# The residuals from the fitted model are usually examined to determine if
# this assumption holds.


# In this chapter we will use p-values.

# However, the "tidymodels" framework promotes confidence intervals over p-values
# as a method for quantifying the evidence for an alternative hypothesis.

# Bayesian methods are often superior to both p-values and confidence intervals
# in terms of ease of interpretation, although they are computationally more expensive.


## 21.1 Inference for count data ----

# We use biochemistry publication data from the "pscl" package.
# It consists of information on 915 Ph.D biochemistry graduates and tries to
# explain factors that impact their academic productivity as measured by the number
# or count of articles published within three years.

# The predictors include gender, marital status, number of children older than five years,
# the prestige of the department, the number of articles produced by their mentor in the same period.

# The data reflects biochemistry doctorates who finished their education between 1956 and 1963.

# The data is a biased sample of all biochemistry doctorates during this period.


# In Chapter 19 we asked the question
# "Is the model applicable for predicting a specific data point?"

# We need to define what populations an inferential analysis applies to.


# We crate a plot of the data indicating that many graduates did not publish
# any articles in the time examined and that the outcome follows a right-skewed distribution:
library(tidymodels)
tidymodels_prefer()


data("bioChemists", package = "pscl")


theme_set(theme_bw())

ggplot(data = bioChemists, mapping = aes(x = art)) +
  geom_histogram(binwidth = 1, color = "white") +
  labs(
    title = "Distribution of the number of articles written within 3 years of graduation",
    x = "Number of articles within 3y of graduation"
  )

graphics.off()

# Since the outcome data are counts, the most common distribution assumption
# to make is that the outcome has a Poisson distribution.


## 21.2 Comparisons with two-sample tests ----

# The original author's goal with this data set on biochemistry publication data
# was to determine if there is a difference in publications between men and women.
bioChemists |> 
  group_by(fem) |> 
  summarise(
    count = sum(art),
    n = length(art)
  )
# A tibble: 2 * 3
# fem     count       n
# <fct>   <int>   <int>
# Men     930       494
# Women   619       421


# There were many more publications by men, although there were also more men 
# in the data set.

# We perform a two-sample comparison using the `poisson.test()` function
# from the "stats" package.


# This test requires the counts for one or two groups and the hypotheses:
# H0: There is no difference in publications between the sexes
# HA: There is a difference.

stats::poisson.test(c(930, 619), T = 3)
# The function returns a p-value and a confidence interval for the ratio of
# the publication rates.

# The result indicates that the observed difference is greater than the
# experiential noise and favors the alternative hypothesis.


# The results were returned as an object of type `htest`
res <- stats::poisson.test(c(930, 619), T = 3)

class(res)
# "htest"


# The special structure of the class "htest" object makes it difficult
# to use the results in reporting or visualizations.

# We use `broom::tidy()` to return a "tibble":
stats::poisson.test(c(930, 619), T = 3) |> 
  broom::tidy()

# Between the "broom" and "broom.mixed" packages, the `tidy()` function
# possesses methods for more than 150 models!


# The `poisson.test()` function relies on the assumption that the outcome
# data follow a Poisson distribution.

# If we would like to test our hypothesis with a test that makes
# fewer distributional assumptions, we may use
# the bootstrap test or the permutation test.


# The "infer" package from the "tidymodels" framework
# includes such tests.


# First, we `specify()` that we use the difference in the mean number of
# articles between the sexes and then we `calculate()` the test statistic.


# The maximum likelihood estimator for the Poisson mean is the sample mean.


# With the "infer" package, we first specify the outcome and covariate,
# then the statistic of interest:
library(infer)


observed <- bioChemists |> 
  specify(formula = art ~ fem) |> 
  calculate(stat = "diff in means", order = c("Men", "Women"))

print(observed)
# Response: art (numeric)
# Explanatory: fem (factor)
# A tibble: 1 * 1
# stat <dbl>: 0.412


# We compute a confidence interval for this mean by creating the bootstrap
# distribution with `generate()`;
# The same test statistic is computed for each resampled version of the data:
set.seed(2101)

bootstrapped <- bioChemists |> 
  specify(formula = art ~ fem) |> 
  generate(reps = 2000, type = "bootstrap") |> 
  calculate(stat = "diff in means", order = c("Men", "Women"))

print(bootstrapped)


# Calculate a percentile interval:
percentile_ci <- get_ci(bootstrapped)

print(percentile_ci)
# lower_ci: 0.158, upper_ci: 0.653

# The "infer" package has a high-level APO to show the analysis results:
visualise(bootstrapped) +
  shade_confidence_interval(endpoints = percentile_ci)

graphics.off()


# If we require a p-value, the "infer" package can compute the value with
# a permutation test.

# We add a `hypothesize()` verb to state the assumption to test
# and the `generate()` call contains an option to shuffle the data:
set.seed(2102)

permuted <- bioChemists |> 
  specify(formula = art ~ fem) |> 
  hypothesise(null = "independence") |> 
  generate(reps = 2000, type = "permute") |> 
  calculate(stat = "diff in means", order = c("Men", "Women"))

print(permuted)


# We visualize the results and add a vertical line that signifies
# the observed value:
visualise(permuted) +
  shade_p_value(obs_stat = observed, direction = "two-sided")

graphics.off()


# The actual p-value is:
permuted |> 
  get_p_value(obs_stat = observed, direction = "two-sided")
# p_value <dbl>: 0.002

# The vertical line representing the null hypothesis is far away
# from the permutation distribution.

# This means that if the null hypothesis were true,
# the likelihood of observing data at least as extreme as what is at hand
# is exceedingly small.


## 21.3 Log-linear models ----

# We focus on a generalized linear model (GLM) where we assume that the
# counts follow a Poisson distribution.

# In this model, the covariates/predictors enter the model in a log-linear 
# fashion:

# log(lambda) = b0 + b1*x1 + ... + bp*xp
# "here lambda is the expected value of the counts.


# We fit a simple model that contains all of the predictor columns.

# The "poissonreg" package is a "parsnip" extension package within the
# "tidymodels" framework:
library(poissonreg)

# default engine is "glm"
log_lin_spec <- poisson_reg()


log_lin_fit <- log_lin_spec |> 
  fit(formula = art ~ ., data = bioChemists)

print(log_lin_fit)


# Use `broom::tidy()` to summarize the coefficients and the 90% confidence intervals:
broom::tidy(log_lin_fit, conf.int = TRUE, conf.level = 0.90)

# The lowest p-value is associated with the "phd" variable, the prestige of the departement.
# This may indicate that it has little association with the outcome.

# The "rsample" package has a convenience function to compute bootstrap confidence
# intervals for `lm()` and `glm()` models.

# This allows us to test our hypothesis without relying on the Poisson likelihood
# for the confidence intervals.


# We use this function while declaring `family = poisson` to compute the model fits.

# The 90% confidence bootstrap-t interval is the default.

# percentile intervals are also available.
set.seed(2103)

glm_boot <- reg_intervals(
  formula = art ~ ., 
  data = bioChemists,
  model_fn = "glm", 
  family = poisson
  )

print(glm_boot)

# When we compare these results to the purely parametric results from `glm()`,
# we see that the bootstrap intervals are somewhat wider.

# If the data would truly follow a Poisson distribution,
# these intervals would have more similar widths.


# Next, we might want to determine which predictors to include in the model.

# One approach is to conduct a likelihood ratio test (LRT).

# Based on the confidence interval, we may want to exclude the "phd" variable.

# We fit a smaller model and test if the coefficient b_phd is zero or nonzero.


# This hypothesis was previously tested when we showed the tidied results
# for `log_lin_fit`.

# That particular approach used results from a single model fit via a 
# Wald statistic (i.e., the parameter divided by its standard error).


# For that approach, the p-value was 0.63.
# We can tidy the results for the Likelihood Ratio Test (LRT) to get
# the p-value:
log_lin_reduced <- log_lin_spec |> 
  fit(formula = art ~ ment + kid5 + fem + mar, data = bioChemists)

stats::anova(
  extract_fit_engine(log_lin_reduced),
  extract_fit_engine(log_lin_fit),
  test = "LRT"
  ) |> 
  broom::tidy()

# The results are the same and, based on these and the confidence intervals for
# this parameter, we will exclude the "phd" variable from further analyses
# since it does not appear to be associated with the outcome.


## 21.4 A more complex model ----

# For count data, there are occasions where the number of zero coutns is
# larger than what a simple Poisson distribution would prescribe.

# A more complex model appropriate for this situation is the
# zero-inflated Poisson (ZIP) model.

# There are two sets of covariates:

# - One for the count data
# - Others that affect the probability (pi) of zeros

# The equation for the mean (lambda) is:

# lambda = 0*pi + (1-pi)*lambda_nz

# where
# log(lambda_nz) = b0 + b1*x1 + ... + bp * xp
# log(pi/(1-pi)) + gamma0 + gamma1*z1 + ... + gammaq*zq

# and the *x* covariates affect the count values while the *z* covariates
# influence the probability of a zero.

# The two sets of predictors do not need to be mutually exclusive.


# We fit a model with a full set of *z* covariates:
zero_inflated_spec <- poisson_reg() |> 
  set_engine("zeroinfl")


zero_inflated_fit <- zero_inflated_spec |> 
  fit(
    formula = art ~ fem + mar + kid5 + ment | fem + mar + kid5 + phd + ment,
    data = bioChemists
    )


print(zero_inflated_fit)

# Since the coefficients for this model are also estimated using 
# maximum likelihood, we want to use another likelihood ratio test
# to understand if the new model terms are helpful.

# We want to *simultaneously test* if:
# H0: gamma1 = gamma2 = ... = gamma5 = 0
# Ha: at least one gamma =/= 0


# We use the `stats::anova()` function:
stats::anova(
  extract_fit_engine(zero_inflated_fit),
  extract_fit_engine(log_lin_reduced),
  test = "LRT"
  ) |> 
  broom::tidy()
# Error in UseMethod("anova") :
# no applicable method for 'anova' applied to object of class "zeroinfl"


# There is no method for the `anova()` function defined for objects
# of the class "zeroinfl"!


# We can instead use an *information criterion statistic* such as the
# Akaike information criterion (AIC).

# This computes the log-likelihood from the training set and penalizes
# the value based on the training set size and the number of model
# parameters.

# In R's prameterization, smaller AIC values are better.

# In this case, we do not conduct a formal statistical test
# but instead estimate the ability of the data to fit the model?


# The results indicate that the ZIP model is preferable:
zero_inflated_fit |> 
  extract_fit_engine() |> 
  AIC()
# 3231.585

log_lin_reduced |> 
  extract_fit_engine() |> 
  AIC()
# 3312.349


# However, it is hard to contextualize this pair of single values and assess
# *how* different they actually are.

# To solve this problem, we resample a large number of these two models.

# From these, we compute the AIC values for each resample
# and then we determine how often the results favor the ZIP model.


# This characterizes the uncertainty of the AIC statistic to gauge
# their difference relative to the noise in the data.


# We can also compute more bootstrap confidence intervals for the parameters
# by specifying the `apparent = TRUE` option when creating the bootstrap samples.

# First, we create 4,000 model fits:
zip_form <- art ~ fem + mar + kid5 + ment | fem + mar + kid5 + phd + ment

glm_form <- art ~ fem + mar + kid5 + ment

class(zip_form) 
# "formula"


set.seed(2104)

# Caution: This will take a moment to run.
bootstrap_models <- bootstraps(
  data = bioChemists,
  times = 2000,
  apparent = TRUE
  ) |> 
  mutate(
    glm = map(splits, ~ fit(log_lin_spec, glm_form, data = analysis(.x))),
    zip = map(splits, ~ fit(zero_inflated_spec, zip_form, data = analysis(.x)))
  )


print(bootstrap_models)


# Now we extract the model fits and their corresponding AIC values:
bootstrap_models <- bootstrap_models |> 
  mutate(
    glm_aic = map_dbl(glm, extract_fit_engine(.x) |> AIC()),
    zip_aic = map_dbl(zip, extract_fit_engine(.x) |> AIC())
  )


bootstrap_models <- bootstrap_models %>%
  mutate(
    glm_aic = map_dbl(glm, ~ extract_fit_engine(.x) %>% AIC()),
    zip_aic = map_dbl(zip, ~ extract_fit_engine(.x) %>% AIC())
  )


mean(bootstrap_models$zip_aic < bootstrap_models$glm_aic)
# 1


# It seems definitive form these results that accounting for the
# excessive number of zero counts is a good idea.


# We could have used `fit_resamples()` or a workflow set to conduct these
# computations.

# In this section we used `mutate()` and `map()` to compute the models
# to demonstrate how one might use "tidymodels" tools for models
# that are not supported by one of the "parsnip" packages.


# Since we have computed the resampled model fits, we can create bootstrap 
# intervals for the zero probability model coefficients, i.e. the gammas.

# We extract these with `broom::tidy()` and use `type = "zero"`:
bootstrap_models <- bootstrap_models |> 
  mutate(
    zero_coefs = map(.x = zip, .f = ~ broom::tidy(.x, type = "zero"))
  )

# One example:
bootstrap_models$zero_coefs[[1]]

# We can visualize the bootstrap distribution of the coefficients:
bootstrap_models |> 
  unnest(zero_coefs) |> 
  ggplot(mapping = aes(x = estimate)) +
  geom_histogram(bins = 25, color = "white") +
  facet_wrap(facets = ~ term, scales = "free_x") +
  geom_vline(xintercept = 0, lty = 2, color = "gray70")


graphics.off()


# One of the covariates, "ment" appears to be important but it has a 
# highly skewed distribution.

# The extra space in some of the facets indicates the presence of outliers
# in the bootstrap estimates.

# The outliers are due only to extreme parameter estimates,
# all models converged.


# The "rsample" package contains a set of functions named `int_*()`
# that compute different types of bootstrap intervals.

# Since the `broom::tidy()` method contains standard error estimates,
# the bootstrap-t intervals can be computed.

# We also compute the standard percentile intervals.
# The 90% confidence intervals are the default.

bootstrap_models |> 
  int_pctl(zero_coefs)


bootstrap_models |> 
  int_t(zero_coefs)

# From these results we get a good idea of which predictor(s) to include
# in the zero count probability model.

# It may be sensible to refit a smaller model to assess if the bootstrap
# distribution for "ment" is still skewed.


## 21.5 More inferential analysis ----

# A variety of Bayesian models are available via "parsnip".

# Additionally, the "multilevelmod" package enables the user to
# fit hierarchical Bayesian and non-Bayesian (e.g. mixed models).


# The "broom.mixed" and "tidybayes" packages are excellent tools for
# extracting data for plots and summaries.

# For data sets with a single hierarchy,such as simple longitudinal or repeated
# measures data, "rsample"'s `group_vfold_cv()` function facilitates
# straightforward out-of-sample characterizations of model performance.


# END
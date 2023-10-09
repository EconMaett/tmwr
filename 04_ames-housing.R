# Chapter 04 - The Ames Housing Data ----

# The Ames housing data set (De Cock, 2011) contains information on
# 2,930 properties in Ames, Iowa, including columns related to:
# - housing characteristics (bedrooms, garage, fireplace, pool, porch, etc.)
# - location (neighborhood)
# - lot information (zoning, shape, size, etc.)
# - ratings of condition and quality
# - sale price


# We can use a transformed version from the "modeldata" package.

# the longitude and latitude values for each property are included,
# and the categorical predictors are converted to R's factor data type.


library(modeldata)
data(ames)

# in one line:
data(ames, package = "modeldata")

dim(ames)
# 2930 74


## 4.1 Exploring features of homes in Ames ----

# The outcome we want to predict is the last sales price of the house (in USD).

library(tidymodels)
tidymodels_prefer()

ggplot(data = ames, mapping = aes(x = Sale_Price)) +
  geom_histogram(bins = 50, col = "white") +
  theme_bw()

graphics.off()

# The data are right-skewed. There are more inexpensive houses than expensive houses.

summary(ames$Sale_Price)
# The median house price
median(ames$Sale_Price)
# 160,000 USD

mean(ames$Sale_Price)
# 180,796,1 USD

max(ames$Sale_Price)
# 755,000 USD

# A log-transformation can be used to make the data more symmetric.

# The log-transformation also ensures that no negative prices can be predicted.

# The logarithmic transformation stabilizes the variance of the data.
ggplot(data = ames, mapping = aes(x = Sale_Price)) +
  geom_histogram(bins = 50, col = "white") +
  scale_x_log10() +
  theme_bw()

graphics.off()

# After the log-transformation, the units of the model coefficients
# are more difficult to interpret.

# The root mean squared error (RMSE) is a common performance metric used
# in regression models.

# If the sale price is on the log scale, the differences
# between the predicted and observed values are also on the log scale.

# An RMSE of 0.15 on the log scale is more difficult to interpet than for
# an untransformed model.

# Despite these drawbacks, the log-transformation is a common approach:
ames <- ames |> 
  mutate(
    Sale_Price = log10(Sale_Price)
  )


# The spatial information is contained in the data in two ways:

# - A qualitative `Neighborhood` label 
# - A quantitative `latitude` and `longitude`


# Some basic questions that could be asked about the data:

# - Is there anything unusual about the distributions of the predictors?
#   Is there more skewness or any pathological distributions?

# - Are there high correlations between predictors?
#   Are some redundant?

# - Are there associations between predictors and the outcomes?


# END
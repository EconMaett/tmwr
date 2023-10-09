# Chapter 02 - A Tidyverse Primer ----

## 21. Tidyverse principles ----

# https://design.tidyverse.org/

### 2.1.1 Design for humans -----

# Consider the task of sorting a data frame.
# Using base R, we might write:
mtcars[order(mtcars$gear, mtcars$mpg), ]

# Using dplyr, we might write:
library(dplyr)
arrange(.data = mtcars, gear, mpg)

# The base R code is more concise, but it’s also harder to read.

# The dplyr code is more verbose, but it’s also easier to read.


### 2.1.2 Reuse existing data structures ----

# Whenever possible, functions should avoid returning a novel data structure.

# The data frame is the preferred data structure in tidyverse and tidymodels.

# Specifically, both packages favor the tibble, a modern reimagining of R's data frame.

# The rsample package can be used to create resamples of a data set,
# such as cross-validation or the bootstrap.

# Three bootstrap samples of a data set might look like:
boot_samp <- rsample::bootstraps(mtcars, times = 3)
boot_samp
# Bootstrap sampling
# A tibble: 3 * 2
# splits          id
# <list>          <chr>
# <split [32/12]> Bootstrap1
# <split [32/12]> Boostrap2
# <split [32/12]> Bootstrap3

class(boot_samp)
# "bootstraps" "rset" "tbl_df" "tbl" "data.frame"

# With this approach, vector-based functions can be used with these columns,
# such as `vapply()` or `purrr::map()`.

# This `boot_samp` object has multiple classes but inherits methods for data frames
# (`"data.frame"`) and tibbles (`"tbl_df"`).

# Most importantly, with tibbles and data frames, new columns can be added
# to the results without affecting the class of the data.

# A downside to relying on such common data structures is the potential loass
# of computational performance.
# Obviously, calculations with simple vectors and matrices are much faster.

# Note that the tibble `boot_samp` includes a list-column `splits`.

boot_samp$splits
# [[1]]
# <Analysis/Assess/Total>
# <32/12/32>

# [[2]]
# <Analysis/Assess/Total>
# <32/12/32>

# [[3]]
# <Analysis/Assess/Total>
# <32/12/32>

class(boot_samp$splits[[1]])
# "boot_split" "rsplit"

# Here, the list-column `splits` contains three element, each of which is
# a class `rsplit` object that contains information about which rows of `mtcars`
# belong in the bootstrap sample.

# The tidymodels packages make extensive use of list-columns.


### 2.1.3 Design for the pipe and functional programming ----

# The `magrittr` pipe operator `%>%` is a tool for chaining together a sequence of R functions.

# Consider the task of sorting a data frame and retaining the first 10 rows:
small_mtcars <- arrange(.data = mtcars, gear)
small_mtcars <- slice(.data = small_mtcars, 1:10)

# more compactly:
small_mtcars <- slice(arrange(.data = mtcars, gear), 1:10)

# Using the pipe-operator:
small_mtcars <- mtcars |> 
  arrange(gear) |> 
  slice(1:10)

# This is more readable and works because all functions return the same data structure.

# In ggplot2, you need to use the plus operator `+` to add layers to a plot:
library(ggplot2)
ggplot(data = mtcars, mapping = aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm")
graphics.off()


# R can be used for functional programming.
# Use this coding style to replace iterative loops, such as when a function
# returns a value without other side-effects.

# Suppose you need the logarithm of the ratio of the fuel efficiency to the car weight.

# Using a loop:
n <- nrow(mtcars)
ratios <- rep(x = NA_real_, times = n) # Pre-allocate a vector to store the results

for (car in 1:n) {
  ratios[car] <- log(mtcars$mpg[car] / mtcars$wt[car])
}

head(ratios)
# 2.081348 1.988470 2.285193 1.895564 1.693052 1.654643


# However, many functions in base R are already vectorized, 
# so you can write:
ratios <- log(mtcars$mpg / mtcars$wt)
head(ratios)
# 2.081348 1.988470 2.285193 1.895564 1.693052 1.654643


# Sometimes the element-wise operation you need is to complex for the
# vectorized functions in base R.

# In this case, you might want to write a function that lets you easily repeat
# the calculation.

# When you design a function that you intend to use in functional programming,
# remember that the output depends only on the inputs and that the function
# should have no side-effects.

# Violations of these ideas will lead to unexpected results.
# Consider the following misguieded example:
compute_log_ratio <- function(mpg, wt) {
  log_base <- getOption("log_base", default = exp(1)) # gets external data
  results <- log(mpg / wt, base = log_base)
  print(mean(results)) # prints to the console
  done <<- TRUE # sets a global variable
  return(results)
}

# This function has multiple side-effects, such as changing
# a global option, printing to the console, and setting a global variable.

# A better version is:
compute_log_ratio <- function(mpg, wt, log_base = exp(1)) {
  return(log(mpg / wt, base = log_base))
}

# The purrr package contains many functions for functional programming.
# The `purrr::map_*()` family of functions operates on a vector and always
# returns the same type of output, specified by the suffix.

# The most basic function, `purrr::map(vector, function)` returns a list.

# We could use `purrr::map()` to compute the log ratio for each car:
purrr::map(.x = head(mtcars$mpg), .f = log)
# But maybe the "list" output is not what we want here.

# We can use `purrr::map_dbl()` to return a vector of doubles,
# since the square root returns a double-precision number:
purrr::map_dbl(.x = head(mtcars$mpg, n = 3), .f = sqrt)
# 4.582576 4.582576 4.774935


# There also exists the `purrr::map2_*(vector1, vector2, function)` family of functions that
# operate on multiple vectors.
log_ratios <- purrr::map2(.x = mtcars$mpg, .y = mtcars$wt, .f = compute_log_ratio)
head(log_ratios)


# The `purrr::map_*(vector, function)` and `purrr::map2_*(vector1, vector2, function)`
# families of functions also allow for temporary, anonymous functions defined
# with the tilde `~` operator:
purrr::map2_dbl(.x = mtcars$mpg, .y = mtcars$wt, .f = ~ log(.x / .y)) |> 
  head()
# 2.081348 1.988470 2.285193 1.895564 1.693052 1.654643


## 2.1 Examples of tidyverse syntax ----

# Tibbles naturally work with column names that are not syntactically valid variable names:

# Wants valid names:
data.frame(`variable 1` = 1:2, two = 3:4)
# `variable 1` was changed to `variable.1`.

# But can be coerced to use them with `check.names = FALSE`:
df <- data.frame(`variable 1` = 1:2, two = 3:4, check.names = FALSE)
df
# `variable 1` unchanged.

# Tibbles do not coerce column names:
tbbl <- tibble(`variable 1` = 1:2, two = 3:4)
tbbl


# data frames enable partial matching of arguments, which tibbles prevent:
df$tw
# 3 4

tbbl$tw
# Error: Unknown columns `tw`
# NULL


# Tibbles do not drop a dimension when they are reduced to a single column:
df[, "two"]
# 3 4

is.vector(df[, "two"])
# TRUE

dim(df[, "two"])
# NULL

tbbl[, "two"]
# two
# <int>
# 3
# 4

dim(tbbl[, "two"])
# 2 1


# Chicago's data portal daily ridershipt data for the city's elevated train stations:
# - station identifier (numeric)
# - station name (character)
# - date (character in mm/dd/yyyy format)
# - day of the week (character)
# - number of riders (numeric)

# The tidyverse pipeline will conduct the following tasks:

# 1. Use `readr::read_csv()` to read the data from the source website and
#    convert them into a tibble.

# 2. Filter the data to eliminate the a few columns like the station ID.
#    Change the column `stationname` to `station`.

# 3. Convert the date field to the R date format using `lubridate::mdy()`.

# 3. Get the maximum number of rides for each sation and day combination.


library(tidyverse)

url <- "https://data.cityofchicago.org/api/views/5neh-572f/rows.csv?accessType=DOWNLOAD&bom=true&format=true"

all_stations <- read_csv(file = url) |> 
  select(station = stationname, date, rides) |> 
  mutate(
    date = mdy(date),
    rides = rides / 1000
  ) |> 
  group_by(date, station) |> 
  summarise(
    rides = max(rides),
    .groups = "drop"
  )

head(all_stations)
# The last step calculated the maximum number of rides for each of the
# 1999 unique combinations of date and station.

# END
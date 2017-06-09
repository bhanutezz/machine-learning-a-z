# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set) #if it is executed directly you will get below error
# Error in colMeans(x, na.rm = TRUE) : 'x' must be numeric
# because during encoding stage, factor method was used which returns factors which are not numeric values in R
# so that we eliminate the columns which are factored from feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

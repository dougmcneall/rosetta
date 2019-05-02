# rosetta_loop_lms.R
# An attempt at regular, Logistic
# and lasso regression on Hugo Lambert's 
# convection parameterisation data.
# dougmcneall@gmail.com

# The Lasso
# http://www.stat.cmu.edu/~ryantibs/datamining/lectures/17-modr2.pdf

# Interpretation of Lasso coefficients
# https://stats.stackexchange.com/questions/225970/interpretation-of-lasso-regression-coefficients


# A gentle introduction to logistic regression and lasso regularisation using R
# https://eight2late.wordpress.com/2017/07/11/a-gentle-introduction-to-logistic-regression-and-lasso-regularisation-using-r/

# read random lines from a large CSV file
# https://www.r-bloggers.com/read-random-rows-from-a-huge-csv-file/


# Input. First 28 columns will be potential temperature (theta) on model
# levels up to tropopause. Level 0 is the one next to the surface.
# Next 28 columns are specific humidity (q). Same levels. All in units of J/kg.

# Output: First 28 are theta increments -- 
#ie the amount the convection scheme changes theta. 
# Second 28 are q increments. Again - all J/kg.

# last 30000 are convecting (and in order of increasing convection).
# In fact with some repeats, particularly at the top end because 
# I chose the cases to span different types of convection with equal weights.
# Works well for model of convecting cases.).
# First 90000 are non-convecting.
# Amount of convection I typically determined by meaning levels 11 to 27  
# (starting at zero or 12 to 28 in the way I've encoded it for you).
# (Can be done with moisture instead. I didn't find that much use.) 
# Typically there'd be about 2 to 3 % convecting cases in actual tropical / subtropical
# data.
#

# This part takes a sample of lines from the csv file without opening the
# whole file. I find it slow on the laptop.
#library(LaF)

#sample1 <- function(file, n) {
#  lf <- laf_open(detect_dm_csv(file, sep = ",", header = TRUE, factor_fraction = -1))
#  return(read_lines(lf, sample(1:nrow(lf), n)))
#}

#sample2 <- function(file, ix) {
#  lf <- laf_open(detect_dm_csv(file, sep = ",", header = TRUE, factor_fraction = -1))
#  return(read_lines(lf, rows = ix))
#}

#allzero = function(x){
#  out = all(x == 0)
#  out
#}

#allequal = function(x){
#  length(unique(x)) == 1
#}

#test = matrix(0, nrow = 3, ncol = 2)
#test[1,1] = 1

#which(apply(test, 2, allzero) ==TRUE)
#which(apply(test, 2, all.equal) ==TRUE)

#setwd("/Users/dougmcneall/Documents/work/R/hugo")


#
#X = sample2(file = 'llcsdougdat.csv', ix = ix)

#load required library
library(glmnet)
library(doMC)
library(parallel)
registerDoMC(cores = 2)

 dat = read.csv(file = 'llcsdougdat.csv')
# dat = read.csv(file = 'llcspcsdougdat.csv')
# There are 12000 lines in the file - we want to sample randomly from them

n = nrow(dat)

sample.size = seq(from = 20000, to = 30000, by = 1000)
lm.hr.vec = rep(NA, length = length(sample.size))
logit.hr.vec = rep(NA, length = length(sample.size))
for(i in 1:length(lm.hr.vec)){
  
ix = sample(1:n, sample.size[i])

# If the sample is from the first 30k lines of the file it is convecting
convect = ix > 90000

# First 56 columns are inputs
X = dat[ix, 1:56] 
X.mat = as.matrix(X)

# get some test data
ix.test = sample(1:n, 1000)
convect.test = ix.test > 90000

X.test = dat[ix.test, 1:56] # data frame
X.test.mat = as.matrix(X.test) # matrix

# Fit linear model
lmfit = lm(convect~., data = X)
# The linear model actually does pretty well at keeping most
# coefficients near zero.

# Linear model prediction
lm.prob = predict(lmfit, newdata = as.data.frame(X.test))
lm.predict =  rep(FALSE, length(convect.test))
lm.predict[lm.prob >.5] = TRUE

lm.tab = table(pred = lm.predict, true = convect.test)

# hit rate
lm.hr = sum(diag(lm.tab)) / sum(lm.tab)
lm.hr.vec[i] = lm.hr

logit.fit = glm(convect~., data = X, family = 'binomial')
logit.prob = predict(logit.fit, newdata = X.test, type = 'response')
logit.predict =  rep(FALSE, length(convect.test))
logit.predict[logit.prob >.5] = TRUE
logit.tab = table(pred = logit.predict, true = convect.test)
logit.hr = sum(diag(logit.tab)) / sum(logit.tab)

}

plot(sample.size, lm.hr.vec, type = 'o')
points(sample.size, logit.hr.vec, type = 'o', col = 'red')


cv.out = cv.glmnet(X.mat,convect,alpha=1,
                   family='binomial',
                   type.measure = 'mse',
                   nfolds = 10, 
                   parallel = TRUE)

plot(cv.out)

#min value of lambda
lambda_min = cv.out$lambda.min
#best value of lambda
lambda_1se = cv.out$lambda.1se
#regression coefficients
coef(cv.out,s=lambda_1se)

# Lasso prediction
lasso_prob = predict(cv.out, newx = X.test.mat,s=lambda_1se,type="response")
lasso_class = predict(cv.out, newx = X.test.mat,s=lambda_1se,type="class")

lasso_predict = rep(FALSE, length(convect.test))
lasso_predict[lasso_prob>.5] = TRUE

lasso.tab = table(pred = lasso_predict, true = convect.test)
#lasso.tab = table(pred = as.logical(lasso_class), true = convect.test)

lasso.hr = sum(diag(lasso.tab)) / sum(lasso.tab)

lasso.hr.vec[i] = lasso.hr







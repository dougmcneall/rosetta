# rosetta_random_forests.R


source("https://raw.githubusercontent.com/dougmcneall/packages-git/master/emtools.R")
source("https://raw.githubusercontent.com/dougmcneall/packages-git/master/imptools.R")
source("https://raw.githubusercontent.com/dougmcneall/packages-git/master/vistools.R")

library(randomForest)
dat = read.csv(file = 'llcsdougdat.csv')
n = nrow(dat)

dat.norm = normalize(dat)

sample.size = 10000
ix = sample(1:n, sample.size)

# If the sample is from the first 30k lines of the file it is convecting
convect = ix > 90000

# First 56 columns are inputs
X = data.frame(dat.norm[ix, 1:56]) 
X.mat = as.matrix(X)

# get some test data
ix.test = sample(1:n, 1000)
convect.test = ix.test > 90000

X.test = data.frame(dat.norm[ix.test, 1:56]) # data frame
X.test.mat = as.matrix(X.test) # matrix

wts = c(0.75, 0.25)

rf.fit = randomForest(x = X, y = as.factor(convect), classwt = 1/wts)

rf.pred = predict(rf.fit, newdata = X.test, type = 'response')
rf.tab = table(pred = rf.pred, true = convect.test)
rf.hr = sum(diag(rf.tab)) / sum(rf.tab)

# How important are the individual (scaled) variables?
plot(importance(rf.fit, type = 2))
#varImpPlot(rf.fit, type = 2)



#library(tidyverse)
library('xgboost')
library('caret')
library('ROSE')
library(easyGgplot2)


setwd('C:/Projects/kaggle/santander')
train = read.table('train.csv', header=T, sep=',')
test = read.table('test.csv', header=T, sep=',')

trainId = train$ID_code
trainTarget = train$target

range01 <- function(x){
    for(i in 1:ncol(x)){
        x[,i] = (x[,i] - min(x[,i]))/(max(x[,i])-min(x[,i]))
    }
    return (x)
}

trials = 100
subsetLength = 10000
auc = c()
spe = c()

for(i in 1:trials){
    idx = sample(1:nrow(train),subsetLength)
    train1 = train[idx, ]
    train1 = range01(train1[,3:ncol(train1)])

    #model <- glm(trainTarget[idx] ~.,family=binomial(link='logit'),data=train1)

    model = xgboost(data = data.matrix(train1, rownames.force = NA),
        label = trainTarget[idx] , nrounds = 200, max_depth = 5, objective = "binary:logistic", verbose = 1, print_every_n = 100, eval_metric = "logloss", early_stopping_rounds = 20)

    #pred = predict(model, test1, type="response")

    idx = sample(1:nrow(train),subsetLength)
    test = range01(train[idx, which(names(train) %in% model$feature_names)])
    pred = predict(model, data.matrix(test))

    pred_hard = ifelse (pred > 0.5, 1, 0)
    spe = c(spe, specificity(table(pred_hard,trainTarget[idx])))
  
    auc = c(auc, roc.curve(trainTarget[idx], pred, plotit = F)$auc)
    print(auc)
}

print(paste("Simulated Result", mean(auc)))

output = data.frame(ID_code=test$ID_code, target=pred_hard)
print(paste("Done ", end.time - start.time))
write.csv(output,'Output.csv',row.names=F)


####################
#library(tidyverse)
library('xgboost')
library('caret')
library('ROSE')

range01 <- function(x){
    for(i in 1:ncol(x)){
        x[,i] = (x[,i] - min(x[,i]))/(max(x[,i])-min(x[,i]))
    }
    return (x)
}

train = read.table('../train.csv', header=T, sep=',')
test = read.table('../test.csv', header=T, sep=',')

trainId = train$ID_code
trainTarget = train$target

train1 = range01(train[,3:ncol(train)])

model = xgboost(data = data.matrix(train1, rownames.force = NA),
    label = trainTarget, nrounds = 200, max_depth = 5, objective = "binary:logistic", verbose = 1, print_every_n = 10, eval_metric = "logloss", early_stopping_rounds = 20)

predTrain = predict(model, data.matrix(train1))

test1 = range01(test[, which(names(test) %in% model$feature_names)])
pred = predict(model, data.matrix(test1))


a = data.frame(c(predTrain, pred))
names(a)[1] = 'Score'
a$Fator = rep('TESTE', nrow(a))
a$Fator[1:length(predTrain)] = rep('TREINO', length(predTrain))

ggplot2.histogram(data=a, xName='Score',
    groupName='Fator', legendPosition="top",
    alpha=0.5, addDensity=TRUE,
    addMeanLine=TRUE, meanLineColor="white", meanLineSize=1.5)


threshold = 0.25
analyse = function (threshold){
    print(table(ifelse (predTrain > threshold, 1, 0),trainTarget))
    print(roc.curve(ifelse (predTrain > threshold, 1, 0), trainTarget, plotit = F)$auc)
    print(specificity(table(ifelse (predTrain > threshold, 1, 0),trainTarget)))
}

pred_hard = ifelse (pred > threshold, 1, 0)

output = data.frame(ID_code=test$ID_code, target=pred_hard)
write.csv(output,'Output.csv',row.names=F)



############################################

# library('tidyverse')
library('xgboost')
library('caret')
library('ROSE')

setwd('C:/Projects/kaggle/santander')

range01 = function(x){
    for(i in 1:ncol(x)){
        x[,i] = (x[,i] - min(x[,i]))/(max(x[,i])-min(x[,i]))
    }
    return (x)
}

splitBal = function(majorityClass, numOfBins){
    binsIdx = createFolds(as.factor(majorityClass$target), k = numOfBins)
    return (binsIdx)
}

clusterBal = function(majorityClass, numOfBins){
    kIdx = kmeans(majorityClass[, -ncol(majorityClass)], numOfBins)$cluster
    binsIdx = list()
    for(j in 1:numOfBins){
        binsIdx[[j]] = which(kIdx == j)
    }
    return (binsIdx)
}

# train = read.table('../train.csv', header=T, sep=',')
# test = read.table('../test.csv', header=T, sep=',')
train = read.table('train.csv', header=T, sep=',')
test = read.table('test.csv', header=T, sep=',')

train1 = cbind(train[,1], range01(train[, -1]))
names(train1)[1] = "ID_code"

idx = sample(1:nrow(train1), nrow(train1) * 0.75)
trainSet = train1[idx, ]
testSet = train1[-idx, ]

testTarget = testSet$target
testVars = testSet[ , -which(names(testSet) %in% c("ID_code", "target"))]

majorityClass = trainSet[which(trainSet$target == 0), ]
minorityClass = trainSet[which(trainSet$target == 1), ]

numOfBins = floor(nrow(majorityClass) / nrow(minorityClass))

binsIdx = splitBal(majorityClass, numOfBins)

bins = list()
for(j in 1:length(binsIdx)){
    bins[[j]] = list()
    bin = rbind(majorityClass[binsIdx[[j]], ], minorityClass)
    bin = bin[sample(nrow(bin)), ]

    bins[[j]]$target = bin$target
    bins[[j]]$vars = bin[ , -which(names(bin) %in% c("ID_code", "target"))]
}

pool = list()
for(j in 1:length(bins)){
    dtrain = xgb.DMatrix(data.matrix(bins[[j]]$vars), label = bins[[j]]$target)
    dtest = xgb.DMatrix(data.matrix(testVars), label = testTarget)
    watchlist = list(train = dtrain, eval = dtest)
    param = list(max_depth = 3, silent = 1, nthread = 2, 
        objective = "binary:logistic", eval_metric = "auc")

    pool[[length(pool)+1]] = xgb.train(param, dtrain, nrounds = 500, print_every_n = 10, early_stopping_rounds = 20, watchlist)
}

predf = rep(0, nrow(testSet))
for(j in 1:length(pool)){
    testSet1 = testSet[, which(names(testSet) %in% pool[[j]]$feature_names)]
    pred = predict(pool[[j]], data.matrix(testSet1))
    predf = predf + pred
}
predf = predf/length(pool)

analyse = function (threshold){
    print(table(ifelse (predf > threshold, 1, 0),testTarget))
    print(roc.curve(ifelse (predf > threshold, 1, 0), testTarget, plotit = F)$auc)
    print(specificity(table(ifelse (predf > threshold, 1, 0),testTarget)))
}

threshold = 0.5
analyse(threshold)

predf = rep(0, nrow(test))
for(j in 1:length(pool)){
    test1 = range01(test[, which(names(test) %in% pool[[j]]$feature_names)])
    pred = predict(pool[[j]], data.matrix(test1))
    predf = predf + pred
}
predf = predf/length(pool)

pred_hard = ifelse (predf > threshold, 1, 0)

output = data.frame(ID_code=test$ID_code, target=predf)
write.csv(output,'Output.csv',row.names=F)

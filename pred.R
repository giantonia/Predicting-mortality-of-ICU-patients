library(xgboost)
library(lime)

train = read.csv('train.csv')
val = read.csv('validation.csv')
X = train[, -c(1,2,3)]
X = X[, -which(colnames(X)=='Outcome')]
Y = train[, 119]
val_pred = val[, -c(1,2,3)]
val_pred = val_pred[, -which(colnames(val_pred)=='Outcome')]
Outcome = val[, 119]

param = list(objective = "binary:logistic",
             eval_metric = 'logloss',
             max_depth = 20,
             eta = 0.5,
             gamma = 0.01)

cv.nround = 100
cv.nfold = 10
mdcv = xgb.cv(data = as.matrix(X),
              label = Y,
              params = param, 
              nthread = 5, 
              nfold = cv.nfold,
              nrounds = cv.nround,
              verbose = T)

model = xgboost(data = as.matrix(X),
                label = Y,
                params = param, 
                nrounds = 300)

pred = predict(model, as.matrix(val_pred))
# temp_score = c()
# for (prob in seq(0.05, 0.95, 0.05)){
#   actual_pred = ifelse(pred>prob, 1, 0)
#   TP = sum((actual_pred==1)&(Outcome==1))
#   FN = sum((actual_pred==0)&(Outcome==1))
#   FP = sum((actual_pred==1)&(Outcome==0))
#   prec = TP/(TP+FP)
#   rec = TP/(TP+FN)
#   temp_score = c(temp_score, min(prec, rec))
# }
# score = max(temp_score)
# print(score)

explainer = lime(X, model)
record = val_pred[10,]
explanation = explain(record, explainer, label = 1, n_features = 5)
png(filename = 'Diagnosis.png')
plot_features(explanation)
dev.off()
library(caret)
library(randomForest)
library(tictoc)
library(doParallel)
library(ggplot2)
library(lattice)

data<-read.csv("pml-training.csv",header = TRUE,stringsAsFactors = FALSE) #train/test data set
testdata<-read.csv("pml-testing.csv",header = TRUE,stringsAsFactors = FALSE) #validation data set
elim<-sapply(testdata,is.logical) #elimination of rows that don't have any information in the 
# validation data set
data<-data[,elim==FALSE]
testdata<-testdata[,elim==FALSE]

nonnw<-data[data$new_window=="no",] #eliminating rows which are means (no inclusion in validation
#so no pint in taking them)
nonnw$classe<-as.factor(nonnw$classe) #convert to factor so it works

set.seed(2234)
inTrian<-createDataPartition(y=nonnw$classe,p=0.75,list=FALSE) 
train1<-nonnw[inTrian,] #train data set
test1<-nonnw[-inTrian,] #test data set
test1$classe<-as.factor(test1$classe)
train1$classe<-as.factor(train1$classe)
my_control <- trainControl(method = "cv", number = 3 ) #Control method for crossvalidation

############ Random Forest with PCA
set.seed(1234)
prep<-preProcess(train1[,-60],method="pca",thresh=.9)
trainpred<-predict(prep,newdata =train1[,-60]) #training data transformation to pca's
train3<-data.frame(trainpred[4:24],train1$classe) #training pca & outcome dataframe
tic('model')
model<-train(train1.classe~.,method="parRF",data=train3,ntree=50,trControl=my_control)
toc()

testpred<-predict(prep,newdata = test1[,-60]) #test data transfromation to pca's
tpred1<-predict(model,testpred[,4:24]) #prediction on test data
confusionMatrix(test1$classe,predict(model,testpred[,4:24]))

ftestpred1<-predict(prep,newdata=testdata[-60]) #validation data transformation to pca's
fpred1<-predict(model,ftestpred1[,4:24]) #prediction on validation data
##############################


############ Just Random Forests
tic('model2')
model2<-train(x=train1[,8:59],y=train1$classe,method="parRF",ntree=50,trControl=my_control)
toc()

tpred2<-predict(model2,test1[,8:59]) #prediction on test data
confusionMatrix(test1$classe,predict(model2,test1[,8:59]))

fpred2<-predict(model2,testdata[,8:59]) #prediction on validation data
##################


############ Just LDA
tic('model3')
model3<-train(x=train1[,8:59],y=train1$classe,method="lda",trControl=my_control)
toc()

tpred3<-predict(model3,test1[,8:59])#prediction on test data
confusionMatrix(test1$classe,predict(model3,test1[,8:59]))

fpred3<-predict(model3,testdata[,8:59]) #prediction on validation data
##################


############# Bagging both
tic('modecomb')
combdf<-data.frame(tpred1,tpred3,tpred2,classe=test1$classe) #test predictions data frame
combmodel<-train(classe~.,data=combdf,method="parRF",ntree=50,trControl=my_control)
#model on models, using test predictions
toc()

confusionMatrix(test1$classe,predict(combmodel,combdf[,1:3])) 

finaldf<-data.frame(fpred1,fpred3,fpred2) # validation predictions of models dataframe
names(finaldf)<-c("tpred1","tpred3","tpred2") #same names as in models so it works
final<-predict(combmodel,finaldf) #prediction of predictions of validation data
###################


varImp(model2)

qplot(roll_belt,pitch_forearm,colour=classe,data=test1,xlim=c(-36,36))

xyplot(pitch_forearm ~ roll_belt | classe, data=test1, layout=c(5,1),xlim=c(-36,36), panel = function(x, y, ...) {
       panel.xyplot(x, y) ## First call the default panel function for 'xyplot'
       panel.abline(h = median(y), lty = 2) ## Add a horizontal line at the median
   })

qplot(classe,roll_belt,data=train1,geom=c("boxplot"),ylim=c(-36,36),color=classe)
qplot(classe,pitch_forearm,data=train1,geom=c("boxplot"),color=classe)


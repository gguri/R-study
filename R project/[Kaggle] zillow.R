#--------------------------------------------------------------------
# 00. setting -------------------------------------------------------
#--------------------------------------------------------------------

library(dplyr)
library(ggplot2)
library(data.table)
install.packages("DT")
library(DT)
library(lubridate)
library(tidyr)
library(leaflet)
install.packages("mice")
library(mice)
install.packages("VIM")
library(VIM)
library(caret)
install.packages("h2o")
library(h2o)
# function
## data type transformations - factoring 범주형
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

# load data
train1 <- fread("C:/자료/스터디/워킹캐글/Zillow Prize/데이터셋/train_2016_v2.csv/train_2016_v2.csv")
propert <- fread("C:/자료/스터디/워킹캐글/Zillow Prize/데이터셋/properties_2016.csv/properties_2016.csv")
sample <- fread("C:/자료/스터디/워킹캐글/Zillow Prize/데이터셋/sample_submission.csv/sample_submission.csv")

set.seed(1711)
#--------------------------------------------------------------------
# 01. feature sumamry -----------------------------------------------
#--------------------------------------------------------------------
# 변수 22개 + logerror 선택
useVar <- c("logerror","taxvaluedollarcnt"
            ,"structuretaxvaluedollarcnt"
            ,"landtaxvaluedollarcnt"
            ,"taxamount"
            ,"taxdelinquencyflag"
            ,"lotsizesquarefeet"
            ,"finishedsquarefeet12"
            ,"calculatedfinishedsquarefeet"
            ,"fullbathcnt"
            ,"bathroomcnt"
            ,"bedroomcnt"
            ,"roomcnt"
            ,"fireplaceflag"
            ,"hashottuborspa"
            ,"latitude"
            ,"longitude"
            ,"fips"
            ,"regionidzip"
            ,"regionidcity"
            ,"regionidcounty"
            ,"yearbuilt")
            
#--------------------------------------------------------------------
# 02. feature engineering -------------------------------------------
#--------------------------------------------------------------------
total <- merge(x = train1, y = propert, by = "parcelid", all.x = TRUE)

# 사용할 22개 변수 추출
total <- total %>%  select(useVar)

# NA 채울 변수 list (7개)
MiceDataVar <- c("taxvaluedollarcnt"
              ,"structuretaxvaluedollarcnt"
              ,"landtaxvaluedollarcnt"
              , "taxamount"
              ,"lotsizesquarefeet"
              , "finishedsquarefeet12"
              ,"calculatedfinishedsquarefeet")

# split data
MiceData <- total %>% select(MiceDataVar)
NaData <- total %>% select(-one_of(MiceDataVar))

#NA Imputation
# mice 결과
miceResult <- mice(MiceData, m = 2)
completeMice <- complete(miceResult, 1)
miceResult <- mice(completeMice, m = 2)
completeMice <- complete(miceResult, 1)


# knn 결과
#install.packages("DMwR")
#library(DMwR)
#knnResult <- knnImputation(naData,k=5)

# 범주화
NaData$hashottuborspa <- ifelse(NaData$hashottuborspa == "", 0, 1)
NaData$fireplaceflag <- ifelse(NaData$fireplaceflag == "", 0 ,1)
NaData$taxdelinquencyflag <- ifelse(NaData$taxdelinquencyflag == "Y", 1,0 )

# train set 생성 (mice결과 적용)
train <- cbind(completeMice, NaData)
#train <- cbind(knnResult, noNaData)


#--------------------------------------------------------------------
# 02. model(select importance var) ----------------------------------
#--------------------------------------------------------------------
# formular 지정
formula.init <- "logerror ~ ."
formula.init <- as.formula(formula.init)

# 5-fold cv
control <- trainControl(method = "cv", number = 5)

### 랜덤포레스트
model_rf <- train(formula.init, data=train, method="rf", 
               trControl=control, na.action = na.omit, importance=TRUE)

#변수별 중요도 확인
importance <- varImp(model_rf) #, scale=FALSE
plot(importance, cex.lab=0.5)
model_rf$modelInfo$varImp


### xgboost
#trainMat<-model.matrix(~.,data = train)
#params<-list(max.depth = 5, eta = 0.3, objective = "reg:linear", eval_metric = "error")
#model_xgb<-xgboost(trainMat, label = trainMat[,9], params = params, nrounds = 5)


#--------------------------------------------------------------------
#TEST SET만들기------------------------------------------------------
#--------------------------------------------------------------------

test_mice <- fread("c:/data/completeMice.csv")

useVar1 <- c("taxvaluedollarcnt" #logerror가 없음
            ,"structuretaxvaluedollarcnt"
            ,"landtaxvaluedollarcnt"
            ,"taxamount"
            ,"taxdelinquencyflag"
            ,"lotsizesquarefeet"
            ,"finishedsquarefeet12"
            ,"calculatedfinishedsquarefeet"
            ,"fullbathcnt"
            ,"bathroomcnt"
            ,"bedroomcnt"
            ,"roomcnt"
            ,"fireplaceflag"
            ,"hashottuborspa"
            ,"latitude"
            ,"longitude"
            ,"fips"
            ,"regionidzip"
            ,"regionidcity"
            ,"regionidcounty"
            ,"yearbuilt")

# Mice를 이용해 NA 채울 변수 list
MiceDataVar <- c("taxvaluedollarcnt"
               ,"structuretaxvaluedollarcnt"
               ,"landtaxvaluedollarcnt"
               , "taxamount"
               ,"lotsizesquarefeet"
               , "finishedsquarefeet12"
               ,"calculatedfinishedsquarefeet")
# 사용할 22개 변수 추출
test_tot <- propert %>%  select(useVar1)
#MiceData <- total %>%   select(MiceDataVar) # mice/knn으로  na대체
NaData <- test_tot %>%   select(-one_of(naDataVar)) #그 외 변수

# TEST SET  mice Imputation
#miceResult <- mice(naData, m = 2)
#completeMice <- complete(miceResult, 1)
#miceResult <- mice(completeMice, m = 5)
#completeMice <- complete(miceResult, 1)
#결과 저장
#write.csv(completeMice, "c:/data/completeMice.csv",sep=",")

# 범주화
NaData$hashottuborspa <- ifelse(NaData$hashottuborspa == "", 0, 1)
NaData$fireplaceflag <- ifelse(NaData$fireplaceflag == "", 0 ,1)
NaData$taxdelinquencyflag <- ifelse(NaData$taxdelinquencyflag == "Y", 1,0 )

#test set 생성
test <- cbind(test_mice, NaData)



## xgboost testset
#testMat<-model.matrix(~.,data = test)
#xgb_pred<-predict(model_xgb, testMat)
#predictions<-xgb_pred


#--------------------------------------------------------------------
#모델 적용 및 제출파일 생성------------------------------------------
#--------------------------------------------------------------------

options(scipen = 10000)

predictions_target <- data.frame(predict(model_rf, propert))

predictions <- round(as.vector(predictions_target$predict), 4)

result <- data.frame(cbind(propert$parcelid, predictions, predictions * .99,
                           predictions * .98, predictions * .97, predictions * .96,
                           predictions * .95))
colnames(result)<-c("parcelid","201610","201611","201612","201710","201711","201712")
options(scipen = 999)
write.csv(result, file = "submission_xgb_1016.csv", row.names = FALSE )






###################################아직 적용안함
######## ADD SOME FEATURES #######

# Dist from centroid
p$longmean <- mean((p$longitude), na.rm=TRUE)/1e6
p$latmean <- mean((p$latitude), na.rm=TRUE)/1e6

# Adjusted long lat
p$longitude1 <- p$longitude/1e6
p$latitude1 <- p$latitude/1e6

# Haversine distance
ncol(p)
p$geodist <- distHaversine(p[,31:32], p[,33:34])
# 하버사인 : 구 위의 최단거리를 구하는 공식
p[,longmean := NULL]
p[,latmean := NULL]
p[,longitude1 := NULL]
p[,latitude1 := NULL]

# Tax based info : 땅 값대비 건물의 가치 (작을 수록 good)
pr[, strValRatio := (pr$structuretaxvaluedollarcnt / (pr$landtaxvaluedollarcnt + pr$structuretaxvaluedollarcnt))]
#'landtaxvaluedollarcnt'	구획의 토지면적에 대한 평가가치		
#'structuretaxvaluedollarcnt'	구획(parcel)에 지어진 건물의 평가가치		

# Bathrooms are important : bathroomcnt의 중요도를 높이기 위해 living area 넓이와 곱하여 상호작용을 만듬
pr[, bathInteraction := (pr$bathroomcnt * pr$calculatedfinishedsquarefeet)]
#'bathroomcnt'	Bathroom 개수		
#'calculatedfinishedsquarefeet'	 Calculated total finished living area of the home 


# Sq Ft / Room : 방 1개당 단위 면적
pr[, sqftRoom := (pr$calculatedfinishedsquarefeet / pr$roomcnt)]
#'calculatedfinishedsquarefeet'	 Calculated total finished living area of the home 
#'roomcnt'	 Total number of rooms in the principal residence


# Struc / Lanad : 집크기대비 마당면적, 값이 작을 수록 시골일 확률이 있고 값이 클수록 도심지 일 수 있음
pr[, strucLand := (pr$calculatedfinishedsquarefeet / pr$lotsizesquarefeet)]
#'calculatedfinishedsquarefeet'	 Calculated total finished living area of the home
#'lotsizesquarefeet'	 Area of the lot in square feet (마당넓이)
#calculatedfinishedsquarefeet 에서 포함되는 것 : 집 전체 전용면적

# Age
pr[, age := 2017 - pr$yearbuilt]

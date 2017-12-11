rm(list=ls())
#' @title adult analyis
#' @author Jungchul HA
#' @version 1.0, 2017.11.15~17
#' @description 
#' 성인 인구조사 데이터

### 00. setting -------------------------
set.seed(1711)

## library
library(dplyr)
library(ggplot2)
library(gridExtra)
library(randomForest)
library(caret)
library(e1071)
library(kernlab)
library(car)
library(rpart)
library(ROCR)
library(gmodels)

## function
source("function/functions.R", encoding = "UTF-8")

# # train data load(원본)
# train <- read.csv(file = "data/raw/train.csv",
#                   header = TRUE,
#                   na.strings = "NA")

# train data load(전처리 완료)
train <- read.csv(file = "data/train.csv",
                  header = TRUE)

# # test data load(원본)
# test <- read.csv(file = "data/raw/test.csv",
#                  header = TRUE,
#                  na.strings = "NA")

# test data load(전처리 완료)
test <- read.csv(file = "data/test.csv",
                 header = TRUE)

### 01. data pre-processing ---------------------
## 1. NA 확인 ------------
colSums(is.na(train))
colSums(is.na(test))

# workclass(노동계급)  : 8  class
# occupation(직업)     : 14 class
# native.country(모국) : 41 class
# 다수의 class를 보유한 범주형 변수들이기 때문에 NA가 포함된 row는 제거

# NA 제거
train <- na.omit(train)
test <- na.omit(test)

# NA 확인
colSums(is.na(train))
colSums(is.na(test))

## 2. 범주형 변수 string 형태 -> number(factor) 형태로 변경 ----------
to.numberFactor <- function(df){
  # data split
  # 수치형 변수
  varNum <- train %>% 
    select(2,4,6,12,13,14)
  
  # 범주형 변수 
  # => string 형태의 데이터를 분석에 사용하기 위한 number(factor)형태로 변경
  varFac <- train %>% 
    select(-2,-4,-6,-12,-13,-14)
  # 각 row 구분하기위한 PK 
  varFac <- data.frame(varFac, num = 1:nrow(varFac))

  # cbind용
  result <- c(num = 1:nrow(varFac))

  # string -> number(factor) 변경 
  for(i in 2){
    ## temp 
    ## [ 바꾸고자 하는 factor변수 + PK(row구분 번호) ]
    temp <- varFac %>% 
      select(i,10)
    
    ## temp1
    ## [ factor 변수 name + factor 변수 class ] 
    var <- sort(unique(temp[,1]))
    fac <- seq(1, length(var), 1)
    temp1 <- data.frame(var, fac)
    colnames(temp1) <- c(colnames(temp[1]), 
                         paste(colnames(temp[1]), "Fac", sep = ""))
    
    ## temp2
    ## [ string 형태 + PK(row구분번호) + 변경된 factor) ]
    # temp 기준 left outer join 
    temp2 <- merge(temp, temp1, by.x = colnames(temp)[1])
    # PK(row구분 번호)로 정렬
    temp2 <- temp2[order(temp2[,2]),]
    # 변환된 factor 변수 result에 cbind
    
    # varFac : string형태 -> result : number(factor) 형태
    result <- cbind(result, temp2[3])
  }
  # 불필요 column 제외
  result <- result[,-1]
  varFac <- varFac[,-10]
  
  # result : number(factor)로 변경한 데이터
  result <- cbind(result, varNum)
  colnames(result) <- c(colnames(varFac), colnames(varNum))
  
  return(result)
}

trainTemp <- to.numberFactor(df = train)
testTemp <- to.numberFactor(df = test)

## 변경 확인 
# 각 factor 변수 group_by, 수치형 변수 sum을 통해 확인
# class, workclass, education, marital.status, occupation, relationship, race, sex, native.country
train %>% 
  group_by(relationship) %>% 
  summarise(count = n(),
            age = sum(age),
            fnlwgt = sum(fnlwgt),
            education.num = sum(education.num),
            capital.gain = sum(capital.gain),
            capital.loss = sum(capital.loss),
            hours.per.week = sum(hours.per.week))

trainTemp %>% 
  group_by(relationship) %>% 
  summarise(count = n(),
            age = sum(age),
            fnlwgt = sum(fnlwgt),
            education.num = sum(education.num),
            capital.gain = sum(capital.gain),
            capital.loss = sum(capital.loss),
            hours.per.week = sum(hours.per.week))

# # 데이터 저장
# write.csv(trainTemp, file = "data/train.csv", row.names = FALSE)
# write.csv(testTemp, file = "data/test.csv", row.names = FALSE)

## 3. data type ------------
glimpse(train)
# 범주형 변수 
factorVars <- train %>% 
  select(-10,-11,-12,-13,-14,-15) %>% 
  colnames()

train <- to.factors(df = train, variables = factorVars)

### 02. EDA --------------------------------------
## 1. class ---------------------------
# 종속변수(수입)
# <=50k : 22654  => 0
# >50k  : 7508   => 1
train$class <- ifelse(train$class == 1,0,1)
table(train$class)

train <- to.factors(df = train, variables = factorVars)
# bar
p1 <- train %>% 
  ggplot(aes(class))+
  geom_bar()

# table
df1 <- data.frame(table(train$class))
colnames(df1) <- c("class", "Freq")
df2 <- data.frame(round(prop.table(table(train$class)),2))
colnames(df2) <- c("class", "Prop")
df <- merge(df1, df2, by = "class")
ndf <- df[order(-df$Freq),]
t1 <- tableGrob(ndf)

grid.arrange(p1, t1, ncol = 2)

# 75%(0) : 50k 이하(<=50k)
# 25%(1) : 50k 초과(>50k) 
# 해당 변수 분류 예측하는 모델을 만드는 것이 최종 목표


## 2. workclass ---------------------------
# 일자리
# 1 : Federal-gov
# 2 : Local-gov
# 3 : Private
# 4 : Self-emp-inc
# 5 : Self-emp-not-inc
# 6 : State-gov
# 7 : Without-pay

# bar
train %>% 
  ggplot(aes(workclass))+
  geom_bar()

# Prviate가 가장 많다.
# 1 : private
# 2 : gov 관련(Federal, Local, State)
# 3 : self 관련(Self-emp-inc, Self-emp-not-inc)
# 4 : without-pay

train$workclass <- case_when(
  train$workclass == 1 ~ 2,
  train$workclass == 2 ~ 2,
  train$workclass == 3 ~ 1,
  train$workclass == 4 ~ 3,
  train$workclass == 5 ~ 3,
  train$workclass == 6 ~ 2,
  train$workclass == 7 ~ 4
)
train <- to.factors(df = train, variables = factorVars)

# CrossTable
CrossTable(train$class, train$work, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(train$class, train$work)

# workclass(일자리)와 관계없이 class 변수는 50k 미만인 사람들이 대부분이다. 
# 카이제곱 검정을 통한 p-value가 0.000인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 3. education ---------------------------
# 교육
# 1 : 10th
# 2 : 11th
# 3 : 12th
# 4 : 1st-4th
# 5 : 5th-6th
# 6 : 7th-8th
# 7 : 9th
# 8 : Assoc-acdm
# 9 : Assoc-voc
# 10: Bachelors(학)
# 11: Doctorate(박)
# 12: HS-grad
# 13: Masters(석)
# 14: Preschool
# 15: Prof-school
# 16: Some-college

# bar
train %>% 
  ggplot(aes(education))+
  geom_bar()

# 고졸(12) > 대재(16) > 학사(10) > 석사(13) > 박사(11) > 고등교육 미만
# 1 : 고등교육 미만
# 2 : 대학교 재학
# 3 : 학사
# 4 : 석,박사

train$education <- case_when(
  train$education == 1 ~ 1,
  train$education == 2 ~ 1,
  train$education == 3 ~ 1,
  train$education == 4 ~ 1,
  train$education == 5 ~ 1,
  train$education == 6 ~ 1,
  train$education == 7 ~ 1,
  train$education == 8 ~ 1,
  train$education == 9 ~ 1,
  train$education == 10~ 3,
  train$education == 11~ 4,
  train$education == 12~ 1,
  train$education == 13~ 4,
  train$education == 14~ 1,
  train$education == 15~ 1,
  train$education == 16~ 2
)
train <- to.factors(df = train, variables = factorVars)

train %>% 
  ggplot(aes(education))+
  geom_bar()

# CrossTable
CrossTable(train$class, train$education, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(train$class, train$education)

# 50K 이상 : 교육수준(education)이 석,박사 이상
# 50k 미만 : 교육수준이 낮아질수록 차지하는 비율이 커짐
# 카이제곱 검정을 통한 p-value가 0.000인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 4. marital.status ---------------------------
# 결혼
# 1 : Divorced(이혼)
# 2 : Married-AF-spous
# 3 : Married-civ-spouse
# 4 : Married-spouse-absent
# 5 : Never-married
# 6 : Separated
# 7 : Widowed

# bar
train %>% 
  ggplot(aes(marital.status))+
  geom_bar()

# 고졸(12) > 대재(16) > 학사(10) > 석사(13) > 박사(11) > 고등교육 미만
# 1 : 결혼한적 없음
# 2 : 결혼함(married)
# 3 : 결혼 후 배우자 없음

train$marital.status <- case_when(
  train$marital.status == 1 ~ 3,
  train$marital.status == 2 ~ 2,
  train$marital.status == 3 ~ 2,
  train$marital.status == 4 ~ 2,
  train$marital.status == 5 ~ 1,
  train$marital.status == 6 ~ 3,
  train$marital.status == 7 ~ 3
)
train <- to.factors(df = train, variables = factorVars)

train %>% 
  ggplot(aes(marital.status))+
  geom_bar()

# CrossTable
CrossTable(train$class, train$marital.status, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(train$class, train$education)

# 50K 이상 : 결혼한 사람의 비율이 40% => 안정적인 삶
# 50k 미만 : 다른 그룹 대부분 50K 미만의 수입을 보여준다.
# 카이제곱 검정을 통한 p-value가 0.000인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 5. occupation ---------------------------
# 직업
# 1 : Adm-clerical(목사)
# 2 : Armed-Forces(무장세력)
# 3 : Craft-repair(수리공)
# 4 : Exec-managerial(경영 간부)
# 5 : Farming-fishing(농어업)
# 6 : Handlers-cleaners(핸들러-클리너)
# 7 : Machine-op-inspct(기계 작동)
# 8 : Other-service(서비스업)
# 9 : Priv-house-serv(개인 집 서비스)
# 10: Prof-specialty(전문 스페셜리스트)
# 11: Protective-serv(서비스 보호) 
# 12: Sales(판매)
# 13: Tech-support(기술지원)
# 14: Transport-moving(수송-이송)

# bar
train %>% 
  ggplot(aes(occupation))+
  geom_bar()

# CrossTable
CrossTable(train$class, train$occupation, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 경영간부(4), 전문 스페셜리스트(10) => 50K 이상의 수입을 내는 비중이 높고
# 나머지 그룹들은 낮다.


## 6. relationship ---------------------------
# 관계
# 1 : Husband	
# 2 : Not-in-family
# 3 : Other-relative
# 4 : Own-child	
# 5 : Unmarried	
# 6 : Wife	

train %>% 
  ggplot(aes(relationship))+
  geom_bar()

# CrossTable
CrossTable(train$class, train$relationship, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# Husband(1), Wife(6) => 50K 이상의 수입을 내는 비중이 높고
# 나머지 그룹들은 낮다.


## 7. race ---------------------------
# 인종
# 1 : Amer-Indian-Eskimo
# 2 : Asian-Pac-Islander
# 3 : Black
# 4 : Other
# 5 : White

train %>% 
  ggplot(aes(race))+
  geom_bar()

# 백인이 대부분
train$race <- ifelse(train$race == 5, 1, 0)

# CrossTable
CrossTable(train$class, train$race, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 인종에 따른 수입 차이는 보이지 않는다.


## 8. sex ---------------------------
# 성별
# 1 : Female
# 2 : Male

train %>% 
  ggplot(aes(sex))+
  geom_bar()

# CrossTable
CrossTable(train$class, train$sex, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 성별에 따른 수입 차이는 보이지 않는다.


## 9. native.country -----------------
# 국가
# 대륙별로 차이가 존재?
asia <- c(1, 3, 17, 19, 20, 24, 25, 30, 36, 37, 40)
north.america <- c(2, 5, 6, 8, 13, 14, 16, 23, 26, 27, 28, 33, 38, 39)
south.america <- c(4, 7, 29)
europe <- c(9, 10, 11, 12, 15, 18, 21, 22, 31, 32, 34, 41)
other <- c(35)

train$native.country[train$native.country %in% asia] <- 1
train$native.country[train$native.country %in% north.america] <- 2
train$native.country[train$native.country %in% south.america] <- 3
train$native.country[train$native.country %in% europe] <- 4
train$native.country[train$native.country %in% other] <- 5

train %>% 
  ggplot(aes(native.country))+
  geom_bar()

## 미국이 대부분이다. 
# 미국 / 그 외 2개 변수로 나눔
train$native.country <- ifelse(train$native.country == 2, 1, 0)
train <- to.factors(df = train, variables = factorVars)

train %>% 
  ggplot(aes(native.country))+
  geom_bar()


## 10. age -----------------
# 나이
train %>% 
  ggplot(aes(age))+
  geom_density()

summary(train$age)

# 20 ~ 40대 다수 분포
# 17 ~ 29 : 20대    => 1
# 30 ~ 39 : 30대    => 2
# 40 ~ 49 : 40대    => 3
# 50 ~ 59 : 50대    => 4
# 60 ~ 90 : 그 이상 => 5

age20 <- c(17:29)
age30 <- c(30:39)
age40 <- c(40:49)
age50 <- c(50:59)
age60 <- c(60:90)

train$age[train$age %in% age20] <- 1
train$age[train$age %in% age30] <- 2
train$age[train$age %in% age40] <- 3
train$age[train$age %in% age50] <- 4
train$age[train$age %in% age60] <- 5

train <- to.factors(df = train, variables = factorVars)

train %>% 
  ggplot(aes(age))+
  geom_bar()

CrossTable(train$class, train$age, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(train$class, train$age)
# 카이제곱 검정을 통한 p-value가 0.000인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 11. fnlwgt ------
## 변수 의미 모호 => 제외

## 12. education.num ----------
# education 변수와 동일한 의미를 가지는 변수 => 제외
temp <- read.csv(file = "data/raw/train.csv",
                 header = TRUE,
                 na.strings = "NA")
sort(table(temp$education))
sort(table(temp$education.num))

## 13. capital.gain, capital loss -----------
train %>% 
  ggplot(aes(capital.gain))+
  geom_density()

train %>% 
  ggplot(aes(capital.loss))+
  geom_density()

# 두 변수 모두 왼쪽에 매우 치우친 분포를 보여준다.
# 분류에 있어 중요한 변수로 작용하지 않을 것 같다.

## 14. hours.per.week ----------
train %>% 
  ggplot(aes(hours.per.week))+
  geom_density()

train %>% 
  ggplot(aes(x = class, y = hours.per.week, colour = class))+
  geom_boxplot()

# boxplot 결과를 볼 때 두 집단에 따른 근무시간의 차이는 보이지 않는다.
# 분류에 있어 중요한 변수로 작용하지 않을 것 같다.


### 03. Model --------------------------------------
# data type 변경
factorVars <- train %>% 
  select(-10,-11,-12,-13,-14,-15) %>% 
  colnames()

train <- to.factors(df = train, variables = factorVars)
test <- to.factors(df = test, variables = factorVars)

# EDA을 통한 feature engineering 수행
train <- feature.engineering(df = train)
test <- feature.engineering(df = test)

# 변수 정규화
numericVar <- train %>% 
  select(10,11,12,13,14,15) %>% 
  colnames()

train <- scale.features(train, numericVar)
test <- scale.features(test, numericVar)

# test data split
testFeature <- test[, -1]
testClass <- test[, 1]


# 특징 선택(feature selection)
# 모델을 만들기 전에 랜덤포레스트을 이용하여 어떤 변수가 분류에 중요한 변수인지 확인
# run feature selection
importanceRF <- feature.selection(feature.vars = train[, -1],
                                  class.var = train[, 1])
importanceRF
head(varImp(importanceRF))

# 중요도가 높은 top5 : education, occupation, marital.status, age, hours.per.week

## 1. Logistic Regression ----------------------
formulaInit <- "class ~ ."
formulaInit <- as.formula(formulaInit)
lrModel <- glm(formula = formulaInit, data = train, family = "binomial")
lrPredict <- predict(lrModel, test, type = "response")
lrPredict <- round(lrPredict)
confusionMatrix(data = lrPredict, reference = testClass, positive = "1")
# 전체 변수 모두 사용
# Accuracy(정확도) : 0.85
# 15060명중 중 <=50k : 10538명, >50k : 2244명을 정확하게 예측하여 정확도가 82%이다.
# sensitivity(민감도) : 0.606
# >50k인 3700명 중 2248명이 제대로 예측되어 민감도가 61%이다.
# specificity(특이도) : 0.927
# <=50k인 11360명 중 10537명이 제대로 예측되어 특이도가 93%이다.

car::vif(lrModel)
# 변수 개수가 많음에도 다중공선성은 없음

# RF importance를 통해 확인한 중요도 top5변수를 사용하여 모델 생성
formulaInit <- "class ~ education + occupation + marital.status + age + hours.per.week"
formulaInit <- as.formula(formulaInit)
lrModel <- glm(formula = formulaInit, data = train, family = "binomial")
lrPredict <- predict(lrModel, test, type = "response")
lrPredict <- round(lrPredict)
confusionMatrix(data = lrPredict, reference = testClass, positive = "1")
car::vif(lrModel)
# RF importance top5 변수를 사용한 결과 
# Accuracy(정확도) : 0.78
# 15060명중 중 <=50k : 10502명, >50k : 1278명을 정확하게 예측하여 정확도가 82%이다.
# sensitivity(민감도) : 0.345
# >50k인 3700명 중 2248명이 제대로 예측되어 민감도가 35%이다.
# specificity(특이도) : 0.924
# <=50k인 11360명 중 10537명이 제대로 예측되어 특이도가 92%이다.

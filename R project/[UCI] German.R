rm(list=ls())
#' @title german credit analysis
#' @author Jungchul HA
#' @version 1.0, 2017.11.11~13
#' @description 
#' 독일 신용 데이터
#' 주제 : EDA 진행 후 신용(credit.rating) 예측 분류 모델 생성

### 00. setting -------------------------
set.seed(1710)

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
# type 변환
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}
to.chars <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.character(df[[variable]])
  }
  return(df)
}
# 표준화
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}
# RF importance
feature.selection <- function(num.iters=20, feature.vars, class.var){
  variable.sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv", 
                        verbose = FALSE, returnResamp = "all", 
                        number = num.iters)
  results.rfe <- rfe(x = feature.vars, y = class.var, 
                     sizes = variable.sizes, 
                     rfeControl = control)
  return(results.rfe)
}
# ROC Curve
plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf,col="black",lty=1, lwd=2,
       main=title.text, cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")
  abline(0,1, col="red")
  auc <- performance(predictions,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend=c(paste0("AUC: ",auc)),cex=0.6,bty = "n",box.col = "white")
  
}

## data load
german <- read.csv(file = "data/german_credit.csv",
                   header = TRUE)

### 01. Data overview --------------------------------------
summary(german)
glimpse(german)

# 수치형 months, amount, age 유지
# 범주형 factor 타입으로 변경
factorVars <- c("credit.rating", "status", "history", 
                "purpose", "savings", "employment", 
                "installment", "personal.status", 
                "guarantor", "residence", "property",
                "other", "housing", "bank.credits",
                "job", "dependents", "telephone", "foreign") 
german <- to.factors(df = german, variables = factorVars)
     
  
### 02. EDA --------------------------------------
## 1. credit.rating ---------------------------
## 신용도
# bar
p1 <- german %>% 
  ggplot(aes(credit.rating))+
  geom_bar()

# table
df1 <- data.frame(table(german$credit.rating))
colnames(df1) <- c("credit.rating", "Freq")
df2 <- data.frame(prop.table(table(german$credit.rating)))
colnames(df2) <- c("credit.rating", "Prop")
df <- merge(df1, df2, by = "credit.rating")
ndf <- df[order(-df$Freq),]
t1 <- tableGrob(ndf)

grid.arrange(p1, t1, ncol = 2)

## 해석
# 30%(0) : 신용도 나쁨(Bad)
# 70%(1) : 신용도 좋음(Good)
# => 종속변수로서 해당 신용도를 분류 예측하는 모델을 만드는 것이 최종 목표


## 2. status ---------------------------
# 고객 계좌 잔고 상태
# DM : 독일 화폐 단위
# 1 : 당좌계좌 없음
# 2 : 잔고 없음 
# 3 : 잔고 200DM 미만
# 4 : 잔고 200DM 이상

# bar
p2 <- german %>% 
  ggplot(aes(status))+
  geom_bar()

# table
df1 <- data.frame(table(german$status))
colnames(df1) <- c("status", "Freq")
df2 <- data.frame(prop.table(table(german$status)))
colnames(df2) <- c("status", "Prop")
df <- merge(df1, df2, by = "status")
ndf <- df[order(-df$Freq),]
t2 <- tableGrob(ndf)

grid.arrange(p2, t2, ncol = 2)

# CrossTable
CrossTable(german$credit.rating, german$status, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)
           
# 잔고가 200DM 이상인 3번 그룹은 전체의 10%로서 나머지 그룹에 비해 작은 비중을 차지한다.
# 3번과 4번 그룹을 합쳐 '잔고를 보유'한 그룹으로 변경

german$status <- case_when(
  german$status == 1 ~ 1,
  german$status == 2 ~ 2,
  german$status == 3 ~ 3,
  german$status == 4 ~ 3
  )
german <- to.factors(df = german, variables = factorVars)
# 1 : 당좌계좌 없음
# 2 : 잔고 없음
# 3 : 잔고 보유

# CrossTable
CrossTable(german$credit.rating, german$status, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$status)

# 잔고를 보유한 90%의 사람들은 '신용이 중요하다'라고 생각한다.
# 카이제곱 검정을 통한 p-value가 0.000인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 3. months ---------------------------
## 신용거래 유지 기간

# density
p3_1 <- german %>% 
  ggplot(aes(months))+
  geom_density()

# boxplot
p3_2 <- german %>% 
  ggplot(aes(x = credit.rating, y = months, colour = credit.rating))+
  geom_boxplot()

grid.arrange(p3_1, p3_2, ncol = 2)

# 신용거래 유지 기간 밀도 그래프를 봤을 때 약 5개정도의 최고점을 가지고 있다.
# credit.rating의 0, 1 그룹별로 boxplot을 그려보면 0인 그룹이 1에 비해 중앙값이 높다.

# 나쁜 신용등급을 가지고 있을 수록 신용거래 유지 기간이 길다.
# 즉, '오랫동안 신용거래를 지속할수록 체납률이 높다.' 라고 생각할 수 있다.


## 4. history ---------------------------
# 신용 거래 상태
# 0 : 과거 지불을 지연함 
# 1 : 중요 계정 / 다른 신용 거래 존재(이 은행 아님)
# 2 : 이전에 남아있던 신용거래에 대해 모두 상환
# 3 : 현재 신용 거래에 대해 문제없음
# 4 : 이전 신용 거래에 대해 상환

# bar
p4 <- german %>% 
  ggplot(aes(history))+
  geom_bar()

# table
df1 <- data.frame(table(german$history))
colnames(df1) <- c("history", "Freq")
df2 <- data.frame(prop.table(table(german$history)))
colnames(df2) <- c("history", "Prop")
df <- merge(df1, df2, by = "history")
ndf <- df[order(-df$Freq),]
t4 <- tableGrob(ndf)

grid.arrange(p4, t4, ncol = 2)

# 각 그룹은 3가지로 묶을 수 있다.
# 0, 1 : 신용 상환에 문제있음                   => 1
# 2    : 과거에 문제가 있었지만 지금은 문제없음 => 2
# 3, 4 : 이 은행의 신용 거래 문제없음           => 3

german$history <- case_when(
  german$history == 0 ~ 1,  
  german$history == 1 ~ 1,
  german$history == 2 ~ 2,
  german$history == 3 ~ 3,
  german$history == 4 ~ 3
)
german <- to.factors(df = german, variables = factorVars)

CrossTable(german$credit.rating, german$history, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$history)

# 신용 상환에 문제가 있는 사람들은 신용 등급(credit.rating)이 좋지 않다.
# 신용 상환에 문제 없는 2, 3그룹은 신용 등급을 중요하게 생각한다.
# 카이제곱 검정을 통한 p-value가 0.000인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 5. puspose ---------------------------
# 신용 거래 목적
# 0 : 그 외
# 1 : 새차
# 2 : 중고차
# 3 : 가구
# 4 : 라디오 또는 텔레비전
# 5 : 가전제품
# 6 : 수리
# 7 : 교육
# 8 : 휴가
# 9 : 재교육
# 10: 사업

# bar
p5 <- german %>% 
  ggplot(aes(purpose))+
  geom_bar()

# table
df1 <- data.frame(table(german$purpose))
colnames(df1) <- c("purpose", "Freq")
df2 <- data.frame(prop.table(table(german$purpose)))
colnames(df2) <- c("purpose", "Prop")
df <- merge(df1, df2, by = "purpose")
ndf <- df[order(-df$Freq),]
t5 <- tableGrob(ndf)

grid.arrange(p5, t5, ncol = 2)

# 4, 5, 7, 8, 10 그룹은 다른 그룹에 비해 매우 낮은 비율을 차지하므로 범주 재분류
# 차량 관련(1,2)     => 1
# 집 관련(3,4,5,6,7) => 2
# 그 외(0,8,9,10)    => 3

german$purpose <- case_when(
  german$purpose == 0 ~ 3,
  german$purpose == 1 ~ 1,
  german$purpose == 2 ~ 1,
  german$purpose == 3 ~ 2,
  german$purpose == 4 ~ 2,
  german$purpose == 5 ~ 2,
  german$purpose == 6 ~ 2,
  german$purpose == 7 ~ 2,
  german$purpose == 8 ~ 3,
  german$purpose == 9 ~ 3,
  german$purpose == 10 ~ 3
)
german <- to.factors(df = german, variables = factorVars)

# CrossTable
CrossTable(german$credit.rating, german$purpose, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$purpose)

# 신용 등급이 나쁜 그룹(0)을 볼 때 집&그 외 요인들이 주요 목적이며
# 차량과 관련된 부분은 상대적으로 가장 낮은 비율을 가지고 있다.
# 카이제곱 검정을 통한 p-value가 0.003인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 6. amount ---------------------------
## 신용 거래량
# bar
p6_1 <- german %>% 
  ggplot(aes(amount))+
  geom_density()

# boxplot
p6_2 <- german %>% 
  ggplot(aes(x = credit.rating, y = amount, colour = credit.rating))+
  geom_boxplot()

grid.arrange(p6_1, p6_2, ncol = 2)

# density : 왼쪽으로 치우친 분포
# boxplot : 신용 등급이 나쁜 그룹(0)이 좋은 그룹(1)에 비해 
#           중앙값이 높으며 신용 거래량을 보인다.

# 신용 등급이 나쁜 고객들이 신용 거래량도 많으며, 상환이 어려울 것이라고 생각된다.


## 7. savings ---------------------------
## 저축
# 1 : 저축 없음
# 2 : 100DM 미만
# 3 : 100DM ~ 499DM
# 4 : 500DM ~ 999DM
# 5 : 1000DM 이상

# bar
p7 <- german %>% 
  ggplot(aes(savings))+
  geom_bar()

# table
df1 <- data.frame(table(german$savings))
colnames(df1) <- c("savings", "Freq")
df2 <- data.frame(prop.table(table(german$savings)))
colnames(df2) <- c("savings", "Prop")
df <- merge(df1, df2, by = "savings")
ndf <- df[order(-df$Freq),]
t7 <- tableGrob(ndf)

grid.arrange(p7, t7, ncol = 2)

# 저축이 없는 그룹이 가장 많다.
# 저축액이 100DM ~ 999DM 사이인 그룹 3, 4의 비율이 가장 적다. => 하나의 그룹으로 합침

german$savings <- case_when(
  german$savings == 1 ~ 1,
  german$savings == 2 ~ 2,
  german$savings == 3 ~ 3,
  german$savings == 4 ~ 3,
  german$savings == 5 ~ 4
)
german <- to.factors(df = german, variables = factorVars)

# CrossTable
CrossTable(german$credit.rating, german$savings, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$savings)

# 저축액이 1000DM 이상인 사람은 좋은 신용 등급을 가진 사람들이 대부분(80%)이다.
# 카이제곱 검정을 통한 p-value가 0.003인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 8. employment ---------------------------
## 고용상태
# 1 : 실직
# 2 : 1년 미만
# 3 : 1년 이상, 4년 미만
# 4 : 4년 이상, 7년 이하
# 5 : 7년 이상

# bar
p8 <- german %>% 
  ggplot(aes(employment))+
  geom_bar()

# table
df1 <- data.frame(table(german$employment))
colnames(df1) <- c("employment", "Freq")
df2 <- data.frame(prop.table(table(german$employment)))
colnames(df2) <- c("employment", "Prop")
df <- merge(df1, df2, by = "employment")
ndf <- df[order(-df$Freq),]
t8 <- tableGrob(ndf)

grid.arrange(p8, t8, ncol = 2)

# CrossTable
CrossTable(german$credit.rating, german$employment, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 근속년수가 높아질수록 신용등급이 높은 사람들의 비율이 높다.
# 신용 등급이 나쁜 그룹(0)은 전체 300의 30%를 차지
# 신용 등급이 좋은 그룹(1)은 전체 700의 20%를 차지

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$employment)

# 실직 혹은 1년 미만인 사람들 중에서
# 나쁜 신용등급을 가진 사람들의 비율이 신용등급이 높은 사람들보다 높은것을 알 수 있다.
# 카이제곱 검정을 통한 p-value가 0.001인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 9. installment ---------------------------
## 급여 중 월납금
# 1 : 35% 이상
# 2 : 25% 이상 ~ 35% 미만
# 3 : 20% 이상 ~ 25% 미만
# 4 : 20% 미만

# bar
p9 <- german %>% 
  ggplot(aes(installment))+
  geom_bar()

# table
df1 <- data.frame(table(german$installment))
colnames(df1) <- c("installment", "Freq")
df2 <- data.frame(prop.table(table(german$installment)))
colnames(df2) <- c("installment", "Prop")
df <- merge(df1, df2, by = "installment")
ndf <- df[order(-df$Freq),]
t9 <- tableGrob(ndf)

grid.arrange(p9, t9, ncol = 2)

# CrossTable
CrossTable(german$credit.rating, german$installment, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$installment)

# 신용 등급 상태에 따라 월납금 차이가 있어보이지는 않는다.
# 카이제곱 검정을 통한 p-value가 0.14인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들은 서로 간에 중요한 연관성을 가지지 않는 것으로 보인다.


## 10. personal.status ---------------------------
## 혼인 여부
# 1 : 이혼한 남성
# 2 : 싱글 남성
# 3 : 기혼/사별한 남성
# 4 : 여성

# bar
p10 <- german %>% 
  ggplot(aes(personal.status))+
  geom_bar()

# table
df1 <- data.frame(table(german$personal.status))
colnames(df1) <- c("personal.status", "Freq")
df2 <- data.frame(prop.table(table(german$personal.status)))
colnames(df2) <- c("personal.status", "Prop")
df <- merge(df1, df2, by = "personal.status")
ndf <- df[order(-df$Freq),]
t10 <- tableGrob(ndf)

grid.arrange(p10, t10, ncol = 2)

# CrossTable
CrossTable(german$credit.rating, german$personal.status, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 이혼한 남성 그룹이 다른 그룹에 비해 낮은편 => 싱글 남성에 통합
# 1 : 이혼한/싱글 남성
# 2 : 기혼/사별한 남성
# 3 : 여성
german$personal.status <- case_when(
  german$personal.status == 1 ~ 1,
  german$personal.status == 2 ~ 1,
  german$personal.status == 3 ~ 2,
  german$personal.status == 4 ~ 3
)
german <- to.factors(df = german, variables = factorVars)

# CrossTable
CrossTable(german$credit.rating, german$personal.status, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$personal.status)

# 남성들(1,2 그룹) 중 결혼을 한 남성들의 신용 등급이 상대적으로 높은 것을 볼 수 있다.
# 카이제곱 검정을 통한 p-value가 0.010인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 11. guarantor ---------------------------
## 보증인
# 1 : 없음
# 2 : 공동 채무자
# 3 : 보증인

# bar
p11 <- german %>% 
  ggplot(aes(guarantor))+
  geom_bar()

# table
df1 <- data.frame(table(german$guarantor))
colnames(df1) <- c("guarantor", "Freq")
df2 <- data.frame(prop.table(table(german$guarantor)))
colnames(df2) <- c("guarantor", "Prop")
df <- merge(df1, df2, by = "guarantor")
ndf <- df[order(-df$Freq),]
t11 <- tableGrob(ndf)

grid.arrange(p11, t11, ncol = 2)

# 공동 채무자 or 보증인이 있는 경우의 비율이 낮으므로 있다/없다로 분류
# 없다 : 0
# 있다 : 1
german$guarantor <- ifelse(german$guarantor == 1, 1, 0)
german <- to.factors(df = german, variables = factorVars)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$guarantor)

# CrossTable
CrossTable(german$credit.rating, german$guarantor, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 신용 등급에 따라 별다른 차이가 있어보이진 않는다.
# 카이제곱 검정을 통한 p-value가 1인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들은 서로 간에 중요한 연관성을 가지지 않는 것으로 보인다.


## 12. residence ---------------------------
## 거주 기간
# 1 : 1년 미만
# 2 : 1년 이상 4년 미만
# 3 : 4년 이상 7년 미만
# 4 : 7년 이상

# bar
p12 <- german %>% 
  ggplot(aes(residence))+
  geom_bar()

# table
df1 <- data.frame(table(german$residence))
colnames(df1) <- c("residence", "Freq")
df2 <- data.frame(prop.table(table(german$residence)))
colnames(df2) <- c("residence", "Prop")
df <- merge(df1, df2, by = "residence")
ndf <- df[order(-df$Freq),]
t12 <- tableGrob(ndf)

grid.arrange(p12, t12, ncol = 2)

# CrossTable
CrossTable(german$credit.rating, german$residence, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$residence)

# 거주 기간에 따라 신용 등급의 변화가 있어보이진 않는다.
# 카이제곱 검정을 통한 p-value가 0.861인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들은 서로 간에 중요한 연관성을 가지지 않는 것으로 보인다.


## 13. property ---------------------------
## 자산
# 1 : 자산 없음
# 2 : 차/그 외
# 3 : 생명보험/저축계약
# 4 : 집/땅 소유권

# bar
p13_1 <- german %>% 
  filter(credit.rating == 0) %>% 
  ggplot(aes(property))+
  geom_bar()+
  ggtitle("신용 Bad(0)")+
  theme(plot.title=element_text(hjust=0.5))

p13_2 <- german %>% 
  filter(credit.rating == 1) %>% 
  ggplot(aes(property))+
  geom_bar()+
  ggtitle("신용 Good(1)")+
  theme(plot.title=element_text(hjust=0.5))

grid.arrange(p13_1, p13_2, ncol = 2)

# CrossTable
CrossTable(german$credit.rating, german$property, digits=1,
           prop.r=F, prop.t=F, prop.chisq=F)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$property)

# 신용 Good과 비교했을때 
# 신용 등급이 나쁠 경우 1번(자산 없음)의 비중이 낮고 4번(집/땅 소유권)의 비중이 높다.
# 카이제곱 검정을 통한 p-value가 0.000인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 14. age ---------------------------
## 나이
# bar
p14_1 <- german %>% 
  ggplot(aes(age))+
  geom_bar()

# density
p14_2 <- german %>% 
  ggplot(aes(age))+
  geom_density()

grid.arrange(p14_1, p14_2, ncol = 2)

# 왼쪽(25세~ 40세)에 치우친 분포를 보인다.

# boxplot
p14_3 <- german %>%
  ggplot(aes(x = credit.rating, y = age, colour = credit.rating))+
  geom_boxplot()

# 신용 등급이 좋지 않은 그룹(0)의 중앙값이 좋은 그룹에 비해 낮다.
# 사회에 제대로 정착하지 못하거나 취업을 준비중인 
# 나이 어린 사람들이 은행으로부터 얻은 신용 대출을 상환하는데 실패했기 때문인 것 같다.


## 15. other ---------------------------
## 다른 곳에서 신용 거래 여부
## 1 : 다른 은행
## 2 : 상점
## 3 : 추가 신용 거래 없음

german %>% 
  ggplot(aes(other))+
  geom_bar()

# 추가 신용 거래가 없는 그룹의 비중이 높다.
# 다른 은행이나 상점에 신용 거래가 있다 없다로 분류
# 없다 : 0
# 있다 : 1

german$other <- case_when(
  german$other == 1 ~ 1,
  german$other == 2 ~ 1,
  german$other == 3 ~ 0
)
german <- to.factors(df = german, variables = factorVars)

german %>% 
  ggplot(aes(other))+
  geom_bar()

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$other)

# 카이제곱 검정을 통한 p-value가 0.000인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.


## 16. housing ---------------------------
## 거주하는 집 종류
# 1 : 공짜 아파트
# 2 : 임대 아파트
# 3 : 자가 주택

p16_1 <- german %>% 
  filter(credit.rating == 0) %>% 
  ggplot(aes(housing))+
  geom_bar()+
  ggtitle("신용 Bad(0)")+
  theme(plot.title=element_text(hjust=0.5))

p16_2 <- german %>% 
  filter(credit.rating == 1) %>% 
  ggplot(aes(housing))+
  geom_bar()+
  ggtitle("신용 Good(0)")+
  theme(plot.title=element_text(hjust=0.5))

grid.arrange(p16_1, p16_2, ncol = 2)

# 2번 그룹이 가장 많고 1번 3번 그룹이 뒤따라옴

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$housing)

# 카이제곱 검정을 통한 p-value가 0.000인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.
# 즉, 집 종류의 차이에 따라 신용 등급 차이는 보이지 않는다.

## 17. bank.credits ---------------------------
## 은행에서 대출 건수
# 1 : 하나
# 2 : 둘/셋
# 3 : 넷/다섯
# 4 : 여섯

p17_1 <- german %>% 
  filter(credit.rating == 0) %>%  
  ggplot(aes(bank.credits))+
  geom_bar()+
  ggtitle("신용 Bad(0)")+
  theme(plot.title=element_text(hjust=0.5))

p17_2 <- german %>% 
  filter(credit.rating == 1) %>%  
  ggplot(aes(bank.credits))+
  geom_bar()+
  ggtitle("신용 Good(0)")+
  theme(plot.title=element_text(hjust=0.5))

grid.arrange(p17_1, p17_2, ncol = 2)

# 한번이 가장 많고 넷/다섯 부터는 비율이 급격하게 줄어듬 => 하나 이상 그룹으로 변경
# 0 : 하나
# 1 : 하나 이상

german$bank.credits <- case_when(
  german$bank.credits == 1 ~ 0,
  german$bank.credits == 2 ~ 1,
  german$bank.credits == 3 ~ 1,
  german$bank.credits == 4 ~ 1
)
german <- to.factors(df = german, variables = factorVars)

german %>% 
  ggplot(aes(bank.credits, fill = credit.rating))+
  geom_bar(position = "dodge")
 
# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$bank.credits)

# 카이제곱 검정을 통한 p-value가 0.169인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들은 서로 간에 중요한 연관성을 가지지 않는 것으로 보인다.
# 즉, 대출 건수에 따라 신용 등급 차이는 보이지 않는다.


## 18. dependents ---------------------------
## 부양가족
# 1 : 0부터 2
# 2 : 3 이상

p18_1 <- german %>% 
  filter(credit.rating == 0) %>%  
  ggplot(aes(dependents))+
  geom_bar()+
  ggtitle("신용 Bad(0)")+
  theme(plot.title=element_text(hjust=0.5))

p18_2 <- german %>% 
  filter(credit.rating == 1) %>%  
  ggplot(aes(dependents))+
  geom_bar()+
  ggtitle("신용 Good(0)")+
  theme(plot.title=element_text(hjust=0.5))

grid.arrange(p18_1, p18_2, ncol = 2)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$dependents)

# 카이제곱 검정을 통한 p-value가 1인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들은 서로 간에 중요한 연관성을 가지지 않는 것으로 보인다.
# 즉, 부양가족 수에 따라 신용 등급 차이는 보이지 않는다.


## 19. telephone ---------------------------
## 휴대폰 소지 여부
# 1 : 없음
# 2 : 있음

p19_1 <- german %>% 
  filter(credit.rating == 0) %>%  
  ggplot(aes(telephone))+
  geom_bar()+
  ggtitle("신용 Bad(0)")+
  theme(plot.title=element_text(hjust=0.5))

p19_2 <- german %>% 
  filter(credit.rating == 1) %>%  
  ggplot(aes(telephone))+
  geom_bar()+
  ggtitle("신용 Good(0)")+
  theme(plot.title=element_text(hjust=0.5))

grid.arrange(p19_1, p19_2, ncol = 2)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$telephone)

# 카이제곱 검정을 통한 p-value가 0.278인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들은 서로 간에 중요한 연관성을 가지지 않는 것으로 보인다.
# 즉, 휴대폰 소지 여부에 따라 신용 등급 차이는 보이지 않는다.


## 20. foreign ---------------------------
## 외국인 근로자 여부
# 1 : 맞다
# 2 : 아니다

p20_1 <- german %>% 
  filter(credit.rating == 0) %>%  
  ggplot(aes(foreign))+
  geom_bar()+
  ggtitle("신용 Bad(0)")+
  theme(plot.title=element_text(hjust=0.5))

p20_2 <- german %>% 
  filter(credit.rating == 1) %>%  
  ggplot(aes(foreign))+
  geom_bar()+
  ggtitle("신용 Good(0)")+
  theme(plot.title=element_text(hjust=0.5))

grid.arrange(p20_1, p20_2, ncol = 2)

# 카이제곱 검정(상관관게 분석)
chisq.test(german$credit.rating, german$foreign)

# 시각화에서는 외국인 근로자 여부에 따라 신용 등급 차이는 보이지 않는다. 
# 하지만 카이제곱 검정을 통한 p-value가 0.015인 것을 보아 유의수준 5%에서 
# 통계적으로 두 변수들 사이에 연관성이나 관계가 존재하는 것으로 보인다.

## Feature Engineering 결과 저장 ----------------
write.csv(test, 
          file = "data/german_credit_final.csv", 
          row.names = FALSE)

### 03. Model --------------------------------------
# Feature engineering이 완료된 data load 
german <- read.csv(file = "data/german_credit_final.csv",
                   header = TRUE)

# data type 변경
factorVars <- c("credit.rating", "status", "history", 
                "purpose", "savings", "employment", 
                "installment", "personal.status", 
                "guarantor", "residence", "property",
                "other", "housing", "bank.credits",
                "job", "dependents", "telephone", "foreign") 
german <- to.factors(df = german, variables = factorVars)

# 변수 정규화
numericVar <- c("months", "amount", "age")
german <- scale.features(german, numericVar)

# data split
index <- sample(1:nrow(german), size = 0.7*nrow(german))
trainData <- german[index, ]
testData <- german[-index, ]

# test data split
testFeature <- testData[, -1]
testClass <- testData[, 1]

# 특징 선택(feature selection)
# 모델을 만들기 전에 랜덤포레스트을 이용하여 어떤 변수가 분류에 중요한 변수인지 확인
# run feature selection
importanceRF <- feature.selection(feature.vars = trainData[, -1],
                                  class.var = trainData[, 1])
importanceRF
head(varImp(importanceRF))

# 중요도가 높은 top5 : status, months, history, savings, amount

## 1. Logistic Regression ----------------------
formulaInit <- "credit.rating ~ ."
formulaInit <- as.formula(formulaInit)
lrModel <- glm(formula = formulaInit, data = trainData, family = "binomial")
lrPredict <- predict(lrModel, testData, type = "response")
lrPredict <- round(lrPredict)
confusionMatrix(data = lrPredict, reference = testClass, positive = "1")
# 19개 변수를 모두 사용
# Accuracy(정확도) : 0.740
# 300명중 중 신용 Good : 189명, 신용 Bad : 33명을 정확하게 예측하여 정확도가 74%이다.
# sensitivity(민감도) : 0.927
# 신용이 Good인 204명 중 189명이 제대로 예측되어 민감도가 93%이다.
# specificity(특이도) : 0.344
# 신용이 Bad인 96명 중 33명이 제대로 예측되어 특이도가 34%이다.

car::vif(lrModel)
# 변수 개수가 많음에도 다중공선성은 없음

# RF importance를 통해 확인한 중요도 top5변수를 사용하여 모델 생성
formulaInit <- "credit.rating ~ status + months + history + savings + amount"
formulaInit <- as.formula(formulaInit)
lrModel <- glm(formula = formulaInit, data = trainData, family = "binomial")
lrPredict <- predict(lrModel, testData, type = "response")
lrPredict <- round(lrPredict)
confusionMatrix(data = lrPredict, reference = testClass, positive = "1")
car::vif(lrModel)
# RF importance top5 변수를 사용한 결과 
# Accuracy(정확도) : 0.753
# 300명중 중 신용 Good : 193명, 신용 Bad : 33명을 정확하게 예측하여 정확도가 75%이다.
# sensitivity(민감도) : 0.946
# 신용이 Good인 204명 중 193명이 제대로 예측되어 민감도가 95%이다.
# specificity(특이도) : 0.344
# 신용이 Bad인 96명 중 33명이 제대로 예측되어 특이도가 34%이다.

# 성능 평가
predictions <- prediction(lrPredict, testClass)
plot.roc.curve(predictions, title.text = "LR ROC Curve")
# AUC : 0.64


## 2. SVM (Support Vector Machine) -----------------
formulaInit <- "credit.rating ~ ."
formulaInit <- as.formula(formulaInit)
svmModel <- svm(formula = formulaInit, data = trainData, kernel = "radial", 
                cost = 100, gamma = 1)
svmPredict <- predict(svmModel, testData[,-1])
confusionMatrix(data = svmPredict, reference = testData[,1], positive = "1")
# Accuracy(정확도) : 0.68
# 300명중 중 신용 Good : 204명은 완벽하게 예측했지만, 신용 Bad는 한명도 예측하지 못했다.
# SVM에 적합한 변수를 확인해서 사용

# 5-kold 2번 반복 -> SVM모델 변수 importance 확인
control <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
model <- train(formulaInit, data = trainData,
               method = "svmRadial", trcontrol = control)
importance <- varImp(model, scale = FALSE)
plot(importance, cex.lab = 0.5)

# top5 변수를 활용한 SVM 모델 생성
svmFormulaInit <- "credit.rating ~ status + history + months + savings + age"
svmFormulaInit <- as.formula(svmFormulaInit)
svmModel <- svm(formula = svmFormulaInit, data = trainData, kernel = "radial", 
                cost = 100, gamma = 1)
svmPredict <- predict(svmModel, testFeature)
confusionMatrix(data = svmPredict, reference = testClass, positive = "1")
# SVM importance top5 변수를 사용한 결과
# Accuracy(정확도) : 0.673
# 300명중 중 신용 Good : 165명, 신용 Bad : 37명을 정확하게 예측하여 정확도가 67%이다.
# Sensitivity(민감도) : 0.809
# 신용이 Good인 204명 중 165명이 제대로 예측되어 민감도가 81%이다.
# Specifictiy(특이도) : 0.385
# 신용이 Bad인 96명 중 37명이 제대로 예측되어 특이도가 39%이다.

# 하이퍼파라미터 튜닝
costWeights <- c(0.1, 10, 100)
gammaWeights <- c(0.01, 0.25, 0.5, 1)
tuningResults <- tune(svm, svmFormulaInit, data = trainData, kerner = "radial")
svmBestModel <- tuningResults$best.model
svmPredictBest <- predict(svmBestModel, testFeature)
confusionMatrix(data = svmPredictBest, reference = testClass, positive = "1")
# 5개 변수 사용 + 하이퍼파라미터 튜닝
# Accuracy(정확도) : 0.713
# 300명중 중 신용 Good : 195명, 신용 Bad : 19명을 정확하게 예측하여 정확도가 71%이다.
# Sensitivity(민감도) : 0.956
# 신용이 Good인 204명 중 195명이 제대로 예측되어 민감도가 96%이다.
# Specifictiy(특이도) : 0.198
# 신용이 Bad인 96명 중 19명이 제대로 예측되어 특이도가 20%이다.

# 성능 평가
svmPredictBest <- predict(svmBestModel, testFeature, decision.values = TRUE)
svmPredictValues <- attributes(svmPredictBest)$decision.values
predictions <- prediction(svmPredictValues, testClass)
plot.roc.curve(predictions, title.text = "SVM ROC Curve")
# AUC : 0.74

## 3. Decision Tree ---------------
formulaInit <- "credit.rating ~ ."
formulaInit <- as.formula(formulaInit)
dtModel <- rpart(formula = formulaInit, method = "class",data = trainData, 
                 control = rpart.control(minsplit = 20, cp = 0.05))
dtPredict <- predict(dtModel, testFeature, type = "class")
confusionMatrix(data = dtPredict, reference = testClass, positive = "1")
# 모든 변수를 사용한 초기 SVM과 동일한 결과
# Accuracy(정확도) : 0.68
# 300명중 중 신용 Good : 204명은 완벽하게 예측했지만, 신용 Bad는 한명도 예측하지 못했다.

# 5-fold cross validation(2회 반복)을 통해 Decision Tree 모델 변수의 importance를 확인
control <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
model <- train(formulaInit, data = trainData, method = "rpart", 
               trControl=control)
importance <- varImp(model, scale=FALSE)
plot(importance, cex.lab = 0.5)


# top5 변수를 사용한 모델 생성
formulaInit <- "credit.rating ~ status + history + months + amount + savings"
formulaInit <- as.formula(formulaInit)
dtModel <- rpart(formula = formulaInit, method = "class", data = trainData, 
                 control = rpart.control(minsplit = 20, cp = 0.05),
                 parms = list(prior = c(0.7, 0.3)))
dtPredict <- predict(dtModel, testFeature, type = "class")
confusionMatrix(data = dtPredict, reference = testClass, positive = "1")
# DT importance top5 변수를 사용했을때 
# accuracy(정확도)는 낮아지지만 sepcificity(특이도)는 높은편
# Accuracy(정확도) : 0.507
# 300명중 중 신용 Good : 67명, 신용 Bad : 85명을 정확하게 예측하여 정확도가 51%이다.
# Sensitivity(민감도) : 0.328
# 신용이 Good인 204명 중 67명이 제대로 예측되어 민감도가 33%이다.
# Specifictiy(특이도) : 0.885
# 신용이 Bad인 96명 중 85명이 제대로 예측되어 특이도가 89%이다.

# 성능 평가
dtPredict <- predict(dtModel, testFeature, type = "prob")
dtPredictValues <- dtPredict[, 2]
predictions <- prediction(dtPredictValues, testClass)
plot.roc.curve(predictions, title.text = "DT ROC Curve")
# AUC : 0.71


## 4. Random Forest --------------------
formulaInit <- "credit.rating ~ ."
formulaInit <- as.formula(formulaInit)
rfModel <- randomForest(formulaInit, data = trainData, importance = T, proximity = T)
rfPredict <- predict(rfModel, testFeature, type="class")
confusionMatrix(data=rfPredict, reference=testClass, positive="1")
# Accuracy(정확도) : 0.753
# 300명중 중 신용 Good : 196명, 신용 Bad : 30명을 정확하게 예측하여 정확도가 75%이다.
# Sensitivity(민감도) : 0.961
# 신용이 Good인 204명 중 914명이 제대로 예측되어 민감도가 96%이다.
# Specifictiy(특이도) : 0.312
# 신용이 Bad인 96명 중 32명이 제대로 예측되어 특이도가 31%이다.

# RF importance top5 변수
formulaInit <- "credit.rating ~ status + months + history + savings + amount"
formulaInit <- as.formula(formulaInit)

# 하이퍼파라미터 튜닝
nodesizeVals <- c(2, 3, 4, 5)
ntreeVals <- c(200, 500, 1000, 2000)
tuningResults <- tune.randomForest(formulaInit, 
                                   data = trainData,
                                   mtry = 3, 
                                   nodesize = nodesizeVals,
                                   ntree = ntreeVals)
rfBestModel <- tuningResults$best.model
rfPredictBest <- predict(rfBestModel, testFeature, type = "class")
confusionMatrix(data = rfPredictBest, reference = testClass, positive = "1")
# Accuracy(정확도) : 0.753
# 300명중 중 신용 Good : 178명, 신용 Bad : 48명을 정확하게 예측하여 정확도가 75%이다.
# Sensitivity(민감도) : 0.873
# 신용이 Good인 204명 중 178명이 제대로 예측되어 민감도가 87%이다.
# Specifictiy(특이도) : 0.500
# 신용이 Bad인 96명 중 45명이 제대로 예측되어 특이도가 50%이다.

# 성능 평가
rfPredictiBest <- predict(rfBestModel, testFeature, type = "prob")
rfPredictiBestValues <- rfPredictiBest[, 2]
predictions <- prediction(rfPredictiBestValues, testClass)
plot.roc.curve(predictions, title.text = "DT ROC Curve")
# AUC : 0.72

### 결론 ----------------------------
# 영향도가 높은 top5 변수 + 튜닝 작업을 완료한 각 모델의 성능
Model <- c("LR", "SVM", "DT", "RF")
Accuracy <- c("75%", "71%", "50%", "75%")
Sensitivity <- c("95%", "96%", "33%", "87%")
Specificity <- c("34%", "20%", "89%", "50%")
result <- data.frame(Model, Accuracy, Sensitivity, Specificity)
modelResult <- tableGrob(result)
grid.arrange(modelResult)

# Logistic Regression 
# Accuracy    75% # Sensitivity 95% # Specificity 34%

# Support Vector Machine
# Accuracy    71% # Sensitivity 96% # Specificity 20%

# Decision Tree
# Accuracy    50% # Sensitivity 33% # Specificity 89%

# Random Forest
# Accuracy    75% # Sensitivity 87% # Specificity 50%


# 해당 주제의 목표는 은행에서 신용 등급이 좋고&나쁨을 사전에 분류 예측하여
# 은행의 신용 위험 리스크를 줄이고자 하는 것으로 볼 수 있다.
# 정확도는 RF(Random Forest)와 LR(Logistic Regression)이 가장 높다.
# Sensitivity가 높은 경우는 신용 등급이 좋은 사람들을 잘 예측한 모델이다.
# Specificity가 높은 경우는 신용 등급이 낮은 사람들을 잘 예측한 모델이다. 
# 비즈니스 관점에서 은행은 신용 거래를 통해 돈을 빌려주고 이자를 통해 수익을 창출한다. 
# Sensitivity : 신용 등급이 높은 사람들을 낮은 사람을 잘못 예측한 경우 
#               은행 입장에서는 큰 손해가 아닐수도 있다.
# 하지만
# Specificity : 신용 등급이 낮은 사람들을 높은 사람으로 잘못 예측한 경우
#               원금 상환과 이자 납부에 있어 문제가 발생할 수도 있다.  
#            => 만약 파산 상태가 발생하여 원금 상환 자체가 불가능할 경우 
#               은행 입장에선 막대한 손해가 발생할 수 있다.
# Decision Tree의 경우 신용 등급이 낮은 사람들을 가장 잘 분류한 모델이지만 
# 정확도 측면에서는 다른 모델에 비해 성능이 많이 떨어진다. 
# Random Forest의 경우 높은 정확도와 상대적으로 신용 등급이 낮은 사람들도 잘 분류한다. 
# 따라서 해당 모델이 4개의 모델중에서는 가장 좋다고 생각된다. 
# 하지만 Decision Tree의 경우 모델이 직관적이고 설명하기 쉽기 때문에 
# 어떤 모델을 최종적으로 결정할지는 문제가 주어진 상황이나 비즈니스 도메인 등 다양한 관점에서 의사결정이 필요하다고 생각된다. 


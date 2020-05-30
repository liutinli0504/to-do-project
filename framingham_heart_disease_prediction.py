# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:05:33 2020

@author: USER
"""
#lasso
from sklearn import linear_model
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X_res,y_res)
yTest_predicted_lasso=lasso.predict(XTest)
print(lasso.coef_)
print(lasso.intercept_)
#lasso_準確率
print("準確率:",lasso.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_lasso,digits=3))
#lasso_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_l_a = confusion_matrix(yTest, yTest_predicted_lasso)
print(c_m_l_a)



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import argmax
#匯入資料
df1 = pd.read_csv('C:/Users/USER/Desktop/framingham.csv')

#刪除教育程度
df = df1.drop("education", axis = 1)
print(df)

#資料集形狀
print(df.shape)
#資料前五筆
print(df.head())
#敘述統計
print(df.describe())
#檢視是否有遺漏值
print(df.info())
df.isnull().values.any()
df.isnull().sum()
#看看每個欄位幾種不同數字
print(df.nunique())
#
print(df.keys())

#填補缺失值(眾數)
df['cigsPerDay'] = df['cigsPerDay'].fillna(int(df['cigsPerDay'].mode()))
df['BPMeds'] = df['BPMeds'].fillna(int(df['BPMeds'].mode()))
df['totChol'] = df['totChol'].fillna(int(df['totChol'].mode()))
a = df['BMI'].mode().mean()
df['BMI'] = df['BMI'].fillna(a)
df['heartRate'] = df['heartRate'].fillna(int(df['heartRate'].mode()))
df['glucose'] = df['glucose'].fillna(int(df['glucose'].mode()))

#新增一欄位no disease，欄位值和disease相反
#np.where用法:對數組disease為0的元素變為1，其餘元素維持不變。
df["no TenYearCHD"]=np.where(df["TenYearCHD"]==0,1,0)
print(df.head())


#EDA:名目順序尺度資料
#sex(male)長條圖(得病或是沒得病的人數或是比率)
fig,axes = plt.subplots(nrows=1, ncols=2)
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["male"]).sum()
df1.plot(kind="bar",ax=axes[0])
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["male"]).mean()
df1.plot(kind="bar",ax=axes[1])

#currentSmoker長條圖(得病或是沒得病的人數或是比率)
fig,axes = plt.subplots(nrows=1, ncols=2)
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["currentSmoker"]).sum()
df1.plot(kind="bar",ax=axes[0])
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["currentSmoker"]).mean()
df1.plot(kind="bar",ax=axes[1])

#BPMeds長條圖(得病或是沒得病的人數或是比率)
fig,axes = plt.subplots(nrows=1, ncols=2)
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["BPMeds"]).sum()
df1.plot(kind="bar",ax=axes[0])
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["BPMeds"]).mean()
df1.plot(kind="bar",ax=axes[1])

#prevalentStroke長條圖(得病或是沒得病的人數或是比率)
fig,axes = plt.subplots(nrows=1, ncols=2)
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["prevalentStroke"]).sum()
df1.plot(kind="bar",ax=axes[0])
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["prevalentStroke"]).mean()
df1.plot(kind="bar",ax=axes[1])

#prevalentHyp長條圖(得病或是沒得病的人數或是比率)
fig,axes = plt.subplots(nrows=1, ncols=2)
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["prevalentHyp"]).sum()
df1.plot(kind="bar",ax=axes[0])
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["prevalentHyp"]).mean()
df1.plot(kind="bar",ax=axes[1])

#Diabetes長條圖(得病或是沒得病的人數或是比率)
fig,axes = plt.subplots(nrows=1, ncols=2)
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["diabetes"]).sum()
df1.plot(kind="bar",ax=axes[0])
df1 = df[["TenYearCHD","no TenYearCHD"]].groupby(df["diabetes"]).mean()
df1.plot(kind="bar",ax=axes[1])


#EDA:連續型資料
#箱形圖:中間是中間值,箱形上緣是75%,下緣是25%,最上方的橫線是最大值,最下方的橫線是最小值,圓形代表異常值
#age箱形圖
df.boxplot(column="age",by="TenYearCHD",figsize=(6,5))
#age綜合箱形圖
df.boxplot(column="age",figsize=(6,5))
#離群值
Q1 = df.age.quantile(q = 0.25) 
Q3 = df.age.quantile(q = 0.75) 
# 基於1.5倍的四分位差計算上下須對應的值 
low_whisker = Q1 - 1.5*(Q3 - Q1) 
up_whisker = Q3 + 1.5*(Q3 - Q1) 
# 尋找異常點 (第幾筆,數字)
df.age[(df.age > up_whisker) | (df.age < low_whisker)]
#KDE圖
sns.kdeplot(df["age"])

#cigsPerDay箱形圖
df.boxplot(column="cigsPerDay",by="TenYearCHD",figsize=(6,5))
#cigsPerDay綜合箱形圖
df.boxplot(column="cigsPerDay",figsize=(6,5))
#離群值
Q1 = df.cigsPerDay.quantile(q = 0.25) 
Q3 = df.cigsPerDay.quantile(q = 0.75) 
# 基於1.5倍的四分位差計算上下須對應的值 
low_whisker = Q1 - 1.5*(Q3 - Q1) 
up_whisker = Q3 + 1.5*(Q3 - Q1) 
# 尋找異常點 (第幾筆,數字)
df.cigsPerDay[(df.cigsPerDay > up_whisker) | (df.cigsPerDay < low_whisker)]
#KDE圖
sns.kdeplot(df["cigsPerDay"])

#totChol箱形圖
df.boxplot(column="totChol",by="TenYearCHD",figsize=(6,5))
#totChol綜合箱形圖
df.boxplot(column="totChol",figsize=(6,5))
#離群值
Q1 = df.totChol.quantile(q = 0.25) 
Q3 = df.totChol.quantile(q = 0.75) 
# 基於1.5倍的四分位差計算上下須對應的值 
low_whisker = Q1 - 1.5*(Q3 - Q1) 
up_whisker = Q3 + 1.5*(Q3 - Q1) 
# 尋找異常點 (第幾筆,數字)
df.totChol[(df.totChol > up_whisker) | (df.totChol < low_whisker)]
#KDE圖
sns.kdeplot(df["totChol"])



#sysBP箱形圖
df.boxplot(column="sysBP",by="TenYearCHD",figsize=(6,5))
#sysBP綜合箱形圖
df.boxplot(column="sysBP",figsize=(6,5))
#離群值
Q1 = df.sysBP.quantile(q = 0.25) 
Q3 = df.sysBP.quantile(q = 0.75) 
# 基於1.5倍的四分位差計算上下須對應的值 
low_whisker = Q1 - 1.5*(Q3 - Q1) 
up_whisker = Q3 + 1.5*(Q3 - Q1) 
# 尋找異常點 (第幾筆,數字)
df.sysBP[(df.sysBP > up_whisker) | (df.sysBP < low_whisker)]
#KDE圖
sns.kdeplot(df["sysBP"])


#diaBP箱形圖
df.boxplot(column="diaBP",by="TenYearCHD",figsize=(6,5))
#diaBP綜合箱形圖
df.boxplot(column="diaBP",figsize=(6,5))
#離群值
Q1 = df.diaBP.quantile(q = 0.25) 
Q3 = df.diaBP.quantile(q = 0.75) 
# 基於1.5倍的四分位差計算上下須對應的值 
low_whisker = Q1 - 1.5*(Q3 - Q1) 
up_whisker = Q3 + 1.5*(Q3 - Q1) 
# 尋找異常點 (第幾筆,數字)
df.diaBP[(df.diaBP > up_whisker) | (df.diaBP < low_whisker)]
#KDE圖
sns.kdeplot(df["diaBP"])

#BMI箱型圖
df.boxplot(column="BMI",by="TenYearCHD",figsize=(6,5))
#BMI綜合箱形圖
df.boxplot(column="BMI",figsize=(6,5))
#離群值
Q1 = df.BMI.quantile(q = 0.25) 
Q3 = df.BMI.quantile(q = 0.75) 
# 基於1.5倍的四分位差計算上下須對應的值 
low_whisker = Q1 - 1.5*(Q3 - Q1) 
up_whisker = Q3 + 1.5*(Q3 - Q1) 
# 尋找異常點 (第幾筆,數字)
df.BMI[(df.BMI > up_whisker) | (df.BMI < low_whisker)]
#KDE圖
sns.kdeplot(df["BMI"])

#heartRate箱型圖
df.boxplot(column="heartRate",by="TenYearCHD",figsize=(6,5))
#heartRate綜合箱形圖
df.boxplot(column="heartRate",figsize=(6,5))
#離群值
Q1 = df.heartRate.quantile(q = 0.25) 
Q3 = df.heartRate.quantile(q = 0.75) 
# 基於1.5倍的四分位差計算上下須對應的值 
low_whisker = Q1 - 1.5*(Q3 - Q1) 
up_whisker = Q3 + 1.5*(Q3 - Q1) 
# 尋找異常點 (第幾筆,數字)
df.heartRate[(df.heartRate > up_whisker) | (df.heartRate < low_whisker)]
#KDE圖
sns.kdeplot(df["heartRate"])

#glucose箱型圖
df.boxplot(column="glucose",by="TenYearCHD",figsize=(6,5))
#glucose綜合箱形圖
df.boxplot(column="glucose",figsize=(6,5))
#離群值
Q1 = df.glucose.quantile(q = 0.25) 
Q3 = df.glucose.quantile(q = 0.75) 
# 基於1.5倍的四分位差計算上下須對應的值 
low_whisker = Q1 - 1.5*(Q3 - Q1) 
up_whisker = Q3 + 1.5*(Q3 - Q1) 
# 尋找異常點 (第幾筆,數字)
df.glucose[(df.glucose > up_whisker) | (df.glucose < low_whisker)]
#KDE圖
sns.kdeplot(df["glucose"])


#建立變異數標準化的實體
from sklearn.preprocessing import StandardScaler
#執行變異數標準化處理
#age標準化
age_1=np.array([df['age']])
age_1=np.reshape(age_1,[4240,1])
df=df.drop('age',axis=1)
stdsc = StandardScaler()
age_norm=stdsc.fit_transform(age_1)
age_norm=pd.DataFrame(age_norm,columns=['age'])
df=pd.concat([df,age_norm],axis=1)


#建立變異數標準化的實體
from sklearn.preprocessing import StandardScaler
#執行變異數標準化處理
#cigsPerDay標準化
cigsPerDay_1=np.array([df['cigsPerDay']])
cigsPerDay_1=np.reshape(cigsPerDay_1,[4240,1])
df=df.drop('cigsPerDay',axis=1)
stdsc = StandardScaler()
cigsPerDay_norm=stdsc.fit_transform(cigsPerDay_1)
cigsPerDay_norm=pd.DataFrame(cigsPerDay_norm,columns=['cigsPerDay'])
df=pd.concat([df,cigsPerDay_norm],axis=1)


#建立變異數標準化的實體
from sklearn.preprocessing import StandardScaler
#執行變異數標準化處理
#totChol標準化
totChol_1=np.array([df['totChol']])
totChol_1=np.reshape(totChol_1,[4240,1])
df=df.drop('totChol',axis=1)
stdsc = StandardScaler()
totChol_norm=stdsc.fit_transform(totChol_1)
totChol_norm=pd.DataFrame(totChol_norm,columns=['totChol'])
df=pd.concat([df,totChol_norm],axis=1)


#建立變異數標準化的實體
from sklearn.preprocessing import StandardScaler
#執行變異數標準化處理
#sysBP標準化
sysBP_1=np.array([df['sysBP']])
sysBP_1=np.reshape(sysBP_1,[4240,1])
df=df.drop('sysBP',axis=1)
stdsc = StandardScaler()
sysBP_norm=stdsc.fit_transform(sysBP_1)
sysBP_norm=pd.DataFrame(sysBP_norm,columns=['sysBP'])
df=pd.concat([df,sysBP_norm],axis=1)

#建立變異數標準化的實體
from sklearn.preprocessing import StandardScaler
#執行變異數標準化處理
#diaBP標準化
diaBP_1=np.array([df['diaBP']])
diaBP_1=np.reshape(diaBP_1,[4240,1])
df=df.drop('diaBP',axis=1)
stdsc = StandardScaler()
diaBP_norm=stdsc.fit_transform(diaBP_1)
diaBP_norm=pd.DataFrame(diaBP_norm,columns=['diaBP'])
df=pd.concat([df,diaBP_norm],axis=1)


#建立變異數標準化的實體
from sklearn.preprocessing import StandardScaler
#執行變異數標準化處理
#BMI標準化
BMI_1=np.array([df['BMI']])
BMI_1=np.reshape(BMI_1,[4240,1])
df=df.drop('BMI',axis=1)
stdsc = StandardScaler()
BMI_norm=stdsc.fit_transform(BMI_1)
BMI_norm=pd.DataFrame(BMI_norm,columns=['BMI'])
df=pd.concat([df,BMI_norm],axis=1)

#建立變異數標準化的實體
from sklearn.preprocessing import StandardScaler
#執行變異數標準化處理
#heartRate標準化
heartRate_1=np.array([df['heartRate']])
heartRate_1=np.reshape(heartRate_1,[4240,1])
df=df.drop('heartRate',axis=1)
stdsc = StandardScaler()
heartRate_norm=stdsc.fit_transform(heartRate_1)
heartRate_norm=pd.DataFrame(heartRate_norm,columns=['heartRate'])
df=pd.concat([df,heartRate_norm],axis=1)


#建立變異數標準化的實體
from sklearn.preprocessing import StandardScaler
#執行變異數標準化處理
#glucose標準化
glucose_1=np.array([df['glucose']])
glucose_1=np.reshape(glucose_1,[4240,1])
df=df.drop('glucose',axis=1)
stdsc = StandardScaler()
glucose_norm=stdsc.fit_transform(glucose_1)
glucose_norm=pd.DataFrame(glucose_norm,columns=['glucose'])
df=pd.concat([df,glucose_norm],axis=1)


#去除no disease欄位
df = df.drop("no TenYearCHD", axis=1)

#將disease欄位移至最後
df_TenYearCHD = df.TenYearCHD
df = df.drop('TenYearCHD',axis=1)
df.insert(14,'TenYearCHD',df_TenYearCHD)

#相關性圖
#annot把數值填入
fig=plt.figure(figsize=(18,18))
sns.heatmap(df.corr(),annot=True,fmt='.2f',linewidths=2)



####採樣
####smote
X = df.loc[:, df.columns != 'TenYearCHD']
y = df.loc[:, df.columns == 'TenYearCHD']

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(df.iloc[:,0:-1], df[['TenYearCHD']], test_size=0.3, random_state=0)
X_res, y_res = SMOTE(random_state=0).fit_resample(XTrain, yTrain)

# 確認資料筆數
print("length of oversampled data is ",len(X_res))
print("Number of no subscription in oversampled data",len(y_res[y_res['TenYearCHD']==0]))
print("Number of subscription in oversampled data",len(y_res[y_res['TenYearCHD']==1]))
print("Proportion of no subscription data in oversampled data is ",len(y_res[y_res['TenYearCHD']==0])/len(X_res))
print("Proportion of subscription data in oversampled data is ",len(y_res[y_res['TenYearCHD']==1])/len(X_res))


######Model
#Decision Tree
from sklearn import tree
DT=tree.DecisionTreeClassifier(max_depth=8,random_state=1)
DT.fit(X_res, y_res)
yTest_predicted_DT=DT.predict(XTest)
#DT_準確率
print("準確率:",DT.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_DT,digits=3))
#DT_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_d = confusion_matrix(yTest, yTest_predicted_DT)
print(c_m_d)

# Calculate feature importances
importances = DT.fit(X_res, y_res).feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(XTrain.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(XTrain.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)


#pr curve for Decision Tree
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = DT.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Decision Tree')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for Decision Tree
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = DT.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Decision Tree')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()



#Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
LR = linear_model.LogisticRegression()
LR.fit(X_res, y_res)
yTest_predicted_LR = LR.predict(XTest)
#LR_準確率
print("準確率:",LR.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_LR,digits=3))
#LR_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_l = confusion_matrix(yTest, yTest_predicted_LR)
print(c_m_l)
print("迴歸係數", LR.coef_)      
print("截距:", LR.intercept_)


# Calculate feature importances
importances = LR.coef_[0]
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(X_res.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(X_res.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)


#pr curve for Logistic regression
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = LR.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic regression')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for Logistic regression
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = LR.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic regression')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()


#SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear',probability=True)
svm.fit(X_res, y_res)
yTest_predicted_SVM = svm.predict(XTest)
#SVM_準確率
print("準確率:",svm.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_SVM,digits=3))
#SVM_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_s = confusion_matrix(yTest, yTest_predicted_SVM)
print(c_m_s)


# Calculate feature importances
importances = svm.coef_[0]
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(X_res.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(X_res.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)


#pr curve for SVM
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = svm.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='SVM')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for SVM
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = svm.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='SVM')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()



#Random forest
class_weight={0:0.00005,1:150}
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 1000, random_state = 1,class_weight = class_weight)
RF.fit(X_res, y_res)
yTest_predicted_RF = RF.predict(XTest)
#RF_準確率
print("準確率",RF.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_RF,digits=3))
#RF_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_r = confusion_matrix(yTest, yTest_predicted_RF)
print(c_m_r)


# Calculate feature importances
importances = RF.fit(X_res, y_res).feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(X_res.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(X_res.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)

#pr curve for random forest
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = RF.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Random Forest')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for random forest
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = RF.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Random Forest')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()


#Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_res, y_res)
yTest_predicted_NB = NB.predict(XTest)
#NB_準確率
print("準確率:",NB.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_NB,digits=3))
#NB_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_n = confusion_matrix(yTest, yTest_predicted_NB)
print(c_m_n)

#pr curve for random forest
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = NB.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Naive Bayes Algorithm')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for random forest
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = NB.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Naive Bayes Algorithm')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()


#xgboost
import xgboost
xg = xgboost.XGBClassifier()
xg.fit(X_res, y_res)
yTest_predicted_xg = xg.predict(XTest)
#XG_準確率
print("準確率:",xg.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_xg,digits=3))
#XG_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_x = confusion_matrix(yTest, yTest_predicted_xg)
print(c_m_x)


# Calculate feature importances
importances = xg.fit(X_res, y_res).feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(XTrain.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(XTrain.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)

#pr curve for XGBoost
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = xg.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='XGBoost')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for xgboost
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = xg.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='XGBoost')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()



#light GBM
import lightgbm as lgb
LGB = lgb.LGBMClassifier()
LGB.fit(X_res, y_res)
yTest_predicted_LGB = LGB.predict(XTest)
#LGB_準確率
print("準確率:",LGB.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_LGB,digits=3))
#LGB_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_l_g = confusion_matrix(yTest, yTest_predicted_LGB)
print(c_m_l_g)

# Calculate feature importances
importances = LGB.fit(X_res, y_res).feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(X_res.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(X_res.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)

#pr curve for light GBM
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = LGB.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='light GBM')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for light GBM
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = LGB.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='light GBM')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()


####undersampling
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(df.iloc[:,0:-1], df[["TenYearCHD"]],test_size=0.3,random_state=1)

TenYearCHD_count = df.TenYearCHD.value_counts()
print('Class 0:', TenYearCHD_count[0])
print('Class 1:', TenYearCHD_count[1])
print('Proportion:', round(TenYearCHD_count[0] / TenYearCHD_count[1], 2), ': 1')
TenYearCHD_count.plot(kind='bar', title='Count (TenYearCHD)');

df_xtrain_ytrain = pd.concat([XTrain, yTrain], axis=1)
# Class count
count_class_0, count_class_1 = df_xtrain_ytrain.TenYearCHD.value_counts()
# Divide by class
df_xtrain_ytrain_class_0 = df_xtrain_ytrain[df_xtrain_ytrain['TenYearCHD'] == 0]
df_xtrain_ytrain_class_1 = df_xtrain_ytrain[df_xtrain_ytrain['TenYearCHD'] == 1]

df_class_0_under = df_xtrain_ytrain_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_xtrain_ytrain_class_1], axis=0)

XTrain=df_test_under.iloc[:,0:-1]
yTrain=df_test_under[['TenYearCHD']]

print('Random under-sampling:')
print(df_test_under.TenYearCHD.value_counts())

df_test_under.TenYearCHD.value_counts().plot(kind='bar', title='Count (TenYearCHD)');



###oversampling
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(df.iloc[:,0:-1], df[["TenYearCHD"]],test_size=0.3,random_state=1)

TenYearCHD_count = df.TenYearCHD.value_counts()
print('Class 0:', TenYearCHD_count[0])
print('Class 1:', TenYearCHD_count[1])
print('Proportion:', round(TenYearCHD_count[0] / TenYearCHD_count[1], 2), ': 1')
TenYearCHD_count.plot(kind='bar', title='Count (TenYearCHD)');

df_xtrain_ytrain = pd.concat([XTrain, yTrain], axis=1)
# Class count
count_class_0, count_class_1 = df_xtrain_ytrain.TenYearCHD.value_counts()
# Divide by class
df_xtrain_ytrain_class_0 = df_xtrain_ytrain[df_xtrain_ytrain['TenYearCHD'] == 0]
df_xtrain_ytrain_class_1 = df_xtrain_ytrain[df_xtrain_ytrain['TenYearCHD'] == 1]

df_class_1_over = df_xtrain_ytrain_class_1.sample(count_class_0,replace = True)
df_test_over = pd.concat([df_class_1_over, df_xtrain_ytrain_class_0], axis=0)

XTrain=df_test_over.iloc[:,0:-1]
yTrain=df_test_over[['TenYearCHD']]

print('Random over-sampling:')
print(df_test_over.TenYearCHD.value_counts())

df_test_over.TenYearCHD.value_counts().plot(kind='bar', title='Count (TenYearCHD)');





######Model

#Decision Tree
from sklearn import tree
DT=tree.DecisionTreeClassifier(max_depth=4,random_state=1)
DT.fit(XTrain,yTrain)
yTest_predicted_DT=DT.predict(XTest)
#DT_準確率
print("準確率:",DT.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_DT,digits=3))
#DT_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_d = confusion_matrix(yTest, yTest_predicted_DT)
print(c_m_d)

# Calculate feature importances
importances = DT.fit(XTrain,yTrain).feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(XTrain.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(XTrain.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)


#pr curve for Decision Tree
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = DT.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Decision Tree')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for Decision Tree
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = DT.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Decision Tree')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()



#Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
LR = linear_model.LogisticRegression()
LR.fit(XTrain,yTrain)
yTest_predicted_LR = LR.predict(XTest)
#LR_準確率
print("準確率:",LR.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_LR,digits=3))
#LR_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_l = confusion_matrix(yTest, yTest_predicted_LR)
print(c_m_l)
print("迴歸係數", LR.coef_)      
print("截距:", LR.intercept_)


# Calculate feature importances
importances = LR.coef_[0]
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(XTrain.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(XTrain.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)


#pr curve for Logistic regression
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = LR.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic regression')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for Logistic regression
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = LR.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic regression')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()


#SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear',probability=True)
svm.fit(XTrain, yTrain)
yTest_predicted_SVM = svm.predict(XTest)
#SVM_準確率
print("準確率:",svm.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_SVM,digits=3))
#SVM_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_s = confusion_matrix(yTest, yTest_predicted_SVM)
print(c_m_s)


# Calculate feature importances
importances = svm.coef_[0]
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(XTrain.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(XTrain.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)


#pr curve for SVM
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = svm.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='SVM')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for SVM
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = svm.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='SVM')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()



#Random forest  
class_weight={0:0.00001,1:70}
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 1000,random_state = 1,class_weight=class_weight)
RF.fit(XTrain, yTrain)
yTest_predicted_RF = RF.predict(XTest)
#RF_準確率
print("準確率",RF.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_RF,digits=3))
#RF_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_r = confusion_matrix(yTest, yTest_predicted_RF)
print(c_m_r)


# Calculate feature importances
importances = RF.fit(XTrain,yTrain).feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(XTrain.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(XTrain.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)

#pr curve for random forest
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = RF.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Random Forest')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for random forest
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = RF.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Random Forest')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()


#Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(XTrain, yTrain)
yTest_predicted_NB = NB.predict(XTest)
#NB_準確率
print("準確率:",NB.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_NB,digits=3))
#NB_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_n = confusion_matrix(yTest, yTest_predicted_NB)
print(c_m_n)

#pr curve for random forest
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = NB.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Naive Bayes Algorithm')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for random forest
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = NB.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='Naive Bayes Algorithm')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()


#xgboost
import xgboost
xg = xgboost.XGBClassifier()
xg.fit(XTrain, yTrain)
yTest_predicted_xg = xg.predict(XTest)
#XG_準確率
print("準確率:",xg.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_xg,digits=3))
#XG_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_x = confusion_matrix(yTest, yTest_predicted_xg)
print(c_m_x)


# Calculate feature importances
importances = xg.fit(XTrain,yTrain).feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(XTrain.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(XTrain.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)

#pr curve for XGBoost
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = xg.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='XGBoost')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for xgboost
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = xg.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='XGBoost')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()



#light GBM
import lightgbm as lgb
LGB = lgb.LGBMClassifier()
LGB.fit(XTrain, yTrain)
yTest_predicted_LGB = LGB.predict(XTest)
#LGB_準確率
print("準確率:",LGB.score(XTest,yTest))
#classification report
from sklearn import metrics
print(metrics.classification_report(yTest,yTest_predicted_LGB,digits=3))
#LGB_confusion_matrix
from sklearn.metrics import confusion_matrix
c_m_l_g = confusion_matrix(yTest, yTest_predicted_LGB)
print(c_m_l_g)

# Calculate feature importances
importances = LGB.fit(XTrain,yTrain).feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]
for i in indices:
    print(df.columns[i],":       ",importances[i])
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(XTrain.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(XTrain.shape[1]), names, rotation=90)
# Show plot
plt.show()
#印出第幾個
print(indices)
#印出重要機率
print(importances)

#pr curve for light GBM
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = LGB.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='light GBM')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()

#pr curve for light GBM
from sklearn.metrics import precision_recall_curve
#predict probabilities
yhat = LGB.predict_proba(XTest)
#predict probabilities for the positive outcome only
yhat = yhat[:,1]
#calculate the pr-curve
precision, recall , thresholds = precision_recall_curve(yTest,yhat)
#convert to f-measure
fscore = (2 * precision * recall ) / (precision + recall)
#locate the index of the largest f-measure
ix = argmax(fscore)
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix],fscore[ix]))
#plot precision recall curve for the model
no_skill = len(yTest[yTest==1])/len(yTest)
plt.plot([0,1],[no_skill,no_skill],linestyle='--',label='No Skill')
plt.plot(recall, precision, marker='.', label='light GBM')
plt.scatter(recall[ix], precision[ix], marker='o',color= 'black', label='Best')
#axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#show the plot
plt.show()


#雙變量分析
sns.set(style="ticks", color_codes=True)
sns.pairplot(df)


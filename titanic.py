# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 21:17:16 2021

@author: asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import seaborn as sns

plt.rc("font",family='Arial Unicode MS')

#读取train的数据
train_data=pd.read_csv(r"E:\titanic\train.csv",header=[0])
print(train_data.head())

#读取test的数据
test_data=pd.read_csv(r"E:\titanic\test.csv",header=[0])
print(test_data.head())

#查看train和test的基本信息
print(train_data.info())
print(test_data.info())

#查看train和test数据列不同值的数量
print(train_data.nunique())
print(test_data.nunique())

#删除不需要的列Ticket
tnd=train_data.drop(['Ticket'],axis=1,inplace=True)
print(tnd)
td=test_data.drop(['Ticket'],axis=1,inplace=True)
print(td)

#统计生存者的数量
pd1=pd.DataFrame(train_data.Survived.value_counts(normalize=True)*100)
print(pd1)
train_data.Survived.value_counts().plot(kind='bar',xlabel='生存情况',ylabel='人数',width=0.4,color='g',alpha=0.6)

#统计男女年龄数
pd2=pd.DataFrame(train_data.Sex.value_counts())
print(pd2)

#统计生存概率
pd0=pd.DataFrame(train_data.Survived.value_counts(normalize=True)*100)
print(pd0)

#统计男女生存概率
pd3=pd.DataFrame(train_data.groupby(['Sex'])['Survived'].value_counts(normalize=True)*100)
print(pd3)
pd.crosstab(train_data.Sex,train_data.Survived).plot(kind='bar',ylabel='人数',width=0.4,color=['g','y'],alpha=0.6)

#统计经济地位高低生存概率
pd4=pd.DataFrame(train_data.groupby(['Pclass'])['Survived'].value_counts(normalize=True)*100)
print(pd4)
pd.crosstab(train_data.Pclass,train_data.Survived).plot(kind='bar',ylabel='人数',width=0.4,color=['g','y'],alpha=0.6)

#统计经济地位高低性别生存概率
pd5=pd.DataFrame(train_data.groupby(['Pclass'])['Sex'].value_counts())
print(pd5)

#统计登船SCQ的人数
train_data.Embarked.value_counts().plot(kind='line',xlabel='Starting Point',ylabel='人数',color='b',alpha=0.6,linestyle='--')

#统计登场SCQ生存/死亡人数
pd.crosstab(train_data.Survived,train_data.Embarked).plot(kind='bar',xlabel='Survived',ylabel='人数',width=0.4,color=['g','y','c'],alpha=0.6)

#统计登SCQ各性别生存/死亡数量
pd6=pd.DataFrame(train_data.groupby(['Embarked','Sex'])['Survived'].value_counts())
print(pd6)

#统计各列数据为空的总数
print(train_data.isna().sum())
print(test_data.isna().sum())

#填充缺失值
train_data.Embarked.fillna('S',inplace=True)
print(train_data.isna().sum())

#填充Fare的缺失值
print(test_data.groupby(['Embarked','Pclass'])['Fare'].describe())

print(test_data[test_data['Fare'].isna()])

test_data.Fare.fillna(test_data[(test_data['Pclass']==3)&(test_data['Embarked']=='S')].Fare.median(),inplace=True)
print(test_data.isna().sum())

#填充Cabin的缺失值
train_Fare_Grp=pd.qcut(train_data.Fare,q=4,labels=['Economy','Economy Plus','Business','First'])
train_data['Fare_Gp']=train_Fare_Grp
print(train_data.head())

test_Fare_Grp=pd.qcut(test_data.Fare,q=4,labels=['Economy','Economy Plus','Business','First'])
test_data['Fare_Gp']=test_Fare_Grp
print(test_data.head())

def cabin_fill(df):
    for i in range(len(df)):
        if(df['Cabin'].isna()[i]):
            fgp=df.iloc[i,:]['Fare_Gp']
            pcl=df.iloc[i,:]['Pclass']
            val=df[(df['Fare_Gp']==fgp)|(df['Pclass']==pcl)].Cabin.mode().values[0]
            df['Cabin'].iloc[i]=val
    return (df)

cabin_fill(train_data)
print(train_data.isna().sum())

cabin_fill(test_data)
print(test_data.isna().sum())


def title(df):
    title=[]
    for i in range(len(df)):
        tokens=df.iloc[i,:]['Name'].split(',')
        title.append(tokens[1].split(' ')[1])
    df['Title']=title
    return (df)

title(train_data)
print(train_data.head())

title(test_data)
print(test_data.head())

tdg=train_data.groupby(['Title','Sex'])['Age'].describe()
print(tdg)

tdg2=test_data.groupby(['Title','Sex'])['Age'].describe()
print(tdg2)

print(train_data.Title.value_counts())

train_data.Title.replace(['Mlle.','Mme.','Ms.','Major.','Lady.','Jonkheer.',
                          'Col.','Rev.','Capt.','Sir.','Don.','the','Dr.'],
                         ['Miss.','Miss.','Mrs.','Other','Mrs.','Mr.','Other',
                          'Other','Other','Other','Mr.','Mrs.','Other'],inplace=True)
print(train_data.Title.value_counts())
print(test_data.Title.value_counts())

test_data.Title.replace(['Col.','Rev.','Ms.','Dona.','Dr.'],
                         ['Other','Other','Miss.','Mrs.','Other'],inplace=True)
print(test_data.Title.value_counts())

def age_fill(df):
    for i in range(len(df)):
        if(np.isnan(df.iloc[i,:]['Age'])==True):
            ttl=df.iloc[i,:]['Title']
            val=df[(df['Title']==ttl)].Age.median()
            df['Age'].iloc[i]=val
    return (df)

age_fill(train_data)
print(train_data.isna().sum())
age_fill(test_data)
print(test_data.isna().sum())

print(train_data.head())

train_data.drop(['Name','Fare_Gp','Title'],axis=1,inplace=True)
print(train_data.head())
test_data.drop(['Name','Fare_Gp','Title'],axis=1,inplace=True)
print(test_data.head())

y=train_data.Survived
train_data.drop(['Survived'],axis=1,inplace=True)
print(train_data.head())

data=pd.concat([train_data,test_data],axis=0)
print(data.shape)

data=pd.get_dummies(data,drop_first=True)
print(data.head())

train_data=data[data['PassengerId']<892]
print(train_data.head())
test_data=data[data['PassengerId']>892]
print(test_data.head())

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
train_scaled=scaler.fit_transform(train_data)
test_scaled=scaler.fit_transform(test_data)

print(train_scaled.shape,test_scaled.shape)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train_scaled,y,test_size=0.2)
print(x_train.shape,x_test.shape)

print(y_train)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

#逻辑回归
skf=StratifiedKFold(n_splits=5,shuffle=True)
logmod_scores=cross_val_score(LogisticRegression(solver='liblinear'),train_scaled,y,cv=skf)
print(logmod_scores.mean())

logmod=LogisticRegression(solver='liblinear')
logmod.fit(x_train,y_train)
ypred_logmod=logmod.predict(x_test)
cm_log=confusion_matrix(y_test,ypred_logmod)
sns.heatmap(cm_log,annot=True,cmap='Blues')
plt.xlabel('真实值')
plt.ylabel('预测值')

print(classification_report(y_test, ypred_logmod))

#SVM
svm_scores=cross_val_score(SVC(C=150,kernel='linear'),train_data,y,cv=3)
print(svm_scores.mean())

svmod=SVC(C=150,kernel='linear')
svmod.fit(x_train,y_train)
ypred_svmod=svmod.predict(x_test)
cm_svm=confusion_matrix(y_test,ypred_svmod)
sns.heatmap(cm_svm,annot=True,cmap='Blues')
plt.xlabel('真实值')
plt.ylabel('预测值')

print(classification_report(y_test, ypred_svmod))

#Decision Tree
dectre_score=cross_val_score(DecisionTreeClassifier(),train_scaled,y,cv=skf)
print(dectre_score.mean())

dectre_mod=DecisionTreeClassifier()
dectre_mod.fit(x_train,y_train)
ypred_dectre=dectre_mod.predict(x_test)
cm_dectre=confusion_matrix(y_test,ypred_dectre)
sns.heatmap(cm_dectre,annot=True,cmap='Blues')
plt.xlabel('真实值')
plt.ylabel('预测值')

print(classification_report(y_test, ypred_dectre))

#Random Forest
rf_scores=cross_val_score(RandomForestClassifier(criterion='gini'),train_scaled,y,cv=skf)
print(rf_scores.mean())

rfmod=RandomForestClassifier()
rfmod.fit(x_train,y_train)
ypred_rfmod=rfmod.predict(x_test)
cm_rf=confusion_matrix(y_test,ypred_rfmod)
sns.heatmap(cm_rf,annot=True,cmap='Blues')
plt.xlabel('真实值')
plt.ylabel('预测值')

print(classification_report(y_test, ypred_rfmod))

#Bernoulli NB
nbmod_scores=cross_val_score(BernoulliNB(),train_scaled,y,cv=skf)
print(nbmod_scores.mean())

nbmod=BernoulliNB()
nbmod.fit(x_train,y_train)
ypred_nb=nbmod.predict(x_test)
cm_nb=confusion_matrix(y_test,ypred_nb)
sns.heatmap(cm_nb,annot=True,cmap='Blues')
plt.xlabel('真实值')
plt.ylabel('预测值')

print(classification_report(y_test, ypred_nb))

#Submission
model=RandomForestClassifier(criterion='gini')
model.fit(train_scaled,y)
ypred_rf=model.predict(test_scaled)
path=pd.DataFrame({'PassengerId':data.PassengerId[891:],'Survived':ypred_rf})
print(path.to_csv(r'E:\titanic\Submission.csv',index=False))

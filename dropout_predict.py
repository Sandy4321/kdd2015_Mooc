'''
@lptMusketeers 2017.10.20
'''
from __future__ import unicode_literals
import codecs
import numpy as np
import pylab as pl
from itertools import *
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import cross_validation
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import RandomForestClassifier

class DropoutPredict(object):
    def loadData(self,filename):
        print("loadData...")
        df1 = pd.read_csv(filename)
        df1.drop(["enrollment_id","course_id"],axis=1)
        df2 = df1.drop("dropout",inplace=False,axis=1)
        x = df2.values #DataFrame的值组成的二维数组
        x = scale(x) #去均值后规范化
        y = np.ravel(df1["dropout"])
        return x,y
            
    def logistic_regression(self,x_train,y_train):
        print("logistic_regression...")
        clf1 = LogisticRegression()
        score1 = cross_validation.cross_val_score(clf1,x_train,y_train,cv=10,scoring="accuracy")
        x = [int(i) for i in range(1,11)]
        y = score1
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x,y,label='LogReg')
        pl.legend()
        pl.savefig("picture/LogReg.png")
        print (np.mean(score1))
        
    def svm(self,x_train,y_train):
        print("svm...")
        clf2 = svm.LinearSVC(random_state=2016)
        score2 = cross_validation.cross_val_score(clf2,x_train,y_train,cv=10,scoring='accuracy')
        #print score2
        print ('The accuracy of linearSVM:')
        print (np.mean(score2))
        x = [int(i) for i in range(1, 11)]
        y = score2
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='SVM')
        pl.legend()
        pl.savefig("picture/SVM.png")
        
    def naive_bayes(self,x_train,y_train):
        print("naive_bayes...")      
        clf3 = GaussianNB()
        score3 =  cross_validation.cross_val_score(clf3,x_train,y_train,cv=10,scoring='accuracy')
        print ("The accuracy of Naive Bayes:")
        print (np.mean(score3))
        x = [int(i) for i in range(1, 11)]
        y = score3
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='NB')
        pl.legend()
        pl.savefig("picture/NB.png")  
          
    def decision_tree(self,x_train,y_train):
        print("decision_tree...") 
        clf4 = tree.DecisionTreeClassifier()
        score4 = cross_validation.cross_val_score(clf4,x_train,y_train,cv=10,scoring="accuracy")
        print ('The accuracy of DT:')
        print (np.mean(score4))
        x = [int(i) for i in range(1, 11)]
        y = score4
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='DT')
        pl.legend()
        pl.savefig("picture/DT.png")
        
    def gradient_boosting(self,x_train,y_train):
        print("gradient_boosting...")     
        clf5 = GradientBoostingClassifier()
        score5 = cross_validation.cross_val_score(clf5,x_train,y_train,cv=10,scoring="accuracy")
        print ('The accuracy of GradientBoosting:')
        print (np.mean(score5))
        x = [int(i) for i in range(1, 11)]
        y = score5
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='GBDT')
        pl.legend()
        pl.savefig("picture/GBDT.png")
    def mlp(self,x_train,y_train):   
        print("mlp...") 
        clf = MLPClassifier(hidden_layer_sizes=(1000,),
                            activation='logistic', solver='sgd',
                            learning_rate_init = 0.001, max_iter=100000)
        score = cross_validation.cross_val_score(clf,x_train,y_train,cv=10,scoring="accuracy")
        print ('The accuracy of MLP:')
        print (np.mean(score))
        x = [int(i) for i in range(1, 11)]
        y = score
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='MLP')
        pl.legend()
        pl.savefig("picture/MLP.png")
        
    def random_forest(self,x_train,y_train): 
        print("random_forest...")        
        clf = RandomForestClassifier(n_estimators=100)   
        score = cross_validation.cross_val_score(clf,x_train,y_train,cv=10,scoring="accuracy")
        print ('The accuracy of RandomForest:')
        print (np.mean(score))
        x = [int(i) for i in range(1, 11)]
        y = score
        pl.ylabel(u'Accuracy')
        pl.xlabel(u'times')
        pl.plot(x, y,label='RandForest')
        pl.legend()
        pl.savefig("picture/RandomForest.png")
        
    def drop_predict(self):
        filename = 'feature/final_feature_all.csv'
        x_train,y_train = self.loadData(filename)
        '''
        self.logistic_regression(x_train,y_train)
        self.svm(x_train,y_train)
        self.naive_bayes(x_train,y_train)
        self.decision_tree(x_train,y_train)
        
        self.gradient_boosting(x_train,y_train)
        '''
        self.random_forest(x_train,y_train)
        self.mlp(x_train,y_train)

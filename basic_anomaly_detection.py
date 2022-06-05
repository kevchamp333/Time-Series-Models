# -*- coding: utf-8 -*-
"""
Created on Thu May 19 18:05:54 2022

@author: WooYoungHwang
"""
import sys
sys.path.append(r'C:\Users\WooYoungHwang\Desktop\SPS\연구\code\SVDD-Python-master\SVDD-Python-master')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.BaseSVDD import BaseSVDD
from scipy.stats import beta, f, chi2

# 정상 데이터 생성
mu = [0,1,2,3,-1]
cov = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]

normal_data = np.random.multivariate_normal(mu, cov,500)
print("정상 데이터 크기 : " + str(normal_data.shape))

# 이상 데이터 생성
mu = [0,1,2,3,-1]
cov = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]

anomaly_data = np.random.multivariate_normal(mu, cov, 10)

# 이상치는 정규분포로부터 생성한 데이터에 5를 더했으며, 변수 별로 이상치에 대한 영향력이 다르도록 더해준다. 
# 영향력은 첫번째 변수가 제일 약하고 다섯번째 변수가 제일 강하도록 설정했다.
anomaly_data += np.array([[0,0,0,0,0,0,0,0,0,0],
                          [5,5,0,0,0,0,0,0,0,0],
                          [5,5,5,5,0,0,0,0,0,0],
                          [5,5,5,5,5,5,0,0,0,0],
                          [5,5,5,5,5,5,5,5,5,5]]).T
print("비정상 데이터 크기 : " + str(anomaly_data.shape))

# 합치기
data = np.concatenate([normal_data, anomaly_data])
print("총 데이터 크기 : " + str(data.shape))
y_true = [1 for i in range(0, 500)] + [-1 for i in range(0, 10)]

# =============================================================================
# Local Outlier Factor
# =============================================================================
from sklearn.neighbors import LocalOutlierFactor

clf=LocalOutlierFactor(contamination=0.1)
y_pred=clf.fit_predict(data)

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=['class abnormal', 'class normal']))


# =============================================================================
# Isolation Forest
# =============================================================================
from sklearn.ensemble import IsolationForest

clf=IsolationForest(random_state = 0)
y_pred=clf.fit_predict(data)

print(classification_report(y_true, y_pred, target_names=['class abnormal', 'class normal']))


# =============================================================================
# One Class SVM
# =============================================================================
from sklearn.svm import OneClassSVM

clf = OneClassSVM(gamma='auto')
y_pred = clf.fit_predict(data)

print(classification_report(y_true, y_pred, target_names=['class abnormal', 'class normal']))


# =============================================================================
# SVDD
# =============================================================================
from src.BaseSVDD import BaseSVDD

svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='off')

y_pred = svdd.fit_predict(data)


print(classification_report(y_true, y_pred, target_names=['class abnormal', 'class normal']))


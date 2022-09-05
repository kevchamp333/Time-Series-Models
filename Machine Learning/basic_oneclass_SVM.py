# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:35:07 2022

@author: -
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Read Data
# =============================================================================
'''
Data: NASA Bearing Dataset
NASA Bearing Dataset은 NSF I/UCR Center의 Intelligent Maintenance System의 4개의 bearing에서 고장이 발생할 때까지 
10분 단위로 수집된 센서 데이터이다. 
본 데이터셋은 특정 구간에서 기록된 1-second vibration signal snapshots을 나타내는 여러 개의 파일로 구성되어 있다. 
각 파일은 20 kHz 단위로 샘플링 된 20,480개의 data point를 포함하고 있으며, 각 파일의 이름은 데이터가 수집된 시간을 의미한다. 
해당 데이터셋은 크게 3개의 데이터를 포함하고 있으며, 본 실습에서 사용하는 데이터는 bearing 1에서 outer race failure가 발생할 때까지 
수집된 센서 데이터이다.
'''

data = pd.read_csv(r'C:\Users\WooYoungHwang\Desktop\SPS\외부 활동\교육자료\LG전자 Data Analytics 교육 자료\[15회차] 시계열 이상치 탐지3\data/nasa_bearing_dataset.csv', index_col=0)
data.index = pd.to_datetime(data.index)
data.head()


# 전체 기간의 데이터 분포 확인
data.plot(figsize = (12, 6))
plt.axvline(data.index[int(len(data) * 0.5)], c='black')
plt.axvline(data.index[int(len(data) * 0.7)], c='black')


# 데이터 Split
X_train = data[data['data_type'] == 'train'].iloc[:, :4]
y_train = data[data['data_type'] == 'train'].iloc[:, -2].values

X_test = data[data['data_type'] == 'test'].iloc[:, :4]
y_test = data[data['data_type'] == 'test'].iloc[:, -2].values

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)


# 데이터 정규화
# train 데이터를 기반으로 train/test 데이터에 대하여 standard scaling 적용 (평균 0, 분산 1) 
scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), 
                              columns=X_train.columns, 
                              index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                             columns=X_test.columns, 
                             index=X_test.index)


# =============================================================================
# Utils
# =============================================================================
# Threshold
# score의 min ~ max 범위를 num_step개로 균등 분할한 threshold에 대하여 best threshold 탐색 
def search_best_threshold(score, y_true, num_step):
    best_f1 = 0.5
    best_threshold = None
    for threshold in np.linspace(min(score), max(score), num_step):
        y_pred = threshold < score

        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print('Best threshold: ', round(best_threshold, 4))
    print('Best F1 Score:', round(best_f1, 4))
    return best_threshold


# anomaly score plot 도출
def draw_plot(scores, threshold):
    normal_scores = scores[scores['anomaly'] == False]
    abnormal_scores = scores[scores['anomaly'] == True]

    plt.figure(figsize = (12,5))
    plt.scatter(normal_scores.index, normal_scores['score'], label='Normal', c='blue', s=3)
    plt.scatter(abnormal_scores.index, abnormal_scores['score'], label='Abnormal', c='red', s=3)
    
    plt.axhline(threshold, c='green', alpha=0.7)
    plt.axvline(data.index[int(len(data) * 0.5)], c='orange', ls='--')
    
    plt.xlabel('Date')
    plt.ylabel('Anomaly Score')
    plt.legend()
    
    plt.show()
    
    
# FRR, FAR, F1 score 도출
def calculate_metric(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    tp, fn, fp, tn = cm.ravel()
    
    frr = fp / (fp + tn)
    far = fn / (fn + tp) 
    
    f1 = f1_score(y_true, y_pred)
    
    return frr, far, f1


# =============================================================================
# One Class SVM
# =============================================================================
'''
- OC-SVM 설명
    - kernl: kernel 유형. (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed') 디폴트 'rbf'
    - gamma: 'rbf', 'poly' 및 'sigmoid'에 대한 커널 계수
    - nu: 훈련 오류 비율의 상한값과 지원 벡터 비율의 하한값, 디폴트 0.5
    - max_iter: Solver 내의 iteration의 hard limit, no limit -1 
'''

# OC-SVM 모델 적합
OCSVM_model = OneClassSVM(kernel = 'rbf', gamma = 0.0001, nu = 0.009, max_iter = -1)
OCSVM_model.fit(X_train_scaled)

# 적합된 모델을 기반으로 train/test 데이터의 anomaly score 도출
# train/test 데이터의 OCSVM score 도출
OCSVM_train = -1 * OCSVM_model.score_samples(X_train_scaled)
OCSVM_test = -1 * OCSVM_model.score_samples(X_test_scaled)

# train/test 데이터의 anomaly score 분포 시각화
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize = (8, 8))

sns.distplot(OCSVM_train, bins=100, kde=True, color='blue', ax=ax1)
sns.distplot(OCSVM_test, bins=100, kde=True, color='red', ax=ax2)
ax1.set_title("Train Data")
ax2.set_title("Test Data")

# best threshold 도출
OCSVM_best_threshold = search_best_threshold(OCSVM_test, y_test, num_step=1000)

# 최종 결과 도출
OCSVM_scores = pd.DataFrame(index=data.index)
OCSVM_scores['score'] = list(np.hstack([OCSVM_train, OCSVM_test]))
OCSVM_scores['anomaly'] = OCSVM_best_threshold < OCSVM_scores['score']
OCSVM_scores.head()

# 전체 데이터의 anomaly score 확인
draw_plot(OCSVM_scores, OCSVM_best_threshold)

# F1 Score: 0.9291
frr, far, f1 = calculate_metric(y_test, OCSVM_scores['anomaly'].iloc[int(len(data) * 0.5):])
print("**  FRR: {}  |  FAR: {}  |  F1 Score: {}".format(round(frr, 4), round(far, 4), round(f1, 4)))

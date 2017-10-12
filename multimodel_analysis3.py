
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from ggplot import *

start = time.time()

# include more time variables
#break this model into by brands


from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

churn = pd.read_csv('churn_data3.csv', header=0, sep="|", low_memory=False, skipinitialspace=True, index_col=None)
#==============================================================================
 
#CREATE VARIABLE COMBINATIONS
churn['pts_per_membership_age'] = churn['points_sum']/churn['membership_age']
churn['pts_per_accrualgroupref_count'] =  churn['points_sum']/churn['accrualgroupref_count']
churn['pts_per_coupon_use_value'] =  churn['points_sum']/churn['coupon_use_value']
churn['pts_per_day_tran_count'] =  churn['points_sum']/churn['day_tran_count']
churn['pts_per_avg_days_between_purchases'] =  churn['points_sum']/churn['avg_days_between_purchases']
churn['pts_per_days_since_last_purch'] =  churn['points_sum']/churn['days_since_last_purch']
churn['subtotal_per_membership_age'] =  churn['subtotal']/churn['membership_age']
churn['subtotal_per_accrualgroupref_count'] =  churn['subtotal']/churn['accrualgroupref_count']
churn['subtotal_per_coupon_use_value'] =  churn['subtotal']/churn['coupon_use_value']
churn['subtotal_per_day_tran_count'] =  churn['subtotal']/churn['day_tran_count']
churn['subtotal_per_avg_days_between_purchases'] =  churn['subtotal']/churn['avg_days_between_purchases']
churn['subtotal_per_days_since_last_purch'] =  churn['subtotal']/churn['days_since_last_purch']
churn['couponuse_per_membership_age'] =  churn['coupon_use_value']/churn['membership_age']
churn['couponuse_per_accrualgroupref_count'] =  churn['coupon_use_value']/churn['accrualgroupref_count']
churn['couponuse_per_points_sum'] =  churn['coupon_use_value']/churn['points_sum']
churn['couponuse_per_day_tran_count'] =  churn['coupon_use_value']/churn['day_tran_count']
churn['couponuse_per_avg_days_between_purchases'] =  churn['coupon_use_value']/churn['avg_days_between_purchases']
churn['couponuse_per_days_since_last_purch'] =  churn['coupon_use_value']/churn['days_since_last_purch']

 
#RATIO OF BRAND TRANSACTIONS TO TOTAL TRANSACTIONS
churn['ace05_1_per_accrualgroupref_count'] =  churn['ace05_1']/churn['accrualgroupref_count']
churn['ace05_2_per_accrualgroupref_count'] =  churn['ace05_2']/churn['accrualgroupref_count']
churn['ace05_3_per_accrualgroupref_count'] =  churn['ace05_3']/churn['accrualgroupref_count']
churn['ace05_4_per_accrualgroupref_count'] =  churn['ace05_4']/churn['accrualgroupref_count']
churn['ace05_5_per_accrualgroupref_count'] =  churn['ace05_5']/churn['accrualgroupref_count']
churn['ace05_6_per_accrualgroupref_count'] =  churn['ace05_6']/churn['accrualgroupref_count']
churn['ace05_7_per_accrualgroupref_count'] =  churn['ace05_7']/churn['accrualgroupref_count']
 
churn['use_per_available_coupon_count'] =  churn['coupon_use_count']/churn['coupon_available_count']
churn['type_per_available_coupon_count'] =  churn['coupon_type_count']/churn['coupon_available_count']
churn['use_per_type_coupon_count'] =  churn['coupon_use_value']/churn['coupon_type_count']
churn['couponusecount_per_accrualgroupref_count'] =  churn['coupon_use_count']/churn['accrualgroupref_count']

#CREATE COMBINATIONS OF ABOVE'S VARIABLE COMBINATIONS
churn['pts_per_avg_subtotal_age'] =  churn['points_sum']/churn['subtotal_per_membership_age']
churn['pts_per_avg_subtotal_accrual'] =  churn['points_sum']/churn['subtotal_per_accrualgroupref_count']
churn['pts_per_avg_subtotal_coupon_use'] =  churn['points_sum']/churn['subtotal_per_coupon_use_value']
churn['pts_per_avg_subtotal_day_tran_count'] =  churn['points_sum']/churn['subtotal_per_coupon_use_value']

#==============================================================================
churn = churn.replace([np.inf, -np.inf], np.nan)
churn = churn.drop('target_variable_desc', axis=1)
#churn = churn.drop('target_variable_desc', axis=1)
churn.head(5)
#pts_per_days_last_purch


print(churn.columns.get_loc('target_variable'))

print(len(churn.columns))
print(churn.columns)

churn.fillna(0, inplace=True)
X, y = shuffle(churn[churn.columns[7:]], churn[churn.columns[6]], random_state=13)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X, y, test_size=0.1)

n_estimator = len(churn.columns)-6                                                     
# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,random_state=0)

rt_lm = LogisticRegression()
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

grd = GradientBoostingClassifier(n_estimators=n_estimator
,learning_rate=0.2
,min_samples_split=500
,min_samples_leaf=50
,max_depth=9
,max_features='sqrt'
,random_state=20)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1] 
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

#for i, y_pred in enumerate(grd.staged_predict(X_test)):
#    test_score[i] = grd.loss_(y_test, y_pred)
    
# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

plt.figure(figsize=(20, 10))
ax = plt.axes()
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')                                              
ax.set_xticks((0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1))
ax.set_yticks((0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1))
plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title('Deviance')
#plt.plot(np.arange(n_estimator) + 1, grd.train_score_, 'b-', label='Training Set Deviance')
#plt.plot(np.arange(n_estimator) + 1, test_score, 'r-', label='Test Set Deviance')
#plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
#plt.ylabel('Deviance')

feature_importance = grd.feature_importances_

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(20, 40))
plt.axis([-0.1, 500, -0.1, .3])
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

feature_importance = grd.feature_importances_
plt.rcParams.update({'font.size': 22})
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)[::-1][:len(feature_importance)]
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(80, 20))
plt.axis([-0.1, 500, -0.1, .3])
plt.subplot(1, 2, 2)
plt.bar(pos, feature_importance[sorted_idx], align='center')
plt.xticks(pos, X.columns[sorted_idx], rotation='vertical')
plt.ylabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

print("--- %s seconds ---" % (time.time() - start))





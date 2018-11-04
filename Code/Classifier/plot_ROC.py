import itertools
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd
import matplotlib as mpl
import scipy.io as sio

lw=1
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'


ytest= pd.read_csv('Matine1/ytest_sum_NB.csv', index_col=0) 
ytest_NB=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('Matine1/yscore_sum_NB.csv', index_col=0)
yscore_NB=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('Matine1/Matine_NB_result.csv',index_col=0)
auc_NB=np.array(auc_,dtype=np.float)
auc_score_NB=auc_NB[5,7]
fpr, tpr, _ = roc_curve(ytest_NB[:,0], yscore_NB[:,0])
plt.plot(fpr, tpr, color='brown',
lw=lw, label='NB (AUC = %.4f)' % auc_score_NB)


ytest= pd.read_csv('Matine1/ytest_sum_KNN.csv', index_col=0) 
ytest_KNN=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('Matine1/yscore_sum_KNN.csv', index_col=0)
yscore_KNN=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('Matine1/Matine_KNN_result.csv',index_col=0)
auc_KNN=np.array(auc_,dtype=np.float)
auc_score_KNN=auc_KNN[5,7]
fpr, tpr, _ = roc_curve(ytest_KNN[:,0], yscore_KNN[:,0])
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='KNN (AUC = %.4f)' % auc_score_KNN)

data_train = sio.loadmat('set1_roc.mat')
x_1=data_train.get('x_1')#取出字典里的data
y_1=data_train.get('y_1')
x_=x_1.T
y_=y_1.T
plt.plot(x_, y_, color='forestgreen',
lw=lw, label='LibD3C(AUC = %.4f)' % 1)

ytest= pd.read_csv('Matine1/ytest_sum_AdaBoost.csv', index_col=0) 
ytest_AdaBoost=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('Matine1/yscore_sum_AdaBoost.csv', index_col=0)
yscore_AdaBoost=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('Matine1/Matine_AdaBoost_result.csv',index_col=0)
auc_AdaBoost=np.array(auc_,dtype=np.float)
auc_score_AdaBoost=auc_AdaBoost[5,7]
fpr, tpr, _ = roc_curve(ytest_AdaBoost[:,0], yscore_AdaBoost[:,0]) 
plt.plot(fpr, tpr, color='deeppink',
lw=lw, label='AdaBoost(AUC = %.4f)' % auc_score_AdaBoost)


ytest= pd.read_csv('Matine1/ytest_sum_RF.csv', index_col=0) 
ytest_RF=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('Matine1/yscore_sum_RF.csv', index_col=0)
yscore_RF=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('Matine1/Matine_RF_result.csv',index_col=0)
auc_RF=np.array(auc_,dtype=np.float)
auc_score_RF=auc_RF[5,7]
fpr, tpr, _ = roc_curve(ytest_RF[:,0], yscore_RF[:,0])
plt.plot(fpr, tpr, color='blue',         
lw=lw, label='RF (AUC = %.4f)' % auc_score_RF)

ytest= pd.read_csv('Matine1/ytest_sum_SVC.csv', index_col=0) 
ytest_SVC=np.array(ytest,dtype=np.float)
yscore= pd.read_csv('Matine1/yscore_sum_SVC.csv', index_col=0)
yscore_SVC=np.array(yscore,dtype=np.float)
auc_=pd.read_csv('Matine1/Matine_SVC_result.csv',index_col=0)
auc_SVC=np.array(auc_,dtype=np.float)
auc_score_SVC=auc_SVC[5,7]
fpr, tpr, _ = roc_curve(ytest_SVC[:,0], yscore_SVC[:,0]) 
plt.plot(fpr, tpr, color='red',
lw=lw, label='SVM (AUC = %.4f)' % auc_score_SVC)


plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([-0.03, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")
ax = plt.gca()
ax.spines['left'].set_linewidth(0.8)
ax.spines['right'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
ax.spines['top'].set_linewidth(0.8)

plt.savefig('figure_yeast_dimension.svg',dpi=2000,format='svg')
plt.show()




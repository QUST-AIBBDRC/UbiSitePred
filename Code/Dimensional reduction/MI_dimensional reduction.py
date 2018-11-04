import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import scale,StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from dimensional_reduction import mutual_mutual

data_train = sio.loadmat('set1_yu.mat')
data=data_train.get('AAA')
row=data.shape[0]
column=data.shape[1]
shu=data[:,np.array(range(1,column))]
shu=scale(shu)
label=data[:,0]	
data_2=mutual_mutual(shu,label,k=239)
shu=data_2
X=shu
y=label
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
sepscores = []
skf= StratifiedKFold(n_splits=5)
cv_clf =SVC(C=1.0, gamma='auto', kernel='rbf',probability=True)# "kernel":["rbf","linear","poly","sigmoid"]
for train, test in skf.split(X,y): 
    y_train=utils.to_categorical(y[train])
    hist=cv_clf.fit(X[train], y[train]) 
    y_score=cv_clf.predict_proba(X[test]) 
    y_test=utils.to_categorical(y[test])    
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color=plt.cm.Set1(1 / 7.),
    label='ROC %s (area = %0.2f%%)' % ('SVC', (roc_auc*100)))
    y_class= utils.categorical_probas_to_classes(y_score)
    y_test_tmp=y[test]
    acc,precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('SVC:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))

result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)#½«H1µÄ½á¹ûÌí¼Óµ½sepscores
result=sepscores
colum = ['Acc', 'precision', 'npv', 'Sn', 'Sp','MCC','F1','AUC']
ro=['1', '2', '3','4','5','6']
data_csv = pd.DataFrame(columns=colum, data=result,index=ro)
data_csv.to_csv(r'F:\项目1 ubiquitination泛素化\开源代码\code\svm1_mutual_yu.csv')
sio.savemat(r'F:\项目1 ubiquitination泛素化\开源代码\code\mutual1.mat', {'mutual_mutual':data_2}) 

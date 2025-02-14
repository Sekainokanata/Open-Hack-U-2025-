import numpy as np


import sound_prechange as sp #特徴ベクトル行列
import scikit_learn as sl#SVMモデル

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn import svm

def svm_exe(sp,sl):
    cls = -1
    
    cls = sl.predict(sp)
    
    cls = np.argmax(cls)
    
    print(f'Predicted class: {cls}')
    
    return cls
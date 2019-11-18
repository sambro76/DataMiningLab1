import numpy as np
from sklearn.model_selection import cross_val_score

ATT=np.genfromtxt('ATNTFaceImages400.txt', delimiter=',')
split_cols=1
rows = len(ATT) 
cols=ATT.shape[1]

train=np.zeros((rows, (cols - split_cols*40)))
test=np.zeros((rows, split_cols*40))
#Splitting the dataset into Train & Test Data

  
#deleted

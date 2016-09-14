import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

dirPath = '/Users/Tofigh/Documents/Kaggle/train/features'
X = joblib.load(os.path.join(dirPath,'subj10_series1_features_ch0'))

N=X.shape[0]
covariance = np.sum([np.dot(np.transpose(I),I / N) for I in X],axis = 0)
d,V = np.linalg.eigh(covariance)
D = np.diag(1. / np.sqrt(d+1E-15))
W = np.dot(V,D)
joblib.dump(W,os.path.join(dirPath,'W_white'))
W_inv = np.dot(np.diag(np.sqrt(d+1E-15)), np.transpose(V))

# plt.pcolormesh(X[10000,:,:])
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# plt.plot()
#
#
# plt.pcolormesh(np.dot(X[10000,:,:],W))
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# plt.plot()
#
#
#
# plt.pcolormesh(np.dot(np.dot(X[10000,:,:],W),W_inv))
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# plt.plot()
print ('here')
X_white = np.array([np.dot(I,W) for I in X])
joblib.dump(X_white,os.path.join(dirPath,'subj10_series1_features_ch0_white'))
joblib.dump(W_inv,os.path.join(dirPath,'W_inv'))
print (np.min(X_white))
print(X_white.shape)

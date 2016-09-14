import os,sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import joblib
dirPath = '/Users/Tofigh/Documents/Kaggle/DCGAN-tensorflow/samples_eeg'
image = np.asarray(Image.open(os.path.join(dirPath,'train_00_0099.png')))

print image
print image.shape
i=3
j=4
plt.pcolormesh(image[0+i*28:27+i*28,0+j*28:27+j*28,0])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.plot()

data_dir = '/Users/Tofigh/Documents/Kaggle/train/features'
fd = open(os.path.join(data_dir, 'subj10_series1_features_ch0'))
trX = joblib.load(fd).astype(np.float)

maxNorm = np.max(trX)
trX = trX / maxNorm
plt.pcolormesh(trX[0, :, :])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.plot()


plt.pcolormesh(trX[10000,:,:])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.plot()

plt.pcolormesh(trX[20000,:,:])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.plot()

print trX[20000,:,:]
plt.pcolormesh(trX[30000,:,:])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.plot()

plt.pcolormesh(trX[40000,:,:])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.plot()

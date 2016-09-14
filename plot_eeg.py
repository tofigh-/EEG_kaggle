import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

dirPath = "/Users/Tofigh/Documents/Kaggle/train"

dirFiles = filter(lambda x: x.endswith("_data.csv"), os.listdir(dirPath))

with open(os.path.join(dirPath,dirFiles[0])) as input:
    reader = iter(csv.reader(input))
    header = next(reader)
    data = np.array([row[1:] for row in reader],dtype=int)
    [N, num_channels] = data.shape
    ch = 0
    sampling_space = 1.0/500
    t = np.arange(0.0, N/500.0, sampling_space)

    df = np.linspace(0.0, 1.0 / (2.0 * sampling_space), N / 2)

    win_length = 110 # in ms
    overlap = 100 # in ms
    win_length_sample = win_length * 500 / 1000
    overlap_sample = overlap * 500 / 1000
    num_windows = 28
    s_time = time.time()
    [f, t, Sxx] = spectrogram(data[40000:40000+num_windows*(win_length_sample-overlap_sample) + overlap_sample, ch]-np.mean(data[:,ch])
                            ,window = 'hamming'
                            ,nperseg = win_length_sample
                            ,noverlap = overlap_sample
                            ,nfft = win_length_sample
                            ,fs=500
                            ,scaling ='spectrum')
    e_time = time.time()
    print(s_time- e_time)
    print(f.shape,t.shape,Sxx.shape)
    plt.pcolormesh(t,f,(Sxx-np.mean(Sxx))/168345.0)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    plt.plot()
    # plt.xlabel('freq HZ')
    # plt.ylabel('EEG Channel %d' % ch)
    # plt.title('EEG plot')
    #image1.show()
    #image2.show()

from PIL import Image
dirPath = '/Users/Tofigh/Documents/Kaggle/DCGAN-tensorflow/samples_eeg'
image = np.asarray(Image.open(os.path.join(dirPath,'train_54_0005.png')))

print image
print image.shape
plt.pcolormesh(t, f, image[0:27, 0:27, 0])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.plot()

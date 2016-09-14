import csv
import os
import numpy as np
import cPickle as pk
from scipy.signal import spectrogram
import joblib
import time

dirPath = "/Users/Tofigh/Documents/Kaggle/train"

dirFiles = filter(lambda x: x.endswith("_data.csv") and x.startswith("subj10_"), os.listdir(dirPath))
use_channels = 12
W_white = joblib.load(os.path.join(dirPath, 'features', 'W_white'))
max_val = 20.0
for inFile in dirFiles:
    with open(os.path.join(dirPath,inFile)) as input:

        reader_label = iter(csv.reader(open(os.path.join(dirPath, inFile.replace("_data.csv", "_events.csv")))))
        header_label = next(reader_label)
        labels = np.array([row[1:] for row in reader_label], dtype=int)
        reader = iter(csv.reader(input))
        header = next(reader)
        data = np.array([row[1:] for row in reader],dtype=int)
        data = data - np.mean(data, axis=0)
        assert (data.shape[0] == labels.shape[0])

        [N, num_channels] = data.shape

        sampling_space = 1.0 / 500
        win_length = 110 # in ms
        overlap = 100 # in ms
        win_length_sample = win_length * 500 / 1000
        overlap_sample = overlap * 500 / 1000
        non_overlap_sample = win_length_sample - overlap_sample
        concatenated_win = 28

        number_of_frames = (N - overlap_sample) / non_overlap_sample
        end_left_out_samples = N - (number_of_frames * non_overlap_sample + overlap_sample)
        start_constant = (concatenated_win / 2) * non_overlap_sample
        data_updated = np.concatenate(
            (
                np.zeros((start_constant, num_channels)),
                data,
                np.zeros(((concatenated_win / 2) * non_overlap_sample - end_left_out_samples, num_channels))
            )
            , axis=0)

        labels_updated = np.concatenate(
            (
                np.zeros((start_constant, 6)),
                labels,
                np.zeros(((concatenated_win / 2) * non_overlap_sample - end_left_out_samples, 6))
            )
            , axis=0)

        new_labels = []
        new_data = []
        for i in xrange(number_of_frames):
            multi_labels = np.where(
                np.mean(labels_updated[(i+concatenated_win/2) * non_overlap_sample + win_length_sample/2 - non_overlap_sample: (i+concatenated_win/2) * non_overlap_sample + win_length_sample/2 + non_overlap_sample,:],axis=0)
                >= 0.5
                )[0]
            multi_channel_feature = np.empty(shape=(win_length_sample / 2 + 1, concatenated_win, num_channels))
            for ch in xrange(use_channels):
                windowed_data = data_updated[i*non_overlap_sample: (i+concatenated_win) * non_overlap_sample + overlap_sample, ch]
                [f, t, features] = spectrogram(
                    windowed_data
                    , window='hamming'
                    , nperseg=win_length_sample
                    , noverlap=overlap_sample
                    , nfft=win_length_sample
                    , fs=500
                    , scaling='spectrum')

                multi_channel_feature[:,:, ch] = np.dot(features, W_white) / max_val
            if not multi_labels.size:
                new_labels.append(0)
                new_data.append(multi_channel_feature)
            else:
                for label in multi_labels:
                    new_labels.append(label+1) # plus 1 because label 0 is reserved for the case where neither of the stages are active
                    new_data.append(multi_channel_feature)

        new_labels = np.array(new_labels)
        new_data = np.array(new_data)
        N_samples = new_data.shape[0]
        ind_zeros = np.where(new_labels == 0)[0]
        num_selected_zeros = (N_samples - len(ind_zeros))/6
        seed_select = 0
        np.random.seed(seed_select)
        selected_zeros = np.random.choice(ind_zeros, num_selected_zeros, replace=False)
        print "number of selected zeros %d" %num_selected_zeros

        ind_nonzeros = np.where(new_labels > 0)[0]

        new_labels = np.concatenate((new_labels[selected_zeros], new_labels[ind_nonzeros]))
        new_data = np.concatenate((new_data[selected_zeros, :, :, :], new_data[ind_nonzeros, :, :, :]), axis=0)

        s_t = time.time()
        joblib.dump(new_data,open(os.path.join(dirPath,'features_s3', inFile.replace("_data.csv","_features_white")),'w'))
        pk.dump(new_labels, open(os.path.join(dirPath,'features_s3', inFile.replace("_data.csv", "_labels")), 'w'))
        s_end = time.time()
        print "file %s takes %4f to dump" %(inFile,(s_end-s_t))


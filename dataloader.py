import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing


#padding
def zeroPadding(input, length):
    inputdata = np.zeros([len(input), length])
    for i in range(len(input)):
        k = min(length, input[i].size)
        for j in range(k):
            inputdata[i, j] = input[i][j]
    return inputdata

# z-score
def normSample(data):
    output = []
    for i in range(len(data)):
        output.append(preprocessing.scale(data[i]))
    return output

class ECGDataset(Dataset): 

    def __init__(self, data_file, label_file,sampling_rate,max_length):
        data_list = np.load(data_file, allow_pickle=True)

        if sampling_rate != 300:
            for i in range(len(data_list)):
                data_list[i] = np.interp(np.arange(0, len(data_list[i]), 300 / sampling_rate), np.arange(0, len(data_list[i])),data_list[i])
        data_list = normSample(data_list)
        max_length = int(sampling_rate * max_length)
        data_list = zeroPadding(data_list, max_length)
        self.data_tensor = torch.tensor(data_list, dtype=torch.float32).unsqueeze(1)

        label_list = np.load(label_file, allow_pickle=True)
        self.label_tensor = torch.tensor(label_list, dtype=torch.long) 
    
    def __len__(self):
        return len(self.label_tensor)
    def __getitem__(self, idx):
        return self.data_tensor[idx], self.label_tensor[idx]




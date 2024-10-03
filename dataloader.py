import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing
import os
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
from scipy import io
from pathlib import Path
import wfdb
from wfdb.processing import resample_sig
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
    def __init__(self, data_file, label_file,sampling_rate,max_length,transform = None):
        data_list = np.load(data_file, allow_pickle=True)
        label_list = np.load(label_file, allow_pickle=True)

        # data_list = [sample for sample, label in zip(data_list, label_list) if label not in [2, 3]]
        # label_list = [label for label in label_list if label not in [2, 3]]
        if sampling_rate != 300:
            for i in range(len(data_list)):
                data_list[i] = np.interp(np.arange(0, len(data_list[i]), 300 / sampling_rate), np.arange(0, len(data_list[i])),data_list[i])
        data_list = normSample(data_list)
        max_length = int(sampling_rate * max_length)
        data_list = zeroPadding(data_list, max_length)
        self.data_tensor = torch.tensor(data_list, dtype=torch.float32).unsqueeze(1)
        self.label_tensor = torch.tensor(label_list, dtype=torch.long) 
        self.transform = transform
    
    def __len__(self):
        return len(self.label_tensor)
    def __getitem__(self, idx):        
        data = self.data_tensor[idx]
        if self.transform:
            data = self.transform(data)
        return data, self.label_tensor[idx]

class ECGsubset(Dataset): 
    def __init__(self, dataset, index,transform = None):
        self.dataset = dataset
        self.index = index  
        self.transform = transform 
    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx): 
        data = self.dataset.data_tensor[self.index[idx]]
        if self.transform:
            data = self.transform(data)
        return data, self.dataset.label_tensor[self.index[idx]]

class ECGDataset_AF_NORM(Dataset): 
    def __init__(self, data_file, label_file,sampling_rate,max_length):
        data_list = np.load(data_file, allow_pickle=True)
        label_list = np.load(label_file, allow_pickle=True)

        data_list = [sample for sample, label in zip(data_list, label_list) if label not in [2, 3]]
        label_list = [label for label in label_list if label not in [2, 3]]
        if sampling_rate != 300:
            for i in range(len(data_list)):
                data_list[i] = np.interp(np.arange(0, len(data_list[i]), 300 / sampling_rate), np.arange(0, len(data_list[i])),data_list[i])
        data_list = normSample(data_list)
        max_length = int(sampling_rate * max_length)
        data_list = zeroPadding(data_list, max_length)
        self.data_tensor = torch.tensor(data_list, dtype=torch.float32).unsqueeze(1)       
        self.label_tensor = torch.tensor(label_list, dtype=torch.long) 
    
    def __len__(self):
        return len(self.label_tensor)
    def __getitem__(self, idx):
        return self.data_tensor[idx], self.label_tensor[idx]


class ChapmanDataset(Dataset):
    def __init__(self, data_dir, label_excel_path,sampling_rate = 150,max_length = 10,lead = 1,transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Define your mapping rule here
        self.label_mapping = {
            'AFIB': 0,#AFIB
            'AF': 0,
            'SVT': 1,#GSVT
            'AT': 1,
            'SAAWR': 1,
            'ST': 1,
            'AVNRT': 1,
            'AVRT': 1,
            'SB': 2,#SB
            'SR': 3,#SR
            'SA': 3 #SI
        }

        # Load labels from Excel file
        self.labels_df = pd.read_excel(label_excel_path)  # Assuming the labels are in an Excel sheet

        # Create a mapping between filenames (without extension) and labels
        self.filename_to_label = {}
        for index, row in self.labels_df.iterrows():
            filename = os.path.splitext(row.iloc[0])[0]  # Extract filename without extension
            original_label = row.iloc[1]  # Second column contains original labels
            mapped_label = self.label_mapping.get(original_label, None)
            self.filename_to_label[filename] = mapped_label

        # List of CSV file paths
        self.csv_file_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if
                               filename.endswith('.csv')]

        # Create a list to store the labels for each CSV file
        self.labels = [self.filename_to_label.get(os.path.splitext(os.path.basename(path))[0], None)
                       for path in self.csv_file_paths]

        self.class_counts = Counter(self.labels)

        # Load labels from Excel file
        self.labels_df = pd.read_excel(label_excel_path)  # Assuming the labels are in an Excel sheet

        # Create a mapping between filenames (without extension) and labels
        self.filename_to_label = {}
        for index, row in self.labels_df.iterrows():
            filename = os.path.splitext(row.iloc[0])[0]  # Extract filename without extension
            original_label = row.iloc[1]  # Second column contains original labels
            mapped_label = self.label_mapping.get(original_label, None)
            self.filename_to_label[filename] = mapped_label

        # Create a list to store the data and labels
            
        
        self.data = []
        self.labels = []

        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
            
                csv_file_path = os.path.join(data_dir, filename)
                # Load CSV data
                csv_data = pd.read_csv(csv_file_path, header=None)
   
                # Get label for the current CSV file
                label = self.filename_to_label.get(os.path.splitext(filename)[0], None)

                #
                if label == 0 or label == 3:
                    
                    data = csv_data.iloc[:, lead].values  
                    if sampling_rate != 500:
                        data = np.interp(np.arange(0, len(data), 500 / sampling_rate), np.arange(0, len(data)),data)
                    data = preprocessing.scale(data)  
                    # Convert data to float32 Tensor
                    #data_tensor = torch.tensor(data, dtype=torch.float32)

                    # Append data and label to the lists
                    self.data.append(data)
                    if label == 0:
                        label = 1
                    else:
                        label = 0
                    self.labels.append(label)
        #self.data = pad_data(self.data,(sampling_rate*max_length,12))
        max_length = int(sampling_rate * max_length)
        self.data = zeroPadding(self.data, max_length)
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1)#.permute(0,2,1)
        self.data = torch.nan_to_num(self.data, nan=0.0)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = self.data[idx]
        label = self.labels[idx]
        # Convert label to long Tensor
        label_tensor = torch.tensor(label, dtype=torch.long) if label is not None else None
        return data_tensor,label_tensor

    def get_class_counts(self):
        return self.class_counts


    
class CPSCDataset(Dataset):
    def __init__(self, data_dir, sampling_rate = 150, max_length=15,lead = 1, transform=None):
        self.transform = transform
        self.file_dir = []
        filebase = data_dir + "Training_WFDB"
        inputdata = []

        output_dir = data_dir + "REFERENCE.csv"
        output = pd.read_csv(output_dir)
        self.refference = output
        output = np.array(output['First_label'], 'int') - 1

        for i in range(1, 6878):
            if i < 10:
                file = filebase + '/A000' + str(i) + '.mat'
            elif i < 100:
                file = filebase + '/A00' + str(i) + '.mat'
            elif i < 1000:
                file = filebase + '/A0' + str(i) + '.mat' 
            else:
                file = filebase + '/A' + str(i) + '.mat'
            lead_data = io.loadmat(file)['val'][lead,:]
            if sampling_rate != 500:
                lead_data = np.interp(np.arange(0, len(lead_data), 500 / sampling_rate), np.arange(0, len(lead_data)),lead_data)
                lead_data = preprocessing.scale(lead_data)  
            inputdata.append(lead_data)
             
        #self.output = output
        #print(a,b)
        
        self.data = inputdata
        self.data = [self.data[i] for i in range(len(output)) if output[i] < 2]
        max_length = int(sampling_rate * max_length)
        self.data = zeroPadding(self.data, max_length)

        
        output = output[output < 2]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data = torch.nan_to_num(self.data, nan=0.0).unsqueeze(1)
        self.label_tensor = torch.tensor(output, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label_tensor[idx]    
        return data, label


class MIMICSingleLeadContrastDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_fs=150):
        self.transform = transform
        self.target_fs = target_fs
        self.file_dir = []
        for root, dirs, files in os.walk(data_dir):
            if not dirs:
                path_obj = Path(os.path.abspath(root))
                last_layer = path_obj.parts[-1][1:]
                self.file_dir.append(str(path_obj) + '/' + last_layer)

    def __len__(self):
        return len(self.file_dir)

    def __getitem__(self, idx):
        rec_path = self.file_dir[idx]
        
        record = wfdb.rdrecord(rec_path)
        
        signal_data = record.p_signal
        signal_np = np.array(signal_data).astype(np.float32).transpose()
        signal_np = np.nan_to_num(signal_np, nan=0.0)
        
        
        lead_indices = np.random.choice(signal_np.shape[0], 2, replace=False)
        #lead_indices = np.random.choice(2, 2, replace=False)
        #
        lead_signal_1 = signal_np[lead_indices[0]]
        lead_signal_2 = signal_np[lead_indices[1]]
        
        
        original_fs = record.fs
        if self.target_fs and self.target_fs != original_fs:
            resampled_signal_1, _ = resample_sig(lead_signal_1, original_fs, self.target_fs)
            resampled_signal_2, _ = resample_sig(lead_signal_2, original_fs, self.target_fs)
        else:
            resampled_signal_1 = lead_signal_1
            resampled_signal_2 = lead_signal_2
        
        resampled_signal_1 = resampled_signal_1.reshape(1, -1)
        resampled_signal_2 = resampled_signal_2.reshape(1, -1)

        if self.transform:
            signal_1 = self.transform(resampled_signal_1)
            signal_2 = self.transform(resampled_signal_2)
        else:
            signal_1 = resampled_signal_1
            signal_2 = resampled_signal_2
        
        
        return signal_1, signal_2
    




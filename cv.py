import torch
import numpy as np
from dataloader import ECGDataset,ECGsubset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import configparser
from model import CTRhythm
import datetime
import os 
import torchvision.transforms as transforms
import csv
from utils import train_model, evaluate_model
def getNowTime():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

def try_gpu(num=0):
    if torch.cuda.device_count() >= num + 1:
        return torch.device(f'cuda:{num}')
    return torch.device('cpu')

if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)

    print("start!","|",getNowTime())

    # load config
    config = configparser.ConfigParser()
    filedir = os.path.dirname(__file__)
    configdir = os.path.join(filedir, "../config.ini")

    #dataset parameters
    config.read(configdir)
    data_file = config['Dataset']['data_file']
    label_file = config['Dataset']['label_file']
    sampling_rate = float(config['Dataset']['sampling_rate'])
    max_length = float(config['Dataset']['max_length'])

    #Training parameters
    learning_rate = float(config['Training_Parameters']['learning_rate'])
    num_epochs = int(config['Training_Parameters']['num_epochs'])
    batch_size = int(config['Training_Parameters']['batch_size'])
    device = try_gpu(num = int(config['device']['gpu']))
    criterion = torch.nn.CrossEntropyLoss()

    #Network parameters
    d_model = int(config['Model_Parameters']['d_model'])
    nhead = int(config['Model_Parameters']['nhead'])
    num_layers = int(config['Model_Parameters']['num_layers'])
    dim_feedforward = int(config['Model_Parameters']['dim_feedforward'])
    num_classes = int(config['Model_Parameters']['num_classes'])
    dropout = float(config['Model_Parameters']['dropout'])

    #load dataset
    data_list = np.load(data_file, allow_pickle=True)
    label_list = np.load(label_file, allow_pickle=True)

    transform = transforms.RandomApply([transforms.Lambda(lambda x: x * -1)], p=0.5)
    dataset = ECGDataset(data_file, label_file, sampling_rate, max_length)
    print(len(dataset))
    print("Data load successfully","|",getNowTime())
    
    # cv fold
    cv_train_index = []
    cv_test_index = []
    train_index_file = 'cv_index/cv_train_index.csv'
    text_index_file = 'cv_index/cv_test_index.csv'
    with open(train_index_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            int_row = [int(item) for item in row]
            cv_train_index.append(int_row)
    with open(text_index_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            int_row = [int(item) for item in row]
            cv_test_index.append(int_row)
    print(len(cv_train_index))
    num_folds = int(config['cross_validation']['num_folds'])

    del data_list, label_list
     
    result_dir = filedir + '/result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # cv 
    method_list = [CTRhythm]
    all_results = []
    for num, method in enumerate(method_list):
        method_result_dir = result_dir + '/' + str(num+1)
        if not os.path.exists(method_result_dir):
            os.makedirs(method_result_dir)
        print("method ",num+1,"|",getNowTime())
        
        fold_metrics = [] 
        for fold in range(num_folds):
            train_indices = cv_train_index[fold]
            test_indices = cv_test_index[fold]
            print( "Fold", fold + 1,"|",getNowTime())

            train_dataset = ECGsubset(dataset, train_indices,transform=  transform)
            test_dataset = ECGsubset(dataset, test_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model = method(num_classes = 4)
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.01)
            #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, device=device)
            model_dir = method_result_dir + '/model_' + str(fold+1) + '.pth'
            torch.save(model.state_dict(), model_dir)
            conf_matrix, f1_scores, f1_nao, accuracy, test_loss = evaluate_model(model, test_loader, criterion, device)
            print(conf_matrix)

            fold_metrics.append({
                'conf_matrix': conf_matrix,
                'f1_scores': f1_scores,
                'f1_nao': f1_nao,
                'accuracy': accuracy,
                'test_loss': test_loss,
            })
        num = 0
        for fold in fold_metrics:
            num = num + 1
            print(f"Fold {num} - Accuracy: {fold['accuracy']:.3f} - F1 Nao: {fold['f1_nao']:.3f} - F1 Scores: {', '.join([f'{f1_score:.3f}' for f1_score in fold['f1_scores']])}")
        
        #output results with "mean ± std" 
        all_f1_scores = np.array([fold['f1_scores'] for fold in fold_metrics])
        f1_scores_mean = np.mean(all_f1_scores, axis=0)
        f1_scores_std = np.std(all_f1_scores, axis=0)
        f1_nao_mean = np.mean(f1_scores_mean[:-1])
        f1_nao_std = np.std(f1_scores_mean[:-1])
        accuracy_mean = np.mean([fold['accuracy'] for fold in fold_metrics])
        accuracy_std = np.std([fold['accuracy'] for fold in fold_metrics])
        test_loss_mean = np.mean([fold['test_loss'] for fold in fold_metrics])
        test_loss_std = np.std([fold['test_loss'] for fold in fold_metrics])
        #print(fold_metrics)
        method_result = []
        for i in range(4):
            method_result.append(f'{f1_scores_mean[i]:.3f}±{f1_scores_std[i]:.3f}')
            #print(f'{f1_scores_mean[i]:.3f}±{f1_scores_std[i]:.3f}')
        print( f"F1 Scores Mean: {', '.join([f'{f1_score:.3f}' for f1_score in f1_scores_mean])} ± {', '.join([f'{f1_score:.3f}' for f1_score in f1_scores_std])}")
        print(f"F1 Nao Mean: {f1_nao_mean:.3f} ± {f1_nao_std:.3f}")
        method_result.append(f'{f1_nao_mean:.3f}±{f1_nao_std:.3f}')
        print(f"Accuracy Mean: {accuracy_mean:.3f} ± {accuracy_std:.3f}")
        method_result.append(f'{accuracy_mean:.3f}±{accuracy_mean:.3f}')
        # print(f"Test Loss Mean: {test_loss_mean:.3f} ± {test_loss_std:.3f}")
        all_results.append(method_result)
        print("method_end","|",getNowTime())
    
    all_results_path = result_dir + '/' + 'results.csv'
    with open(all_results_path, mode='w', newline='',encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(all_results)









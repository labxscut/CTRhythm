import torch
import numpy as np
from dataloader import ECGDataset, ChapmanDataset,CPSCDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import configparser
from model import CTRhythm
from utils import train_model, evaluate_model,train_model_all,evaluate_model_all
import datetime
import os 
from sklearn import preprocessing
import csv

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

    data_dir = '/work1/lzy/data/Chapman/ECGDataDenoised'
    label_excel_path = '/work1/lzy/data/Chapman/Diagnostics.xlsx'
    dataset = ChapmanDataset(data_dir, label_excel_path ,sampling_rate = 150,max_length = 10,lead = 1)
    test_loader =  DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("Data load successfully","|",getNowTime())

    all_results = []
    method_list = [CTRhythm]
    for num, method in enumerate(method_list):
        print("method ",num+1,"|",getNowTime())

        fold_metrics = []  
        for fold in range(5):
            print("fold:",fold + 1)
            
            
            # cv 
            model = method(num_classes = 4)
            model_path = filedir + '/result/' + str(num + 1) + '/model_' + str(fold+1) + '.pth'
            model.load_state_dict(torch.load(model_path))#加载参数
            #model = CNN(num_classes = num_classes, dropout=0.3)
            model.to(device)

            conf_matrix, f1_scores, f1_all, accuracy, test_loss = evaluate_model_all(model, test_loader, criterion, device)
            #print(conf_matrix)

            fold_metrics.append({
                'conf_matrix': conf_matrix,
                'f1_scores': f1_scores,
                'f1_all': f1_all,
                'accuracy': accuracy,
                'test_loss': test_loss,
            })
            print()
            print(f"Accuracy: {fold_metrics[fold]['accuracy']:.3f} - F1 All: {np.mean(fold_metrics[fold]['f1_scores'][0:2]):.3f} - F1 Scores: {', '.join([f'{fold_metrics[fold]['f1_scores'][i]:.3f}' for i in range(2) ])}")

        #output results with "mean ± std" 
        all_f1_scores = np.array([fold['f1_scores'][0:2] for fold in fold_metrics])

        f1_scores_mean = np.mean(all_f1_scores, axis=0)
        f1_scores_std = np.std(all_f1_scores, axis=0)
        #f1_all_mean = np.mean(f1_scores_mean[:-1])
        #f1_all_std = np.std(f1_scores_mean[:-1])
        accuracy_mean = np.mean([fold['accuracy'] for fold in fold_metrics])
        accuracy_std = np.std([fold['accuracy'] for fold in fold_metrics])
        test_loss_mean = np.mean([fold['test_loss'] for fold in fold_metrics])
        test_loss_std = np.std([fold['test_loss'] for fold in fold_metrics])

        f1_all_mean = np.mean([f1_scores_mean[0],f1_scores_mean[1]])
        f1_all_std = np.std([f1_scores_mean[0],f1_scores_mean[1]])
        method_result = []
        for i in range(2):
            method_result.append(f'{f1_scores_mean[i]:.3f}±{f1_scores_std[i]:.3f}')
        method_result.append(f'{f1_all_mean:.3f}±{f1_all_std :.3f}')
        method_result.append(f'{accuracy_mean:.3f}±{accuracy_mean:.3f}')
        all_results.append(method_result)
        print(
            f"F1 Scores Mean: {', '.join([f'{f1_scores_mean[i]:.3f}' for i in range(2)])} ± {', '.join([f'{f1_scores_std[i]:.3f}' for i in range(2)])}")
        print(f"Accuracy Mean: {accuracy_mean:.3f} ± {accuracy_std:.3f}")
        print("method end","|",getNowTime())

    all_results_path = filedir + '/result/val_chapman_results.csv'
    with open(all_results_path, mode='w', newline='',encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(all_results)











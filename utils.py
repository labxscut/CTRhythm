from torch import nn as nn
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import datetime


def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device)  
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # f1_scores
    f1_scores = f1_score(y_true, y_pred, average=None)  

    # F1All
    f1_nao = sum(f1_scores[0:-1]) / (len(f1_scores) - 1)

    # accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # loss
    average_loss = total_loss / total_samples

    return conf_matrix, f1_scores,  f1_nao, accuracy, average_loss

def train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, device='cpu',evaluate=True):
    model.to(device)  
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        start_time = datetime.datetime.now() 
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        end_time = datetime.datetime.now() 
        if evaluate and epoch % 5 == 4:
            model.eval()
            average_train_loss = total_loss / len(train_loader)
            conf_matrix, f1_scores, f1_nao, accuracy, test_loss = evaluate_model(model, test_loader, criterion,device)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {average_train_loss:.3f} - Test Loss: {test_loss:.3f} - Accuracy: {accuracy:.3f} - F1 Nao: {f1_nao:.3f} - F1 Scores: {', '.join([f'{f1_score:.3f}' for f1_score in f1_scores])}")
            
                
            print("Time:", end_time - start_time)  



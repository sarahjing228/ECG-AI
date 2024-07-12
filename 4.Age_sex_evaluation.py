import os
import torch
import numpy as np
import pandas as pd
from model import AgeSexModel
from funcs import randomseed
from torch.utils.data import DataLoader
from data_prepare import ASDataset
from torch.utils.data import DataLoader,Dataset

def load_pt_dataset(data_path):
    data = pd.read_csv(data_path)
    x_data = data.drop(columns=['time', 'event'])  # assuming 'time' and 'event' are your targets
    y_time = data['time']
    y_event = data['event']

    dataset = ASDataset(x_data, y_time, y_event)
    return DataLoader(dataset, batch_size=64, shuffle=True)

def evaluation(model, data_loader, datatype, num_event=2):
    model.eval()
    time_list = []
    event_list = []
    predict_prob_list = []

    with torch.no_grad():
        for inputs, time, event, mask1, mask2 in data_loader:  # Assumed data structure
            outputs = model(inputs)
            # extracting the probabilities of events
            predict_prob = torch.sum(outputs, dim=1).numpy()
            predict_prob_list.append(predict_prob)
            time_list.append(time.numpy())
            event_list.append(event.numpy())
    
    df_time = pd.Series([x for row in time_list for x in row])
    df_event = pd.Series([x for row in event_list for x in row])
    df_predict_prob = pd.Series([x for row in predict_prob_list for x in row])
    
    # Write to CSV file
    for E in range(num_event):
        comsumdf = pd.DataFrame(np.cumsum(df_predict_prob[E::num_event], axis=0), columns=[f"Day{i}" for i in range(num_event)])
        comsumdf['event'] = df_event
        comsumdf['time'] = df_time
        comsumdf.to_csv(f'../probcsv/comsum_EVENT{E+1}_ASmodel_{datatype}.csv', index=False)


def main(layernum = 1, Nodes=10, LR=0.01, alpha= 1.0, beta = 1000, epochsnum=100, model_folder_path=None):
    randomseed()
    train_data = torch.load('../main_train_dataset_AS.pt')
    valid_data = torch.load('../main_valid_dataset_AS.pt')
    test_data = torch.load('../main_test_dataset_AS.pt')
    
    train_dataset = ASDataset(train_data)
    valid_dataset = ASDataset(valid_data)
    test_dataset = ASDataset(test_data)

    # dataload
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create your model
    model = AgeSexModel(layernum=1, nodes=10, lr=0.01, num_period=9, num_event=2, alpha=1.0, beta=1.0, epochsnum=100)

    # Train your model
    AgeSexModel(model, train_data_loader, valid_data_loader, epochsnum=100)

    # Load the best model
    model.load_state_dict(torch.load('..//best_model.pt'))

    # Evaluate your model
    evaluation(model, valid_data_loader, "Valid", num_event=2)
    evaluation(model, test_data_loader, "Test", num_event=2)
    

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import copy 
from copy import deepcopy
import time 
import os
from tqdm import tqdm
import seaborn as sns



def custom_loss(time, event, prob_matrix, mb_mask1, mb_mask2, num_Event, num_Category, alpha, beta):
    ### LOSS-FUNCTION 1 -- Log-likelihood loss
    I_1 = torch.sign(event)
    #for uncenosred: log P(T=t,K=k|x)
    tmp1 = torch.sum(mb_mask1 * prob_matrix, dim=2).sum(dim=1, keepdim=True)
    tmp1 = I_1 * torch.log(tmp1)
    # for censored: log \sum P(T>t|x) 
    tmp2 = torch.sum(mb_mask1 * prob_matrix, dim=2).sum(dim=1, keepdim=True)
    tmp2 = (1. - I_1) * torch.log(tmp2)
    # sum of Log-likelihood loss
    LOSS_1 = - torch.mean(tmp1 + 1.0*tmp2)

    ### LOSS-FUNCTION 2 -- Ranking loss
    sigma1 = torch.FloatTensor([0.1])
    eta = []
    for e in range(num_Event):
        one_vector = torch.ones_like(time)
        I_2 = (event == e+1).float()
        I_2 = torch.diag(I_2.squeeze())
        tmp_e = prob_matrix[:, e, :].reshape(-1, num_Category)
        R = torch.mm(tmp_e, torch.transpose(mb_mask2, 0, 1))
        diag_R = torch.diag(R).reshape(-1, 1)
        R = torch.mm(one_vector, diag_R.transpose(0, 1)) - R
        R = R.transpose(0, 1)                               
        T = F.relu(torch.sign(torch.mm(one_vector, time.transpose(0, 1)) - torch.mm(time, one_vector.transpose(0, 1))))
        T = torch.mm(I_2, T) 
        tmp_eta = torch.mean(T * torch.exp(-R/sigma1), axis=1, keepdim=True)
        eta.append(tmp_eta)
    eta = torch.stack(eta, axis=1)
    eta = torch.mean(eta.reshape(-1, num_Event), axis=1, keepdim=True)
    LOSS_2 = torch.mean(eta)
    return LOSS_1, LOSS_2
 

class CustomCheckpoint(torch.nn.Module):
    def __init__(self, encoder, filepath):
        super().__init__()
        self.monitor = 'val_loss'
        self.best = float('inf')
        self.filepath = filepath
        self.encoder = encoder

    def forward(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            torch.save(self.encoder.state_dict(), self.filepath)

## Model for age and sex

class AgeSexModel(nn.Module):
    def __init__(self, layernum=1, nodes=10, lr=0.01, num_period=9, num_event=2, alpha=1.0, beta=1.0, epochsnum=100):
        super(AgeSexModel, self).__init__()

        self.layernum = layernum
        self.Nodes = nodes
        self.num_event = num_event
        self.num_period = num_period
        self.alpha = alpha
        self.beta = beta
        self.layers = nn.ModuleList()
        for _ in range(layernum):
            self.layers.append(nn.Linear(2 if not self.layers else nodes, nodes))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.BatchNorm1d(nodes))
            self.layers.append(nn.Linear(nodes, num_event * num_period))
            self.layers.append(nn.Softmax(dim=1))
            self.layers.append(nn.Linear(num_event * num_period, num_event * num_period))  # Reshape layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.reshape(-1, self.num_event, self.num_period)

    def train_model(model, train_dataloader, valid_dataloader, LR=0.01, epochsnum=100, model_folder_path='../model/Age_sex/'):
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # Create a learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        # Define early stopping parameters
        n_epochs_stop = 10
        epochs_no_improve = 0
        best_loss = float('inf')

        # For loss plotting
        train_losses = []
        valid_losses = []

        # Training loop
        for epoch in range(epochsnum):
            running_loss = 0.0
            model.train()
            for inputs, time, event, mask1, mask2 in train_dataloader:  # Assumed data structure
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_1, loss_2 = custom_loss(time, event, outputs, mask1, mask2, model.num_event, model.num_period, model.alpha, model.beta)
                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_losses.append(running_loss / len(train_dataloader))

            # Validation
            model.eval()
            with torch.no_grad():
                valid_loss = 0.0
                for inputs, time, event, mask1, mask2 in valid_dataloader:  # Assumed data structure
                    outputs = model(inputs)
                    loss_1, loss_2 = custom_loss(time, event, outputs, mask1, mask2, model.num_event, model.num_period, model.alpha, model.beta)
                    loss = loss_1 + loss_2
                    valid_loss += loss.item()
            valid_losses.append(valid_loss / len(valid_dataloader))

            # Early stopping
            if valid_loss < best_loss:
                epochs_no_improve = 0
                best_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(model_folder_path, 'best_model.pt'))
            else:
                epochs_no_improve += 1
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    break

        # Plotting the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


class Encoder01(nn.Module):
    def __init__(self, NumCh=16, seed_val=123):
        super(Encoder01, self).__init__()
        torch.manual_seed(seed_val)

        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, NumCh, (12, 7), padding=(6, 3))  # padding="same" means padding to preserve output dimensions
        self.leakyrelu1 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(NumCh)
        self.conv2 = nn.Conv2d(NumCh, NumCh, (12, 7), stride=(12, 1), padding=(6, 3))
        self.leakyrelu2 = nn.LeakyReLU()

        self.skip_conv = nn.Conv2d(1, NumCh, (1, 1))  # skip connection
        self.pool = nn.MaxPool2d((12, 1), padding=(6, 0))  # same padding

    def forward(self, x):
        #print(x)
        process1 = self.bn1(x)
        process1 = self.conv1(process1)
        process1 = self.leakyrelu1(process1)
        process1 = self.bn2(process1)
        process1 = self.conv2(process1)
        process1 = self.leakyrelu2(process1)

        process2 = self.skip_conv(x)
        process2 = self.pool(process2)

        en_process = process1 + process2  

        return en_process

class Encoder02(nn.Module):
    def __init__(self, NumCh=16, i=1, StepUp=True, seed_val=123):
        super(Encoder02, self).__init__()
        
        torch.manual_seed(seed_val)

        ResNum = i // 3 + 1 
        if StepUp:
            Ch_hold = ResNum
            pre_Ch_hold = (i - 1) // 3 + 1
        else:
            Ch_hold = 1
            pre_Ch_hold = 1

        temp_Kernel = 9 - ResNum * 2

        self.bn1 = nn.BatchNorm2d(pre_Ch_hold * NumCh)
        self.conv1 = nn.Conv2d(pre_Ch_hold * NumCh, Ch_hold * NumCh, (temp_Kernel, 1), padding=(temp_Kernel//2, 0))
        self.bn2 = nn.BatchNorm2d(Ch_hold * NumCh)
        self.conv2 = nn.Conv2d(Ch_hold * NumCh, Ch_hold * NumCh, (temp_Kernel, 1), stride=(1, 1), padding=(temp_Kernel//2, 0))  
        self.skip_conv = nn.Conv2d(pre_Ch_hold * NumCh, Ch_hold * NumCh, (1, 1))
        self.pool = nn.MaxPool2d((2, 1))

    def forward(self, x):
        # conv layers
        process1 = self.bn1(x)
        process1 = F.leaky_relu(self.conv1(process1))
        process1 = self.bn2(process1)
        process1 = F.leaky_relu(self.conv2(process1))

        # skip layers
        process2 = self.skip_conv(x)
        process2 = self.pool(process2)

        #print(process1.shape)
        #print(process2.shape)

        en_process = process1 + process2

        return en_process

class LSTMLayer(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(LSTMLayer, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        output, _ = super().forward(*args, **kwargs)
        return output


class Encoder03(nn.Module):
    def __init__(self, ResBlock=3, NumCh=16, StepUp=True, seed_val=123, Return_seq=True):
        super(Encoder03, self).__init__()

        torch.manual_seed(seed_val)

        if ResBlock==3:
            self.len_seq=8
        elif ResBlock==2:
            self.len_seq=64
        elif ResBlock==1:
            self.len_seq=512
    
        if StepUp:
            self.Ch_hold = ResBlock
        else:
            self.Ch_hold = 1

        self.return_seq = Return_seq
        self.NumCh = NumCh
    
        self.lstm1 = LSTMLayer(input_size=self.len_seq * NumCh * self.Ch_hold,
                             hidden_size=int(self.len_seq * NumCh * self.Ch_hold / 2),
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)
    def forward(self, x):

        batch_size = x.size(0)
        num_features = self.len_seq * self.NumCh * self.Ch_hold
        x = x.view(batch_size, -1, num_features)

        x = self.lstm1(x)

        if not self.return_seq:
            x = x[:, -1, :]

        return x


class PTModel(nn.Module):
    def __init__(self, ResBlock=3, LSTM=True, NumCh=24, DropRate=0.2, StepUp=True, LR=0.0001, seed_val=123):
        super(PTModel, self).__init__()
        self.ResBlock = ResBlock
        self.LSTM = LSTM
        self.NumCh = NumCh
        self.LR = LR
        self.DropRate = DropRate
        self.StepUp = StepUp

        # Initialize the encoder modules
        self.en01 = Encoder01(NumCh=NumCh, seed_val=seed_val)
        self.en02 = Encoder02(NumCh=NumCh, i=1, StepUp=StepUp, seed_val=seed_val)
        self.en03 = Encoder02(NumCh=NumCh, i=2, StepUp=StepUp, seed_val=seed_val)
        self.en04 = Encoder02(NumCh=NumCh, i=3, StepUp=StepUp, seed_val=seed_val)
        self.en05 = Encoder02(NumCh=NumCh, i=4, StepUp=StepUp, seed_val=seed_val)
        self.en06 = Encoder02(NumCh=NumCh, i=5, StepUp=StepUp, seed_val=seed_val)
        self.en07 = Encoder02(NumCh=NumCh, i=6, StepUp=StepUp, seed_val=seed_val)
        self.en08 = Encoder02(NumCh=NumCh, i=7, StepUp=StepUp, seed_val=seed_val)
        self.en09 = Encoder02(NumCh=NumCh, i=8, StepUp=StepUp, seed_val=seed_val)
        self.en10 = Encoder03(ResBlock=ResBlock, NumCh=NumCh, StepUp=StepUp, seed_val=seed_val, Return_seq=True)
        self.en11 = Encoder03(ResBlock=ResBlock, NumCh=NumCh, StepUp=StepUp, seed_val=seed_val, Return_seq=False)
        

        self.dropout = nn.Dropout(p=self.DropRate)

        # Initialize the output layers
        self.age_layer = nn.Linear(self.en11.len_seq * NumCh * self.en11.Ch_hold, 1) 
        self.gender_layer = nn.Linear(self.en11.len_seq * NumCh * self.en11.Ch_hold, 1)


        # Dropout layer
        self.dropout = nn.Dropout(DropRate)

    def forward(self, x):
        x = self.en01(x)
        x = self.en02(x)
        x = self.en03(x)
        x = self.dropout(x)
        if self.ResBlock >= 2:
            x = self.en04(x)
            x = self.en05(x)
            x = self.en06(x)
            x = self.dropout(x)
        if self.ResBlock == 3:
            x = self.en07(x)
            x = self.en08(x)
            x = self.en09(x)
            x = self.dropout(x)
        if self.LSTM:
            x = self.en10(x)
            x = self.en11(x)
            #print("Shape of x after Encoder03: ", x.shape)
            x = self.dropout(x)
        # output layer
        x = torch.flatten(x, start_dim=1)
        age_pred = self.age_layer(x)
        gender_pred = torch.sigmoid(self.gender_layer(x))
        return age_pred, gender_pred

    def train_model(self, train_dataloader, val_dataloader, epochsnum=100, patience=10):
    # Define loss functions and optimizer
        age_criterion = nn.L1Loss()  # equivalent to Mean Absolute Error
        gender_criterion = nn.BCEWithLogitsLoss()  # equivalent to Binary Cross Entropy
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)

        # To track validation loss for early stopping
        best_val_loss = float('inf')
        best_model = None
        epochs_no_improve = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        checkpoint_dir = '../checkpoints'

        # Create the directory for checkpoints
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Create lists to store losses for each epoch
        
        train_losses = []
        val_losses = []

        # Create lists to store losses for age and gender for each epoch
        train_age_losses = []
        train_gender_losses = []
        val_age_losses = []
        val_gender_losses = []

        start_time = time.time()
        # Train-validation loop
        for epoch in range(epochsnum):
            self.train()
            # Training loop
            train_loss = 0
            train_age_loss = 0  
            train_gender_loss = 0  
            pbar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch_index, (inputs, targets) in enumerate(pbar):
                print(f'Batch size: {inputs.size(0)}')
                inputs = inputs.to(device)
                targets = [target.to(device) for target in targets]

                # Zero the gradients
                optimizer.zero_grad()
            
                # Forward pass
                age_pred, gender_pred = self.forward(inputs)
                age_targets, gender_targets = targets
                age_targets = age_targets.unsqueeze(1)
                gender_targets = gender_targets.unsqueeze(1)
                age_loss = age_criterion(age_pred, age_targets)
                gender_loss = gender_criterion(gender_pred, gender_targets)
                total_loss = age_loss + gender_loss

                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item()
                train_age_loss += age_loss.item()  # add
                train_gender_loss += gender_loss.item()  # add
                pbar.set_description(f"Epoch {epoch+1}, Train Loss: {train_loss / (batch_index+1):.4f}")
                
            # Store average training loss and age and gender losses
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            train_age_loss /= len(train_dataloader) 
            train_age_losses.append(train_age_loss)  
            train_gender_loss /= len(train_dataloader)  
            train_gender_losses.append(train_gender_loss)  

            # Validation loop


            self.eval() 
            val_loss = 0
            val_age_loss = 0  
            val_gender_loss = 0  
            pbar_val = tqdm(val_dataloader, total=len(val_dataloader))
            with torch.no_grad():
                for batch_index_val, (inputs, targets) in enumerate(pbar_val):
                    inputs = inputs.to(device)
                    age_targets, gender_targets = targets
                    age_targets = age_targets.to(device).unsqueeze(1)
                    gender_targets = gender_targets.to(device).unsqueeze(1)
        
                    age_pred, gender_pred = self.forward(inputs)
                    age_loss = age_criterion(age_pred, age_targets)
                    gender_loss = gender_criterion(gender_pred, gender_targets)
                    loss = age_loss + gender_loss
                    val_loss += loss.item()
                    val_age_loss += age_loss.item() 
                    val_gender_loss += gender_loss.item()  
                    pbar_val.set_description(f"Epoch {epoch+1}, Validation Loss: {val_loss / (batch_index_val+1):.4f}")                   

       
            # Compute average validation loss and age and gender losses
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            val_age_loss /= len(val_dataloader)  
            val_age_losses.append(val_age_loss)  
            val_gender_loss /= len(val_dataloader)  
            val_gender_losses.append(val_gender_loss)  

            print(f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss {val_loss}')
            print(f'Epoch {epoch}, Train Age Loss: {train_age_loss}, Validation Age Loss {val_age_loss}')
            print(f'Epoch {epoch}, Train Gender Loss: {train_gender_loss}, Validation Gender Loss {val_gender_loss}')
            
            # check for improvement in validation loss
            if val_loss < best_val_loss:
                print(f'Validation loss improved from {best_val_loss} to {val_loss}. Saving model...')
                best_val_loss = val_loss
                best_model = deepcopy(self.state_dict())
                epochs_no_improve = 0
                checkpoint_file = os.path.join(checkpoint_dir, f'model_best_{epoch}.pt')
                torch.save(self.state_dict(), checkpoint_file)
            else:
                epochs_no_improve += 1
                print(f'No improvement in validation loss for {epochs_no_improve} epochs.')
                if epochs_no_improve >= patience:
                    print('Early stopping triggered.')
                    self.load_state_dict(best_model)
                    break
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time :{training_time} seconds")
       
        # plot the train and validation loss per epoch
        plt.figure(figsize=(10, 7))
        plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('../tloss_plot1.png')

        # plot the train and validation age loss per epoch
        plt.figure(figsize=(10, 7))
        plt.plot(range(len(train_age_losses)), train_age_losses, label='Train Age Loss')
        plt.plot(range(len(val_age_losses)), val_age_losses, label='Validation Age Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Age Loss over time')
        plt.legend()
        plt.savefig('../ageloss_plot1.png')

        # plot the train and validation gender loss per epoch
        plt.figure(figsize=(10, 7))
        plt.plot(range(len(train_gender_losses)), train_gender_losses, label='Train Gender Loss')
        plt.plot(range(len(val_gender_losses)), val_gender_losses, label='Validation Gender Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Gender Loss over time')
        plt.legend()
        plt.savefig('../_plot1.png')

 
class MainModel(nn.Module):
    def __init__(self, 
                ResBlock = 3, LSTM=True, NumCh=24, DropRate=0.2, StepUp=False,
                LR=0.0001, rc=True, seed_val=123, PT=True,
                num_period = 9,num_event = 2, alpha= 1.0, beta = 1000.0, 
                model_folder_path="../model/PT/"):
        
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)

        super(MainModel, self).__init__()

        self.LR=LR
        self.alpha = alpha
        self.beta = beta
        self.num_event = num_event
        self.num_period = num_period
        self.model_folder_path = model_folder_path

        self.en01 = Encoder01(NumCh=NumCh,seed_val=seed_val)
        self.en02 = Encoder02(NumCh=NumCh, i=1, StepUp=StepUp, seed_val=seed_val)
        self.en03 = Encoder02(NumCh=NumCh, i=2, StepUp=StepUp, seed_val=seed_val)
        self.en04 = Encoder02(NumCh=NumCh, i=3, StepUp=StepUp, seed_val=seed_val)
        self.en05 = Encoder02(NumCh=NumCh, i=4, StepUp=StepUp, seed_val=seed_val)
        self.en06 = Encoder02(NumCh=NumCh, i=5, StepUp=StepUp, seed_val=seed_val)
        self.en07 = Encoder02(NumCh=NumCh, i=6, StepUp=StepUp, seed_val=seed_val)
        self.en08 = Encoder02(NumCh=NumCh, i=7, StepUp=StepUp, seed_val=seed_val)
        self.en09 = Encoder02(NumCh=NumCh, i=8, StepUp=StepUp, seed_val=seed_val)
        self.en10 = Encoder03(ResBlock=ResBlock, NumCh=NumCh, StepUp=StepUp, seed_val=seed_val, Return_seq=True)
        self.en11 = Encoder03(ResBlock=ResBlock, NumCh=NumCh, StepUp=StepUp, seed_val=seed_val, Return_seq=False)
        #print("Shape of x after Encoder03: ", x.shape)

        # use PyTorch's torch.nn.Dropout() function for dropout
        self.dropout = nn.Dropout(DropRate)

        # output layer
        self.output_layer = nn.Linear(ResBlock * NumCh, num_event*num_period)

    def forward(self, x):
        # Reshape the input
        x = x.view(-1, 1, 5000, 12)

        # Apply the model
        processed = self.en01(x)
        processed = self.en02(processed)
        processed = self.en03(processed)
        processed = self.dropout(processed)

        if ResBlock >= 2:
            processed = self.en04(processed)
            processed = self.en05(processed)
            processed = self.en06(processed)
            processed = self.dropout(processed)
    
        if ResBlock == 3:
            processed = self.en07(processed)
            processed = self.en08(processed)
            processed = self.en09(processed)
            processed = self.dropout(processed)

        if LSTM:
            processed = self.en10(processed)
            processed = self.en11(processed)
            processed = self.dropout(processed)

        processed = self.output_layer(processed.view(processed.size(0), -1))

        # reshape the output as needed
        output_data = processed.view(-1, self.num_event, self.num_period)
        return output_data

    def train_model(self, train_dataset, validation_dataset, num_epochs):
        # Define the optimizer
        optimizer = Adam(self.parameters(), lr=self.LR)
        
        # Define the loss function
        criterion = nn.CrossEntropyLoss()

        # Convert the datasets into DataLoader instances
        train_loader = DataLoader(train_dataset, batch_size=32)
        validation_loader = DataLoader(validation_dataset, batch_size=32)

        for epoch in range(num_epochs):
            # Training
            for x_train in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(x_train)
                loss = criterion(outputs, x_train)
                loss.backward()
                optimizer.step()

            # Validation
            with torch.no_grad():
                for x_val in validation_loader:
                    outputs = self.forward(x_val)
                    val_loss = criterion(outputs, x_val)
                    print("Epoch: {}, Validation Loss: {}".format(epoch, val_loss))


def ensemble_forward(models, input):
    outputs = [model(input) for model in models]
    return sum(outputs) / len(outputs)

def ensemble_model(ResBlock = 3, LSTM=True, NumCh=24, DropRate=0.2, StepUp=False, ensemble_num=10, model_folder_path=None):
    models = []
    for i in range(ensemble_num):
        model = MainModel(PT=False, 
                        ResBlock=ResBlock,
                        LSTM=LSTM,
                        NumCh=NumCh,
                        StepUp=StepUp,
                        DropRate=DropRate,
                        seed_val=i+123)
        model.load_state_dict(torch.load(model_folder_path + 'Mainmodel.h5'))
        models.append(model)
    return models

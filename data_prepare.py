import torch
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader

# Load CSV data
PT_train_df = pd.read_csv('../Train_Data.csv')
PT_valid_df = pd.read_csv('../Validation_Data.csv')

# Save as jb format
joblib.dump(PT_train_df, '../PTTrain_Data.jb')
joblib.dump(PT_valid_df, '../PTValidation_Data.jb')


# fmask1
def f_get_fc_mask1(time_data, label, num_Event, num_Category): 
    time_data = time_data.values
    mask = np.zeros([np.shape(time_data)[0], num_Event, num_Category]) 
    for i in range(np.shape(time_data)[0]):
        if label[i] != 0:
            mask[i,int(label[i]-1),int(time_data[i])] = 1
        else: 
            mask[i,:,int(time_data[i]+1):] =  1 
    return mask

# fmask2
def f_get_fc_mask2(time_data, num_Category):
    time_data = time_data.values
    mask = np.zeros([np.shape(time_data)[0], num_Category])
    for i in range(np.shape(time_data)[0]):
        t = int(time_data[i]) 
        mask[i,:(t+1)] = 1 
    return mask 

# Pre-train dataset
class PTDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item_path = self.df.iloc[idx]['path']
        # Read the CSV file
        df_item = pd.read_csv(item_path,encoding="utf-8")
        
        # Convert all columns to numeric, if possible
        for column in df_item.columns:
            df_item[column] = pd.to_numeric(df_item[column], errors='coerce')

        df_item = df_item.fillna(0)

        X_data = np.clip((df_item.values/4096),-1,1)
        X_data = X_data.transpose((1, 0)) 
        X_data = X_data[:, :, np.newaxis]  
        X_data = torch.tensor(X_data, dtype=torch.float32).permute(2, 0, 1)  

        age = torch.tensor(self.df.iloc[idx]['Age'], dtype=torch.float32)
        female = torch.tensor(self.df.iloc[idx]['Female'], dtype=torch.float32)
        targets = torch.stack([age, female])
        
        return X_data, (age, female)
        #print(type(self.x_data), type(self.y_time), type(self.y_event), type(self.df_mask1), type(self.df_mask2))


# Main dataset
class Main_Dataset(Dataset):
    def __init__(self, in_df, num_event=2, num_period=9):
        self.csv_paths = in_df['path'].to_list()
        self.X_data = []
        for item_path in self.csv_paths:
            data = pd.read_csv(item_path, header=None, encoding="utf-8").fillna(0).values
            data = np.clip(data / 4096, -1, 1)
            self.X_data.append(data)
        self.X_data = torch.tensor(np.array(self.X_data), dtype=torch.float32)
        
        self.y_time = torch.tensor(in_df['time'].values, dtype=torch.float32)
        self.y_event = torch.tensor(in_df['event'].values, dtype=torch.float32)
        self.df_mask1 = f_get_fc_mask1(in_df['time'], in_df['event'], num_event, num_period)
        self.df_mask2 = f_get_fc_mask2(in_df['time'], num_period)

    def __len__(self):
        return len(self.csv_paths)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_time[idx], self.y_event[idx], self.df_mask1[idx], self.df_mask2[idx]

# Age and sex dataset
class ASDataset(Dataset):
    def __init__(self, in_df, num_event=2, num_period=9):
        x_data_np = in_df[['Age',"Female"]].values
        self.x_data = torch.tensor(x_data_np, dtype=torch.float32)
        y_time_np = in_df['time'].values
        self.y_time = torch.tensor(y_time_np, dtype=torch.float32)
        y_event_np = in_df['event'].values
        self.y_event = torch.tensor(y_event_np, dtype=torch.float32)
        df_mask1_np = f_get_fc_mask1(in_df['time'], in_df['event'], num_event, num_period)
        self.df_mask1 = torch.tensor(df_mask1_np, dtype=torch.float32)
        df_mask2_np = f_get_fc_mask2(in_df['time'], num_period)
        #print("Shape of df_mask2_np:", df_mask2_np.shape)
        self.df_mask2 = torch.tensor(df_mask2_np, dtype=torch.float32)
        #print("Shape of df_mask2 tensor:", self.df_mask2.shape)

    # ... rest of your class
    def fillna(self, value):
        self.data.fillna(value, inplace=True)

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_time[idx], self.y_event[idx], self.df_mask1[idx], self.df_mask2[idx]
        #print(type(self.x_data), type(self.y_time), type(self.y_event), type(self.df_mask1), type(self.df_mask2))

def main():
    # pretraining data (train & val data for stacked autoencoder)
    pretrain_Traindf = joblib.load('../PTTrain_Data.jb')
    pretrain_Validdf = joblib.load('../PTValidation_Data.jb')
    

    pretrain_train_dataset =  PTDataset(pretrain_Traindf)
    pretrain_valid_dataset =  PTDataset(pretrain_Validdf)

    train_dataloader = DataLoader(pretrain_train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(pretrain_valid_dataset, batch_size=64, shuffle=False)

    torch.save(train_dataloader, '../PTtrain_dataloader.pt')
    torch.save(valid_dataloader, '../PTvalid_dataloader.pt')


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader
from model2 import PTModel
from data_prepare import PTDataset
import faulthandler

faulthandler.enable()
import sys
sys.setrecursionlimit(2000)

def pad_collate(batch):
    # Find the maximum sequence length in the batch
    max_length = max([sample[0].size(2) for sample in batch])
    
    # Padding inputs
    padded_inputs = [torch.nn.functional.pad(sample[0], (0, max_length - sample[0].size(2))) for sample in batch]
    padded_inputs = torch.stack(padded_inputs)

    # Extracting age and gender targets
    age_targets = [sample[1][0] for sample in batch]
    gender_targets = [sample[1][1] for sample in batch]

    age_targets = torch.stack(age_targets)
    gender_targets = torch.stack(gender_targets)
    
    return padded_inputs, (age_targets, gender_targets)


def main(model_folder_path="../model/PT/", NumCh=24, StepUp=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if CUDA is available

    # Load the saved dataloaders
    saved_train_dataloader = torch.load('../PTtrain_dataloader.pt')
    saved_valid_dataloader = torch.load('../PTvalid_dataloader.pt')

    # Recreate dataloaders with the custom collate function
    train_dataloader = DataLoader(saved_train_dataloader.dataset, batch_size=saved_train_dataloader.batch_size, shuffle=True, collate_fn=pad_collate)
    valid_dataloader = DataLoader(saved_valid_dataloader.dataset, batch_size=saved_valid_dataloader.batch_size, shuffle=False, collate_fn=pad_collate)

    # Model instantiation
    PT = PTModel(NumCh=NumCh, StepUp=StepUp, LR=0.001) 
    PT.train_model(train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()

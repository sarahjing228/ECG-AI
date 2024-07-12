import os
from model import PTModel
# List all checkpoint files
checkpoint_dir = 'path_to_checkpoints_directory'
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]

# Dictionary to store evaluation results for each checkpoint
results = {}

# Evaluate each checkpoint
for checkpoint_file in checkpoint_files:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    
    # Load checkpoint into model
    model = PTModel()
    model = model.to(device)
    model = load_checkpoint(model, checkpoint_path)
    
    # Evaluate the model
    average_age_loss, average_gender_loss, gender_accuracy = evaluate_model(model, val_dataloader, device)
    
    # Store results
    results[checkpoint_file] = {
        'average_age_loss': average_age_loss,
        'average_gender_loss': average_gender_loss,
        'gender_accuracy': gender_accuracy
    }

# Select best checkpoint based on a criterion (e.g., gender accuracy)
best_checkpoint = max(results, key=lambda k: results[k]['gender_accuracy'])
print(f"Best checkpoint based on gender accuracy: {best_checkpoint}")

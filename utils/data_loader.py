import json
import numpy as np
import os
import torch
from models.autoencoder import AutoEncoder

# Data loading function
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Example preprocessing steps, modify as needed for your use case
    numeric_data = np.array(data['numeric_data'])
    string_data = np.array([len(s) for s in data['string_data']])  # Example: convert strings to their lengths
    mixed_data_scores = np.array([item['scores'] for item in data['mixed_data']])

    # Concatenate or otherwise combine your data as needed
    combined_data = np.concatenate([numeric_data, string_data, mixed_data_scores.flatten()])
    combined_data = combined_data.reshape(-1, 1)  # Reshape as needed for your model

    return combined_data

# Main script
if __name__ == "__main__":
    # Load data
    data_path = os.path.join('data', 'data.json')
    X_train = load_data(data_path)

    # Initialize and train the model
    clf = AutoEncoder(contamination=0.05, epoch_num=50, batch_size=64)
    clf.fit(X_train)

    # Save the model
    model_save_path = os.path.join('models', 'autoencoder.pth')
    torch.save(clf.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

from labgym_encoder import LabGym_Encoder
from labgym_transformer import BehaviorTransformer, training, validate, predict, save_model, load_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pathlib
import math
import numpy as np

# Paths for synthetic data
synthetic_folder_path = "/Users/jackmcclure/Desktop/pip_LabGym_version_control/synthetic_test"
synthetic_data_path = pathlib.Path(synthetic_folder_path, "synthetic_data.xlsx")

# Paths for test all_events.xlsx in the pip package
all_events_path = pathlib.Path("all_events.xlsx")

# Using synthetic data
tokenizer = LabGym_Encoder(file_path=synthetic_data_path, target_row=0)
encoded_sequence = tokenizer.grab_encoded_behaviors_sequence()
print(encoded_sequence)
data = torch.tensor(encoded_sequence, dtype=torch.int)

# Add one because you shift up one behavior index b/c 0 is padding
vocab_size = tokenizer.get_num_behaviors()+1

embedding_dimensions = 128

num_heads = 8

src_len = 30
tgt_len = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))

encodings = tokenizer.get_behavior_map()
model = BehaviorTransformer(vocab_size, embedding_dimensions, num_heads, src_len, tgt_len, encodings).to(device)
training(model, data)
validate(model, data)
# Save to synthetic data path
save_model(model=model, filefolder=synthetic_folder_path)
# Load model from synthetic data path
model = load_model(pathlib.Path(synthetic_folder_path, "transformer_model.pth")).to(device)
input_sequence = encoded_sequence[len(encoded_sequence)//2+30:len(encoded_sequence)//2+60]  # Last 30 elements as input
predictions = predict(model, input_sequence)
print("Input sequence (encodings):", input_sequence)
print("Input sequence:", tokenizer.convert_tokens_to_behaviors(input_sequence))
print("Predicted next behaviors (encodings):", predictions[0])
print("Predicted next behaviors:", tokenizer.convert_tokens_to_behaviors(predictions[0]))
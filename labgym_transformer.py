from labgym_encoder import LabGym_Encoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

class BehaviorTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dimensions,
        num_heads,
        src_len,
        tgt_len,
        encodings,
    ):
        
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dimensions = embedding_dimensions
        self.num_heads = num_heads
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.encodings = encodings


        self.input_token_embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dimensions)

        self.register_buffer("src_pos_encoding", self.PositionalEncoding(src_len, embedding_dimensions).unsqueeze(0))
        self.register_buffer("tgt_pos_encoding", self.PositionalEncoding(tgt_len, embedding_dimensions).unsqueeze(0))

        self.transformer = nn.Transformer(
            d_model = self.embedding_dimensions,
            nhead = self.num_heads,
            num_encoder_layers = 6,
            num_decoder_layers = 6,
            batch_first = True,
        )

        self.final_linear = nn.Linear(self.embedding_dimensions, self.vocab_size)


    # Sinusoidal positional encoding
    def PositionalEncoding(self, max_sequence_length, embedding_dimensions):
        # Tensor of zeros to store positional encoding vector for each item in sequence
        encoding = torch.zeros(max_sequence_length, embedding_dimensions, device=self.input_token_embedding_layer.weight.device)
        # For each item in the sequence
        for pos in range(max_sequence_length):
            #For each embedding dimension
            for i in range(0, embedding_dimensions, 2):
                # Formulas for positional encoding from Attention Is All You Need paper
                encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embedding_dimensions)))
                # Could go out of bounds if embedding_dimensions is odd
                if(i+1 < embedding_dimensions):
                    encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / embedding_dimensions)))
        return encoding

    def forward(self, src, tgt):    
        src = src.to(torch.int64)
        tgt = tgt.to(torch.int64)
        src = self.input_token_embedding_layer(src) + self.src_pos_encoding
        tgt = self.input_token_embedding_layer(tgt) + self.tgt_pos_encoding

        transformer_output = self.transformer(
            src, 
            tgt, 
            # Don't want time series to look ahead and cheat during training
            tgt_mask = self.transformer.generate_square_subsequent_mask(self.tgt_len).to(src.device),

            # Because all sequence lengths in batch should be the same as determined in the time series dataset class
            src_key_padding_mask = None,
            tgt_key_padding_mask = None,
        )

        final_linear_layer_output = self.final_linear(transformer_output)

        return final_linear_layer_output
    

from torch.utils.data import Dataset
import torch

class TimeSeriesDataset(Dataset):
    # Concern: Data between clips may not be continuous, don't want to learn at boundaries
    # Solution: Store each sequence from each clip and make it something the dataloader can load from
    # Benefit: Allows batching - if you can train on all the data at once instead of calling train multiple times it's a lot faster
    def __init__(self, sequence_list, src_len, tgt_len):
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.windows = []

        for seq in sequence_list:
            seq = torch.tensor(seq, dtype=torch.int)
            max_i = len(seq) - src_len - tgt_len + 1
            if max_i < 1:
                continue  # Skip sequences that are too short
            for i in range(max_i):
                src = seq[i : i + src_len]
                tgt_input = seq[i + src_len - 1 : i + src_len + tgt_len - 1]
                tgt_output = seq[i + src_len : i + src_len + tgt_len]
                self.windows.append((src, tgt_input, tgt_output))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        src, tgt_input, tgt_output = self.windows[idx]
        return src.clone(), tgt_input.clone(), tgt_output.clone()

    
def training(model, data):
    print("Beginning model training!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    src_len = model.src_len
    tgt_len = model.tgt_len

    dataset = TimeSeriesDataset(data, src_len, tgt_len)
    # Shuffle was false, why?
    dataloader = DataLoader(dataset, batch_size=32, shuffle = True)

    data = np.array(data, dtype=np.int64).flatten()

    # number of classweights + 1?
    all_classes = np.arange(0, len(model.encodings) + 1)
    present_classes = np.unique(data)

    # Compute weights only for present classes
    weights_for_present = compute_class_weight(class_weight="balanced", classes=present_classes, y=data)

    # Build full weight vector
    class_weights_np = np.zeros(len(all_classes), dtype=np.float32)
    class_weights_np[present_classes] = weights_for_present

    # Now convert to tensor
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

    print("Class_weights:", class_weights)

    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 6

    model.train()


    num_epochs = 6

    model.train()

    training_losses = []
    for epoch in range(num_epochs):
        for src, tgt_input, tgt_output in dataloader:

            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            optimizer.zero_grad()
            output = model(src, tgt_input)

            tgt_output = tgt_output.view(-1).long()
            output = output.view(-1, model.vocab_size)

            loss = loss_function(output, tgt_output)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())
            # print("Epoch: " + str(epoch) + " Loss: " + str(loss.item()))
            if loss.item() < 0.01:
                epoch = num_epochs
                break
        print("Finished training epoch", epoch)
        print("End loss:", training_losses[-1])

    plt.plot(training_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.show()

def validate(model, data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    src_len = model.src_len
    tgt_len = model.tgt_len

    dataset = TimeSeriesDataset(data, src_len, tgt_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle = False)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.eval()
    total_loss = 0.0
    #Disable gradient calculation
    with torch.no_grad():
        for src, tgt_input, tgt_output in dataloader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            output = model(src, tgt_input)

            tgt_output = tgt_output.view(-1).long()
            output = output.view(-1, model.vocab_size)

            loss = loss_function(output, tgt_output)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def predict(model, input_sequence):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Using {} device".format(device))

    model.eval()

    #If the input sequence isn't already a tensor, make it one
    if not isinstance(input_sequence, torch.Tensor):
        input_sequence = torch.tensor(input_sequence, dtype=torch.int64)

    #Check that input_sequence is on the right device and has the correct shape
    input_sequence = input_sequence.to(device).unsqueeze(0)  # Shape: (1, src_len)

    # Create a target sequence filled with -1 (don't use 0 b/c zero is a possible behavior)
    # ^^IGNORE THE ABOVE, 0 is ok b/c changed behaviors so no 0 behavior
    # And 0 is never predicted because probability is zeroed
    target_sequence = torch.full((1, model.tgt_len), fill_value=0, dtype=torch.int64).to(device)


    with torch.no_grad():
        output = model(input_sequence, target_sequence)

    # Set padding token (0) probabilities very low before applying argmax so it's not chosen
    output[:, :, 0] = -float("inf")  # Ensure 0 is never chosen

    # Get the predictions (take the index of the highest probability for each time step)
    # Get the probability from softmax of output logits (relative probability for highest)
    predictions = torch.argmax(output, dim = -1).squeeze(0)
    probabilities = torch.nn.functional.softmax(output, dim=-1).squeeze(0)

    return predictions.cpu().tolist(), probabilities.cpu().tolist()

def save_model(model, filefolder, filename="transformer_model.pth"):
    """Saves the trained model's state dictionary."""
    model_info = {
        "encodings": model.encodings,
        "state_dict": model.state_dict(),
        "vocab_size": model.vocab_size,
        "embedding_dimensions": model.embedding_dimensions,
        "num_heads": model.num_heads,
        "src_len": model.src_len,
        "tgt_len": model.tgt_len,
    }
    path = os.path.join(filefolder, filename)
    torch.save(model_info, path)
    print(f"Model saved to {path}")

def load_model(filepath):
    """Loads the trained model from a file."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_info = torch.load(filepath, map_location=device)
    state_dict = model_info["state_dict"]
    model = BehaviorTransformer(
        model_info["vocab_size"],
        model_info["embedding_dimensions"],
        model_info["num_heads"],
        model_info["src_len"],
        model_info["tgt_len"],
        model_info["encodings"],
    )
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {filepath}")
    return model
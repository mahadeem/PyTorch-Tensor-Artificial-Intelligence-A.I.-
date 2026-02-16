import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json

##model
class InfoModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # Tell embedding 0 is padding
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        # 1. Embed the tokens
        x = self.embedding(x)
        
        # 2. "Pack" the sequence (hides the padding from LSTM)
        # lengths is a list of how many real tokens are in each JSON
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # 3. Pass through LSTM
        _, (h, _) = self.lstm(packed_x)
        
        # 4. Use the last hidden state for classification
        return self.fc(h[-1])


# 1. Dataset must now include labels and return lengths
class MultiJSONDataset(Dataset):
    def __init__(self, json_list, labels, word_to_idx):
        self.samples = []
        self.labels = labels
        for j_str in json_list:
            tokens = j_str.lower().split()
            indices = [word_to_idx.get(t, 0) for t in tokens]
            self.samples.append(torch.tensor(indices))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# 2. Updated Collate Function to handle lengths
def collate_fn(batch):
# Sort batch by length descending (required for some PyTorch versions)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)

    lengths = torch.tensor([len(s) for s in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return sequences_padded, labels, lengths

# --- Setup ---
raw_data = [
    '{"status": "success", "id": 101}', 
    '{"status": "success", "id": 102, "name": "Ahmed"}', 
    '{"status": "error"}'
]
raw_labels = [0, 0, 1] # 0 = Success, 1 = Error

# Build Vocab
all_tokens = " ".join(raw_data).lower().split()
unique_words = sorted(list(set(all_tokens)))
word_to_idx = {word: i+1 for i, word in enumerate(unique_words)}
word_to_idx["<PAD>"] = 0

# Initialize Model, Device, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InfoModel(vocab_size=len(word_to_idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Loader
ds = MultiJSONDataset(raw_data, raw_labels, word_to_idx)
loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

# --- Training Loop ---
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels, lengths in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass (now passing lengths!)
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")


    def predict_json(json_string, model, word_to_idx):
        model.eval() # Switch to evaluation mode
        with torch.no_grad():
            # 1. Preprocess
            tokens = json_string.lower().split()
            indices = [word_to_idx.get(t, 0) for t in tokens] # Use 0 if word is unknown
            
            # 2. Prepare Tensors
            input_tensor = torch.tensor([indices]).to(device) # Add batch dimension
            length = torch.tensor([len(indices)])
            
            # 3. Forward Pass
            output = model(input_tensor, length)
            
            # 4. Get Result
            _, predicted_class = torch.max(output, 1)
            
            class_map = {0: "Success", 1: "Error"}
            return class_map[predicted_class.item()]

# --- Test it out ---
new_json = '{"status": "error", "message": "auth failed"}'
result = predict_json(new_json, model, word_to_idx)
print(f"Prediction for new JSON: {result}")




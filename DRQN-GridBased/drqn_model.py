import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQN(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, output_size=5): # ⚠️ Changed 11 to 13
        super(DRQN, self).__init__()
        self.hidden_size = hidden_size

        # 1. Feature Extractor (MLP)
        # Converts the raw 11-integer array into a rich feature vector
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, hidden_size)

        # 2. Memory Core (LSTM)
        # batch_first=True makes the tensor shape (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        # 3. Q-Value Output Head
        # Outputs 5 numbers representing the predicted value of [Up, Down, Left, Right, Wait]
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        # DYNAMIC SHAPE HANDLING
        # If x is just 2D (Batch, Features) from Unity, add a fake Sequence dimension (Batch, 1, Features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1) 

        # Pass through Feature Extractor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Pass through Memory Core
        # The LSTM spits out the sequence data (x) and the updated memory states
        x, new_hidden_state = self.lstm(x, hidden_state)

        # Pass through Output Head
        # We slice [:, -1, :] because we only want the network to make a decision based on the LAST step of the sequence
        q_values = self.fc3(x[:, -1, :])

        return q_values, new_hidden_state

    def init_hidden(self, batch_size=1, device="cpu"):
        # The LSTM requires two memory tensors to start: (hidden_state, cell_state)
        # Shape: (Num_Layers, Batch_Size, Hidden_Size)
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))

# Quick Test to ensure it compiles
if __name__ == "__main__":
    model = DRQN()
    print("✅ PyTorch DRQN Initialized Successfully!")
    print(model)
import os
# This stops the Intel CPU math library from crashing on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQN(nn.Module):
    def __init__(self, input_size=6, hidden_size=128):
        super(DRQN, self).__init__()
        
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # LSTM processes sequences: input shape = [batch, seq_len, hidden_size]
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        self.head_move = nn.Linear(hidden_size, 3)
        self.head_turn = nn.Linear(hidden_size, 3)
        self.head_interact = nn.Linear(hidden_size, 2)
        self.head_forks = nn.Linear(hidden_size, 3)

    def forward(self, x, hidden_state=None):
        # x expected shape: [batch, seq_len, input_size]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        
        q_move = self.head_move(lstm_out)
        q_turn = self.head_turn(lstm_out)
        q_interact = self.head_interact(lstm_out)
        q_forks = self.head_forks(lstm_out)
        
        return [q_move, q_turn, q_interact, q_forks], new_hidden_state

    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)

if __name__ == "__main__":
    # 1. Look for the RTX graphics card
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type != 'cuda':
        print("WARNING: PyTorch is not detecting your GPU! It is stuck on the CPU.")

    # 2. Build the brain and push it to the graphics card
    model = DRQN().to(device)
    
    # 3. Create dummy data and push it to the graphics card
    dummy_input = torch.randn(1, 5, 6).to(device) 
    
    q_values, _ = model(dummy_input)
    print("Network compiled successfully on the GPU!")
    print(f"Move Q-Values shape: {q_values[0].shape}")

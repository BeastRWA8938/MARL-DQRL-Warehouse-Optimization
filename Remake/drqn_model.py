import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQN(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_actions=3):
        super(DRQN, self).__init__()
        # Feature extractor
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Recurrent layer for temporal memory
        # batch_first=True expects shape (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        # Q-Value output layer
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x, hidden_state=None):
        # x shape: (batch, seq_len, features)
        x = F.relu(self.fc1(x))
        
        out, hidden_state = self.lstm(x, hidden_state)
        
        # Output Q-values for the final step in the sequence
        q_values = self.fc2(out[:, -1, :]) 
        
        return q_values, hidden_state
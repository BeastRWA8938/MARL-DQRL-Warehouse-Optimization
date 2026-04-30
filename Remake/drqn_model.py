import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQN(nn.Module):
    # CHANGE: input_size is now 8
    def __init__(self, input_size=8, hidden_size=64, num_actions=3): 
        super(DRQN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x, hidden_state=None):
        x = F.relu(self.fc1(x))
        out, hidden_state = self.lstm(x, hidden_state)
        q_values = self.fc2(out[:, -1, :]) 
        
        return q_values, hidden_state
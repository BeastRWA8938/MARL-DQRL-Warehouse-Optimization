import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQN(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, output_size=5): 
        super(DRQN, self).__init__()
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, hidden_size)
        
        # batch_first=True ensures shape is (batch, sequence, features)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        # DYNAMIC SHAPE HANDLING
        if len(x.shape) == 2:
            x = x.unsqueeze(1) 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x, new_hidden_state = self.lstm(x, hidden_state)
        
        # Only extract the output from the final sequence step
        q_values = self.fc3(x[:, -1, :])

        return q_values, new_hidden_state

    def init_hidden(self, batch_size=1, device="cpu"):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))

if __name__ == "__main__":
    model = DRQN()
    print("✅ PyTorch DRQN Initialized Successfully!")
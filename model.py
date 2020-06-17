import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layer1 = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.BatchNorm1d(num_features=fc1_units),
            nn.PReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.BatchNorm1d(num_features=fc2_units),
            nn.PReLU()
        )

        self.layer3 = nn.Linear(fc2_units, action_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fin = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fin)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, 
                 fc1_units=128, fc2_units=128, fc3_units=128, fc4_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)
        
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn4 = nn.BatchNorm1d(fc3_units)
        
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.bn5 = nn.BatchNorm1d(fc4_units)
        self.fc5 = nn.Linear(fc4_units, action_size)
        
        #self.fc3 = nn.Linear(fc2_units, action_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc5.weight.data.uniform_(-1e-3, 1e-3)
        #self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(self.bn1(state)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = F.relu(self.fc3(self.bn3(x)))
        x = F.relu(self.fc4(self.bn4(x)))
        return F.tanh(self.fc5(self.bn5(x)))
        #return F.tanh(self.fc3(self.bn3(x)))
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, 
                 fcs1_units=128, fc2_units=128, fc3_units=128, fc4_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.bn4 = nn.BatchNorm1d(fc4_units)
        self.fc5 = nn.Linear(fc4_units, 1)
        
        #self.fc3 = nn.Linear(fc2_units, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        #self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        # modification for multi-agents
        s = state.view(-1, 48)
        a = action.view(-1, 4)
        
        xs = F.relu(self.fcs1(s))
        x = torch.cat((self.bn(xs), a), dim=1)
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return self.fc5(x)
        
        #return self.fc3(x)
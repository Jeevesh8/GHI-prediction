import torch.nn as nn
import torch

class lstm(nn.Module) :
    def __init__(self, seq_len=256, ini_len=18, final_len=1) :
        super().__init__()
        self.d_model = ini_len 
        self.seq_len = seq_len
        self.hidden_size = 32
        self.num_layers = 1
        self.lstm = nn.LSTM(self.d_model,self.hidden_size,self.num_layers,batch_first=True)
        self.final = nn.Sequential(nn.Linear(self.hidden_size*self.seq_len,512),nn.ReLU(),nn.Linear(512,final_len))
        
    def forward(self,batch) :
        batch, (_b,_a) = self.lstm( batch )
        del _b, _a
        out = self.final(batch.reshape(-1,self.hidden_size*self.seq_len))
        return out

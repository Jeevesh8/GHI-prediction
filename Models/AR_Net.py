import torch.nn as nn
import torch

class ar_nt(nn.Module) :
    def __init__(self, seq_len=256, ini_len = 18, final_len=1) :
        super().__init__()
        self.d_model = 20
        self.seq_len = seq_len
        self.init_trnsfrm = nn.Sequential(nn.Linear(ini_len,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,self.d_model))
        self.batch_norm = nn.BatchNorm1d(self.d_model)
        self.final = nn.Sequential(nn.Linear(self.d_model*self.seq_len,512),nn.ReLU(), nn.Linear(512,256),nn.ReLU(),nn.Linear(256,final_len))
    
    def forward(self,batch) :
        batch = self.init_trnsfrm(batch)
        batch = self.batch_norm(batch.transpose(1,2)).transpose(1,2)
        batch = batch.transpose(0,1)
        t_out = batch.reshape(-1,self.d_model*self.seq_len)
        out = self.final(t_out)
        return out

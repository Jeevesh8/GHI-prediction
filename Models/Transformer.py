import torch.nn as nn
import torch

class trnsfrmr_nt(nn.Module) :
    def __init__(self, seq_len=256, ini_len=18, final_len=1) :
        super().__init__()
        self.d_model = 20
        self.seq_len = seq_len
        self.n_head = 4
        self.dim_feedforward = 2048
        self.init_trnsfrm = nn.Sequential(nn.Linear(ini_len,32),nn.ReLU(),nn.Linear(32,self.d_model))
        self.batch_norm = nn.BatchNorm1d(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model,self.n_head,self.dim_feedforward)
        self.trnsfrmr =  nn.TransformerEncoder(encoder_layer,2)
        self.trnsfrmr2 =  nn.TransformerEncoder(encoder_layer,2)
        self.final = nn.Sequential(nn.Linear(self.d_model*self.seq_len,512),nn.ReLU(), nn.Linear(512,final_len))
    
    def forward(self,batch) :
        batch = self.init_trnsfrm(batch)
        batch = self.batch_norm(batch.transpose(1,2)).transpose(1,2)
        batch = batch.transpose(0,1)
        batch = self.trnsfrmr2(self.trnsfrmr(batch))
        t_out = batch.transpose(0,1).reshape(-1,self.d_model*self.seq_len)
        out = self.final(t_out)
        return out
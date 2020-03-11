import torch.nn as nn
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class cnn_lstm(nn.Module) :
    def __init__(self,seq_len=128, ini_len=18, final_len=1) :
        super().__init__()
        self.d_model = 20 
        self.seq_len = seq_len
        self.hidden_size = 32
        self.num_layers = 1
        self.init_trnsfrm = nn.Sequential(nn.Linear(ini_len,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,self.d_model))
        self.batch_norm = nn.BatchNorm1d(self.d_model)
        self.cnn = nn.Sequential(nn.Conv1d(self.d_model, 32,4,1),
                                 nn.Conv1d(32,16,4,1),
                                 nn.Conv1d(16,16,4,1),
                                 nn.MaxPool1d(2,2),
                                 nn.Conv1d(16,self.d_model,4,1))
        self.lstm = nn.LSTM(self.d_model,self.hidden_size,self.num_layers,batch_first=True)
        self.final = nn.Sequential(nn.Linear(self.hidden_size*56,512),nn.ReLU(),nn.Linear(512,final_len))
    
    def forward(self,batch) :
        batch = self.cnn(self.batch_norm(self.init_trnsfrm(batch).transpose(1,2))).transpose(1,2)
        batch,(h_n,c_n) = self.lstm(batch)
        del h_n,c_n
        out = self.final(batch.reshape(-1,self.hidden_size*56))
        return out

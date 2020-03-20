import pandas as pd
from torch.utils.data import DataLoader
import multiprocessing as mp
import argparse
from DataSet import Dataset
import torch
import torch.nn as nn
from os import path
import Infer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--seq_len', type=int, default=256)
parser.add_argument('--root_dir')
parser.add_argument('--tr_start_year', type=int, help='Training Start year')
parser.add_argument('--tr_final_year', type=int, help='Training Final year')
parser.add_argument('--val_start_year', type=int, help='Validation Start year')
parser.add_argument('--val_final_year', type=int, help='Validation Final year')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--loss', default='mse', help='Choose from qr_loss,mse')
parser.add_argument('--gamma_list', nargs='*', type=float, help='All gammas to be predicted by 1 model')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--model', default='ar_net', help='Choose From ar_net, trfrmr, cnn_lstm, lstm')
parser.add_argument('--ini_len', type=int, default=18, help='Number of Columns in Data<i>.csv')
parser.add_argument('--final_len', type=int, default=1, help='Number of numbers your model will predict.')
parser.add_argument('--steps', type=int, default=1, help='How many step ahead do you want to predict?')
parser.add_argument('--optimizer', default='Adam', help='Choose from Adam and RAdam.')
parser.add_argument('--param_file', help='Path to file to store weights.May not exist.')
args = parser.parse_args()

b_sz = args.batch_size
n_wrkrs = mp.cpu_count()
seq_len = args.seq_len
epochs = args.epochs
 
tr_csv_paths = [args.root_dir+'/Data'+str(i)+'.csv' for i in range(args.tr_start_year, args.tr_final_year+1)]
val_csv_paths = [args.root_dir+'/Data'+str(i)+'.csv' for i in range(args.val_start_year, args.val_final_year+1)]

if args.gamma_list is not None and len(args.gamma_list)>1 and len(args.gamma_list)%2!=0 and args.loss=='qr_loss':
    print('Invalid gamma list')
    exit(0)

dataset_final_len = args.final_len  #if args.loss!='qr_loss' else 1    #or len(args.gamma_list)<=1 else int(args.final_len/2) 
model_final_len = args.final_len*len(args.gamma_list) if args.gamma_list!=None else args.final_len

train_dataset = Dataset.SRdata(tr_csv_paths, seq_len, steps=args.steps, final_len=dataset_final_len)
train_data_loader = DataLoader(train_dataset, batch_size = b_sz, num_workers=n_wrkrs, drop_last = True)

test_dataset = Dataset.SRdata(val_csv_paths, seq_len, steps=args.steps, final_len=dataset_final_len)
test_data_loader = DataLoader(test_dataset, batch_size = b_sz, num_workers=n_wrkrs, drop_last=True)


if args.loss=='mse' :
    lossfn = nn.MSELoss().to(device)

elif args.loss=='qr_loss' :
    maximum  = nn.ReLU()
    gamma_list_len = len(args.gamma_list)
    gammas = torch.tensor(args.gamma_list, dtype=torch.float64, device=device)
    gammas = gammas.repeat_interleave(args.final_len)
    def qr_loss(tar, pred) :
        if gamma_list_len!=1 :
            tar = torch.cat([tar]*gamma_list_len,dim=1)
        n = tar.shape[0]
        m = tar.shape[1]
        loss = (1-gammas)*maximum(tar-pred)+(gammas)*maximum(pred-tar)
        return loss.sum()/(n*m)
    lossfn = qr_loss


if args.model=='ar_net' :
    from Models import AR_Net
    t = AR_Net.ar_nt(seq_len = seq_len, ini_len=args.ini_len, final_len=model_final_len).to(device)
    
elif args.model=='cnn_lstm' :
    from Models import CNN_LSTM
    t = CNN_LSTM.cnn_lstm(seq_len = seq_len, ini_len=args.ini_len, final_len=model_final_len).to(device)

elif args.model=='trfrmr' :
    from Models import Transformer
    t = Transformer.trnsfrmr_nt(seq_len = seq_len, ini_len=args.ini_len, final_len=model_final_len).to(device)

elif args.model=='lstm' :
    from Models import LSTM
    t = LSTM.lstm(seq_len = seq_len, ini_len=args.ini_len, final_len=model_final_len).to(device)

if path.exists(args.param_file) :
    t.load_state_dict(torch.load(args.param_file))

if args.optimizer == 'RAdam' :
    from optimizers import RAdam
    optimizer = RAdam.RAdam(t.parameters(),lr=args.lr)
elif args.optimizer == 'Adam' :
    optimizer = torch.optim.Adam(t.parameters(),lr=args.lr)

t = t.double()
train_mse = []
test_mse = [10000]

for ij in range(epochs) :
    loss_list = []
    for i, batch in enumerate(train_data_loader) :
        optimizer.zero_grad()
        in_batch = batch['in'].to(device)
        out = t(in_batch)
        loss = lossfn(batch['out'].to(device), out)
        loss_list.append(loss)
        loss.backward()
        optimizer.step()
    print('Avg. Training Loss in '+str(ij)+ 'th epoch :- ', sum(loss_list)/len(loss_list))
    train_mse.append(sum(loss_list)/len(loss_list))
    loss_list=[]
    test_mse.append(Infer.evaluate(t, loss = args.loss, test_dataset=test_dataset, args_from_train=args))
    if test_mse[-1]==min(test_mse) :
        print('saving:- ', test_mse[-1])
        torch.save(t.state_dict(),args.param_file)
    

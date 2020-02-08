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
parser.add_argument('--tr_start_year', type=int)
parser.add_argument('--tr_final_year', type=int)
parser.add_argument('--val_start_year', type=int)
parser.add_argument('--val_final_year', type=int)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--loss', default='mse', help='Choose from qr_loss,mse')
parser.add_argument('--gamma', type=float, default=0.90, help='gamma for qr_loss')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--model', default='ar_net', help='Choose From ar_net, trfrmr, cnn_lstm')
parser.add_argument('--ini_len', type=int, default=18, help='Number of Columns in Data<i>.csv')
parser.add_argument('--steps', type=int, default=1, help='How many step ahead do you want to predict?')
parser.add_argument('--optimizer', default='Adam'. help='Choose from Adam and RAdam.')
parser.add_argument('--param_file', help='Path to file to store weights.May not exist.')
args = parser.parse_args()

b_sz = args.batch_size
n_wrkrs = mp.cpu_count()
seq_len = args.seq_len
epochs = args.epochs
 
tr_csv_paths = [args.root_dir+'/Data'+str(i) for i in range(args.tr_start_year, args.tr_final_year+1)]
val_csv_paths = [args.root_dir+'/Data'+str(i) for i in range(args.val_start_year, args.val_final_year+1)]

train_dataset = Dataset.SRData(tr_csv_paths, seq_len)
train_data_loader = DataLoader(train_dataset, batch_size = b_sz, num_workers=n_wrkrs, drop_last = True, steps=args.steps)

test_dataset = Dataset.SRData(val_csv_paths, seq_len)
test_data_loader = DataLoader(test_dataset, batch_size = b_sz, num_workers=n_wrkrs, drop_last=True, steps=args.steps)



if args.loss=='mse' :
    lossfn = nn.MSELoss().to(device)
if args.loss=='qr_loss'
    gamma = args.gamma
    maximum  = nn.ReLU()
    def qr_loss(tar, pred) :
        n = tar.shape[0]
        loss = (1-gamma)*maximum(tar-pred)+(gamma)*maximum(pred-tar)
        return loss.sum()/n
    lossfn = qr_loss


if args.model=='ar_net' :
    from Models import AR_Net
    t = AR_Net.ar_nt(seq_len = seq_len, ini_len=args.ini_len).to(device)
    if path.exists(args.param_file) :
        t.load_state_dict(torch.load(args.param_file))
elif args.model=='cnn_lstm' :
    from Models import CNN_LSTM
    t = CNN_LSTM.cnn_lstm(seq_len = seq_len, batch_size = b_sz, ini_len=args.ini_len).to(device)
    if path.exists(args.param_file) :
        t.load_state_dict(torch.load(args.param_file))
elif args.model=='trfrmr' :
    from Models import Transformer
    t = Transformer.trnsfrmr_nt(seq_len = seq_len, ini_len=args.ini_len).to(device)
    if path.exists(args.param_file) :
        t.load_state_dict(torch.load(args.param_file))

if args.optimizer == 'RAdam' :
    from optimizers import RAdam
    optimizer = RAdam.RAdam(t.parameters(),lr=args.lr)
if args.optimizer == 'Adam' :
    optimizer = torch.optim.Adam(t.parameters(),lr=args.lr)

train_rmse = []
test_rmse = [10000]

for i in range(epochs) :
    loss_list = []
    for i, batch in enumerate(train_data_loader) :
        optimizer.zero_grad()
        in_batch = batch['in'].to(device)
        out = t(in_batch)
        loss = lossfn(out.reshape(b_sz),batch['out'].to(device))
        loss_list.append(loss)
        loss.backward()
        optimizer.step()
    print('Avg. Training Loss This epoch :- ', sum(loss_list)/len(loss_list))
    train_rmse.append(sum(loss_list)/len(loss_list))
    loss_list=[]
    t.cpu()
    test_rmse.append(Infer.evaluate(t,steps=1, test_dataset=test_dataset))
    t.to(device)
    if test_rmse[-1]==min(test_rmse) :
        print('saving:- ', test_rmse[-1])
        torch.save(t.state_dict(),args.param_file)
    
import torch.nn as nn
import torch
import argparse
from torch.utils.data import DataLoader
import multiprocessing as mp
from os import path

n_wrkrs = mp.cpu_count()
abs_loss_fn = nn.L1Loss(reduction='none')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def define_variables(args_from_train=None) :
    
    global mask_gammas, maximum, gamma_list_len, gammas, real_vals_sum, pred_loss_sum      
    
    if args_from_train is not None :
        args=args_from_train
    
    maximum  = nn.ReLU()
    gamma_list_len = max(1,len(args.gamma_list))
    
    if hasattr(args,'mask_gamma_list') and args.mask_gamma_list is not None :
        mask_gammas = torch.tensor(args.mask_gamma_list, device=device, dtype=torch.float64)
        print(mask_gammas)
    else :
        mask_gammas = torch.ones(gamma_list_len, device=device, dtype=torch.float64)
    
    mask_gammas = mask_gammas.repeat_interleave(args.final_len)
    gammas = torch.tensor(args.gamma_list, dtype=torch.float64, device=device)
    gammas = gammas.repeat_interleave(args.final_len)
        
    real_vals_sum = 0 #For q-risk
    pred_loss_sum = 0 #For q-risk
        

def mape_loss(pred,real) :
    return torch.div(abs_loss_fn(pred,real),torch.abs(real))

def interval_loss(pred, tar) :
    
    global real_vals_sum, pred_loss_sum
    
    if gamma_list_len!=1 :
        tar = torch.cat([tar]*gamma_list_len,dim=1)
        tar = mask_gammas*tar
        pred = mask_gammas*pred
        real_vals_sum += torch.abs(tar).sum().item()
    
    n = tar.shape[0]
    m = mask_gammas.sum() #tar.shape[1] #/gamma_list_len
    
    if lossfn_i == 'qr_loss' :
        loss = (1-gammas)*maximum(tar-pred)+(gammas)*maximum(pred-tar)
    else :
        loss = lossfn_i(tar, pred)
    
    pred_loss_sum += loss.sum().item()
    return loss.sum()/(n*m)
    
def run_to_eval(t, lossfn, give_lists=False, test_dataset=None, times_to_run_model=0, batch_size=1) :
    loss_list = []
    i = 0
    tot_loss = 0
    t.eval()
    test_data_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=n_wrkrs, drop_last=True)
    it = iter(test_data_loader)
    
    if give_lists :
        pred_lis = []
        actual_lis = []
        time_lis = []
    
    for batch in it :
        
        in_batch = batch['in'].to(device)
        out = t(in_batch)
            
        if give_lists :
            pred_lis.append(out.tolist())
            actual_lis.append(batch['out'].tolist())
            time_lis.append(in_batch[0][-1][0:5].int().tolist())
        else :
            loss = lossfn(out,batch['out'].to(device))
            tot_loss += loss.item()
        i+=1
        if i>times_to_run_model and give_lists :
            print(pred_lis)
            print(actual_lis)
            print(time_lis)
            break
    
    print('Evaluation Loss:- ', tot_loss/i)
    t.train()
    return tot_loss/i

def mae_loss(x,y) :
    return torch.abs(x-y)

def diff(x,y) :
    return x-y

def evaluate(t, loss = 'mse', test_dataset=None, args_from_train=None) :
    t.eval()
    define_variables(args_from_train)
    lossfn = interval_loss
    global lossfn_i
    if loss == 'mse' :
        lossfn_i = nn.MSELoss(reduction='none')
    elif loss == 'mape' :
        lossfn_i = mape_loss
    elif loss == 'mae' :
        lossfn_i = abs_loss_fn
    elif loss == 'mbe' :
        lossfn_i = diff
    elif loss == 'qr_loss' :
        lossfn_i = 'qr_loss'
    else :
        lossfn_i = nn.MSELoss(reduction='none')
    return run_to_eval(t, lossfn, test_dataset=test_dataset)


def predict_next(t, date_lis, test_dataset) :
    batch = test_dataset.getitem_by_date(date_lis)
    in_batch = batch['in'].to(device).unsqueeze(dim=0)
    out = t(in_batch)
    if 'out' in batch :
        print('Real output :-', batch['out'].tolist())
    print('Predicted Output :-', out)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='avg_loss', help='Choose from avg_loss, predict_list, predict_at_date')
    parser.add_argument('--loss', default='mse', help='Choose from mse, mbe, mae, mape, qr_loss')
    parser.add_argument('--model', default='ar_net', help='Choose from ar_net, trfrmr, cnn_lstm, lstm')
    parser.add_argument('--ini_len', type=int, help='Number of columns of input data')
    parser.add_argument('--param_file',help='Path to model\'s param file')
    parser.add_argument('--batch_size', type=int, default=1, help='To be used in avg_loss mode only.')

    parser.add_argument('--date_lis', nargs='*', type=int, help='List of form [Year, Month, Day, Hour, Minute]')
    
    parser.add_argument('--steps', type=int, default=1, help='Number of steps-ahead model was trained to predict')
    parser.add_argument('--final_len', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=256)

    parser.add_argument('--root_dir',help='Directory where Data*.csv files are located.')
    parser.add_argument('--test_start_year', type=int, help='Starting test year. Use only when mode is avg_loss')
    parser.add_argument('--test_final_year', type=int, help='Final test year. Use only when mode is avg_loss.')
    parser.add_argument('--test_year', type=int, default=-1, help='test data year.')
    
    parser.add_argument('--times_to_run' , type=int, default=200, help='Times to run the model when mode is predict_list')
    
    parser.add_argument('--gamma_list', type=float, nargs='*', help='Gammas for calculating q-risk')
    parser.add_argument('--mask_gamma_list', type=int, nargs='*', help='Masks for Gamma values, e.g. use :- to calculate only p50 or p90 risk')
    
    args = parser.parse_args()
    
    from DataSet import Dataset 
    if args.test_year != -1 :
        csv_paths = [args.root_dir+'/Data'+str(args.test_year)+'.csv']
    else :
        csv_paths = [args.root_dir+'/Data'+str(i)+'.csv' for i in range(args.test_start_year, args.test_final_year+1)]
    
    model_final_len = args.final_len*len(args.gamma_list) if args.gamma_list!=None else args.final_len
    dataset_final_len = args.final_len #if not args.interval or args.final_len<=1 else int(args.final_len/2) 
    test_dataset = Dataset.SRdata(csv_paths, seq_len = args.seq_len, steps = args.steps, final_len=dataset_final_len)

    
    if args.model=='ar_net' :
        from Models import AR_Net
        t = AR_Net.ar_nt(seq_len = args.seq_len, ini_len=args.ini_len, final_len=model_final_len).to(device)
        
    elif args.model=='cnn_lstm' :
        from Models import CNN_LSTM
        t = CNN_LSTM.cnn_lstm(seq_len = args.seq_len, ini_len=args.ini_len, final_len=model_final_len).to(device)
        
    elif args.model=='trfrmr' :
        from Models import Transformer
        t = Transformer.trnsfrmr_nt(seq_len = args.seq_len, ini_len=args.ini_len, final_len=model_final_len).to(device)
    
    elif args.model=='LSTM' :
        from Models import LSTM
        t = LSTM.lstm(seq_len = args.seq_len, ini_len=args.ini_len, final_len=model_final_len).to(device)
    
    t.load_state_dict(torch.load(args.param_file))

    t = t.double()
    
    if args.mode=='avg_loss' :
        print(evaluate(t,args.loss, test_dataset, args))
    
    elif args.mode=='predict_list' :
        print(run_to_eval(t, None, True, test_dataset, args.times_to_run))
    
    elif args.mode == 'predict_next' :
        print(predict_next(t,args.date_lis,test_dataset))

    if args.loss=='qr_loss' :
        print('Q-risk = ', 2*pred_loss_sum/real_vals_sum)


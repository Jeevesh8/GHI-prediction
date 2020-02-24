import torch.nn as nn
import torch
import argparse

abs_loss_fn = nn.L1Loss() #.to(device)

def mape_loss(pred,real) :
    return torch.sum(torch.div(abs_loss_fn(pred,real),torch.abs(real)))/b_sz

def run_to_eval(t, lossfn, give_lists=False, test_dataset=None, times_to_run_model=0) :
    loss_list = []
    i = 0
    tot_loss = 0
    
    test_data_loader = DataLoader(test_dataset, batch_size = 1 , num_workers=n_wrkrs, drop_last=True)
    it = iter(test_data_loader)
    
    if give_lists :
        pred_lis = []
        actual_lis = []
        time_lis = []
    
    for batch in it :
        
        in_batch = batch['in'] #.to(device)
        out = t(in_batch)
            
        if give_lists :
            pred_lis.append(out.tolist())
            actual_lis.append(batch['out'].tolist())
            time_lis.append(in_batch[0][-1][0:5].int().tolist())
        
        loss = lossfn(out.reshape(-1),batch['out'])
        tot_loss += loss.item()
        i+=1
        if i>times_to_run_model and give_lists :
            print(pred_lis)
            print(actual_lis)
            print(time_lis)
            break
    
    print('Evaluation Loss:- ', tot_loss/i)
    return tot_loss/i

def mae_loss(x,y) :
    return torch.abs(x-y)

def diff(x,y) :
    return x-y

def evaluate(t, loss = 'rmse', test_dataset=None) :
    t.eval()
    if loss == 'rmse' :
        lossfn = nn.MSELoss()#.to(device)
    elif loss == 'mape' :
        lossfn = mape_loss
    elif loss == 'mae' :
        lossfn = abs_loss_fn
    elif loss == 'mbe' :
        lossfn = diff
    else :
        lossfn = nn.MSELoss()
    return run_to_eval(t, lossfn, test_dataset=test_dataset)


def predict_next(t, date_lis, test_dataset) :
    batch = test_dataset.getitem_by_date(date_lis)
    in_batch = batch['in']
    out = t(in_batch)
    if 'out' in batch and not args.interval:
        print('Real output :-', batch[out].tolist())
    print('Predicted Output :-', out)

if __name__=='main':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='avg_loss', help='Choose from avg_loss, predict_list, predict_at_date')
    parser.add_argument('--loss', default='rmse', help='Choose from rmse, mbe, mae, mape')
    parser.add_argument('--model', default='ar_net', help='Choose from ar_net, trfrmr, cnn_lstm')
    parser.add_argument('--ini_len', type=int, help='Number of columns of input data')
    parser.add_argument('--param_file',help='Path to model\'s param file')

    parser.add_argument('--interval', type=bool, default=False, help='set true if model predicts interval')
    
    parser.add_argument('--date_lis', nargs='*', help='List of form [Year, Month, Day, Hour, Minute]')
    
    parser.add_argument('--steps', type=int, default=1, help='Number of steps-ahead model was trained to predict')
    parser.add_argument('--final_len', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=256)

    parser.add_argument('--root_dir',help='Directory where Data*.csv files are located.')
    parser.add_argument('--test_start_year', type=int, help='Starting test year. Use only when mode is avg_loss')
    parser.add_argument('--test_final_year', type=int, help='Final test year. Use only when mode is avg_loss.')
    parser.add_argument('--test_year', type=int, default=-1, help='test data year.')
    
    parser.add_argument('--times_to_run' , type=int, default=200, help='Times to run the model when mode is predict_list')
    args = parser.parse_args()
    
    from DataSet import Dataset 
    if args.test_year != -1 :
        csv_paths = [args.root_dir+'/Data'+str(args.test_year)+'.csv']
    else :
        csv_paths = [args.root_dir+'/Data'+str(i)+'.csv' for i in range(args.test_start_year, args.test_final_year+1)]
    
    dataset_final_len = args.final_len if not args.interval or args.final_len<=1 else int(args.final_len/2) 
    test_dataset = Dataset.SRdata(csv_paths, seq_len = args.seq_len, steps = args.steps, final_len=dataset_final_len)

    
    if args.model=='ar_net' :
        from Models import AR_Net
        t = AR_Net.ar_nt(seq_len = args.seq_len, ini_len=args.ini_len, final_len=args.final_len).to(device)
        if path.exists(args.param_file) :
            t.load_state_dict(torch.load(args.param_file))
    
    elif args.model=='cnn_lstm' :
        from Models import CNN_LSTM
        t = CNN_LSTM.cnn_lstm(seq_len = args.seq_len, ini_len=args.ini_len, final_len=args.final_len).to(device)
        if path.exists(args.param_file) :
            t.load_state_dict(torch.load(args.param_file))
    
    elif args.model=='trfrmr' :
        from Models import Transformer
        t = Transformer.trnsfrmr_nt(seq_len = args.seq_len, ini_len=args.ini_len, final_len=args.final_len).to(device)
        if path.exists(args.param_file) :
            t.load_state_dict(torch.load(args.param_file))

    t = t.double()
    
    if args.mode=='avg_loss' :
        print(evaluate(t,args.loss,test_dataset))
    
    elif args.mode=='predict_list' :
        print(run_to_eval(t, args.loss, True, test_dataset, args.times_to_run))
    
    elif args.mode == 'predict_next' :
        for i in range(len(date_lis)) :
            date_lis[i] = int(date_lis[i])
        print(predict_next(t,args.date_lis,test_dataset))

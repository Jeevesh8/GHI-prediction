import torch.nn as nn
import torch

abs_loss_fn = nn.L1Loss().to(device)

def mape_loss(pred,real) :
    return torch.sum(torch.div(abs_loss_fn(pred,real),torch.abs(real)))/b_sz

def run_to_eval(t, lossfn, steps=1, give_lists=False, test_dataset=None) :
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
        j = 0
        
        if give_list :
            cur_lis = []
            actual_lis_cur = []
            cur_time_lis = []
        
        in_batch = batch['in'] #.to(device)
        while j<steps :
            out = t(in_batch)
            if give_lists :
                cur_time_lis.append(in_batch[0][-1][0:5].int().tolist())
                actual_lis_cur.append(batch['out'].item())
                cur_lis.append(out.item())
            if j!=steps-1 :
                try :
                    batch = next(it)
                except :
                    print('Evaluation Loss:- ', tot_loss/i)
                    return tot_loss/i
                in_batch = batch['in']
                in_batch[0][-1][-1] = out.item()
            j+=1
        
        if give_lists :
            pred_lis.append(cur_lis)
            actual_lis.append(actual_lis_cur)
            time_lis.append(cur_time_lis)
        
        loss = lossfn(out.reshape(-1),batch['out']) #.to(device))
        tot_loss += loss.item()
        i+=1
        if i>200 and give_lists :
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

def evaluate(t, loss = 'rmse', steps=24, test_dataset=None) :
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
    return run_to_eval(t, lossfn, steps, test_dataset)


'''
t = trnsfrmr_nt(ini_len=15).double().to(device)
t.eval() 
t.load_state_dict(torch.load('/content/drive/My Drive/SolarDataIndia/SolarData(In)/TransformerFast256OnlyWeather(In).param',map_location=cpu))
loss_types = [0] #['mae','mbe','rmse']
steps = [1,12]
losses = []
for loss in loss_types :
    one_type_losses = []
    for step in steps :
        one_type_losses.append(evaluate(t, loss=loss, steps=step, ))
    losses.append(one_type_losses)
#print(losses)
'''
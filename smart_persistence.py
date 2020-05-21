import argparse
import numpy as np
from scipy.interpolate import griddata
import pandas as pd

def date_to_nth_day(year, month, day):
    date = pd.Timestamp(year=year,month=month,day=day)
    new_year_day = pd.Timestamp(year=year, month=1, day=1)
    return (date - new_year_day).days + 1

def get_df(csv_paths) :
    df_lis = []
    for path in csv_paths :
        df_lis.append(pd.read_csv(path))
    final_df = pd.concat(df_lis,ignore_index=True).drop(['Unnamed: 0'],axis=1)
    return final_df

def day_passed_ratio(hour, minute) :
    return (hour*60+minute)/24*60

def caller(series) :
    series['nthDay'] = int(date_to_nth_day(series['Year'], series['Month'], series['Day']))
    series['diff_hours'] = day_passed_ratio(series['Hour'], series['Minute'])
    return series

def lossfn(a, b, loss='mse') :
     if loss == 'mse' :
        return (a-b)*(a-b)
    elif loss == 'mape' :
        return np.abs(a-b)/np.abs(b)
    elif loss == 'mae' :
        return np.abs(a-b)
    elif loss == 'mbe' :
        return a-b

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', default='mse', help='Choose from mse, mbe, mae, mape')
    parser.add_argument('--test_start_year', type=int, help='Starting test year. Use only when mode is avg_loss')
    parser.add_argument('--test_final_year', type=int, help='Final test year. Use only when mode is avg_loss.')
    parser.add_argument('--tr_start_year', type=int, help='Training Start year')
    parser.add_argument('--tr_final_year', type=int, help='Training Final year')
    parser.add_argument('--root_dir')
    parser.add_argument('--steps', type=int, default=1, help='How many values do you want to skip b/w 2 consecutive predictions?')
    parser.add_argument('--get_preds', action='store_true', help='Set this flag if you want to get predictions of Smart Persistence')
    
    csv_paths=[root_dir+'Data'+str(i)+'.csv' for i in range(tr_start_year, tr_end_year+1)]
    final_df = get_df(csv_paths)
    csv_paths=[root_dir+'Data'+str(i)+'.csv' for i in range(val_start_year, val_end_year+1)]
    val_final_df = get_df(csv_paths)

    final_df['nthDay'] = np.nan
    final_df['diff_hours'] = np.nan
    final_df = final_df.apply(caller, axis=1)

    val_final_df['nthDay'] = np.nan
    val_final_df['diff_hours'] = np.nan
    val_final_df = final_df.apply(caller, axis=1)

    final_df = final_df[['GHI', 'nthDay', 'diff_hours']]
    
    values = final_df.groupby(['nthDay','diff_hours']).mean()
    values = values.reset_index()

    points = values[['nthDay', 'diff_hours']].to_numpy()
    ghi_values = values[['GHI']].to_numpy()

    points_to_interpolate_to = val_final_df[['nthDay', 'diff_hours']][::args.steps].to_numpy()
    real_ghi_vals = val_final_df[['GHI']][::args.steps].to_numpy()

    clear_sky_preds = griddata(points, ghi_values, points_to_interpolate_to, method='linear')
    
    smart_persistence_preds = []
    loss = 0
    for i in range(len(real_ghi_vals)) :
        if i==0 :
            continue
        if clear_sky_preds[i-1]==0 :
            smart_persistence_preds.append(clear_sky_preds[i])
        else :
            smart_persistence_preds.append( (clear_sky_preds[i]*real_ghi_vals[i-1])/clear_sky_preds[i-1])
        loss += lossfn(smart_persistence_preds[i-1], real_ghi_vals[i], args.loss)

    if arge.get_preds :
        print(smart_persistence_preds)
    
    print("Loss=", loss)    
import pickle
import numpy as np
import argparse

def c_to_i(a,b) :
    if a==5 and b==30 : return 0
    if b==30 : return 2*(a-6)+2
    else : return 2*(a-6)+1  

def date_to_nth_day(year, month, day):
    date = pd.Timestamp(year=year,month=month,day=day)
    new_year_day = pd.Timestamp(year=year, month=1, day=1)
    return (date - new_year_day).days + 1


def calc_means(tar_dir, dat, start_year=2000, final_year=2014) :
    years=[i for i in range(start_year,final_year+1)]
    morning_dat = dat.loc[(dat['Hour']<20)&(dat['Hour']>5)|((dat['Hour']==5)&(dat['Minute']==30))].copy().reset_index(drop=True)
    morning_dat['nthDay'] = np.nan
    
    for i in range(len(morning_dat)) :
        morning_dat.loc[i,'nthDay'] = int(date_to_nth_day(morning_dat.loc[i,'Year'].item(),morning_dat.loc[i,'Month'].item(),
                                                          morning_dat.loc[i,'Day'].item()))
    
    yearly_avg = morning_dat['GHI'].mean(axis=0)
    morning_dat['year_residual'] = pd.Series(morning_dat['GHI']-yearly_avg)
    morning_dat.drop(morning_dat.tail(1).index,inplace=True) 
    monthly_residual_avg = [morning_dat.loc[morning_dat['Month']==i+1,'year_residual'].mean(axis=0) for i in range(12)]
    morning_dat['month_residual'] = np.nan
    
    for index in range(len(morning_dat)) :
        morning_dat.loc[index,'month_residual'] = morning_dat.loc[index,'year_residual'] - monthly_residual_avg[int(morning_dat.loc[index,'Month'].item())-1]
    
    daily_residual_avg = [morning_dat.loc[morning_dat['nthDay']==i+1,'month_residual'].mean(axis=0) for i in range(366)]
    
    morning_dat['day_residual'] = np.nan
    
    for index in range(len(morning_dat)) :
        morning_dat.loc[index,'day_residual'] = morning_dat.loc[index,'month_residual'] -\
        daily_residual_avg[int(date_to_nth_day(morning_dat.loc[index,'Year'].item(),
                                               morning_dat.loc[index,'Month'].item(),
                                               morning_dat.loc[index,'Day'].item())) -1 ]
    times = [[5,30]]
    i=6
    time_avgs = [[None]*29]*366
    while(i<20) :
        times.append([i,0])
        times.append([i,30])
        i+=1
    
    for x in range(len(times)) :
        for i in range(366) :
            time_avgs[i][x] = morning_dat.loc[(morning_dat['nthDay']==i+1)&(morning_dat['Hour']==times[x][0])&(morning_dat['Minute']==times[x][1]),'day_residual'].mean()    
    
    with open(tar_dir+'AllMean(In).pickle','wb+') as f :
        pickle.dump(yearly_avg,f)  
    with open(tar_dir+'monthly_residual_means(In).pickle','wb+') as f :
        pickle.dump(monthly_residual_avg,f)  
    with open(tar_dir+'daily_residual_means(In).pickle','wb+') as f :
        pickle.dump(daily_residual_avg, f)
    with open(tar_dir+'time_residual_means(In).pickle','wb+') as f :
        pickle.dump(time_avgs, f)


def generate_data_files(csv_file_path_prefix, tar_dir, start_year, final_year, t_start_year, t_final_year) :
    
    data = [pd.read_csv(csv_file_path_prefix+str(year)+'.csv',skiprows=[0,1]) for year in range(start_year, final_year+1)]
    data = pd.concat(data, axis = 0)
    calc_means(pat, data, start_year, final_year)
    pat = tar_dir                
    
    with open(pat+'AllMean(In).pickle','rb') as f :
        yearly_avg = pickle.load(f)  
    with open(pat+'monthly_residual_means(In).pickle','rb') as f :
        monthly_avg = pickle.load(f)  
    with open(pat+'daily_residual_means(In).pickle','rb') as f :
        daily_avg = pickle.load(f)
    with open(pat+'time_residual_means(In).pickle','rb') as f :
        time_avgs = pickle.load(f)
    
    all_year_lis = []
    for year in range(start_year, final_year+1) :
        all_year_lis.append(year)
    for year in range(t_start_year, t_final_year+1) :
        all_year_lis.append(year)      
    for year in all_year_lis :
        path = csv_file_path_prefix+str(year)+'.csv'
        df = pd.read_csv(path, skiprows=[0,1])
        dat = df
        dat['isNight'] = ~(((dat['Hour']<20)&(dat['Hour']>5))|((dat['Hour']==5)&(dat['Minute']==30)))
        x = pd.DataFrame(index = dat.index, columns = ['year_residual','month_residual','day_residual','ResidueForlstm'])
        x[['Year','Month','Day','Hour','Minute','GHI','isNight']] = dat[['Year','Month','Day','Hour','Minute','GHI','isNight']]
        x['year_residual'] = pd.Series(x['GHI']-yearly_avg)
        x.drop(x.tail(1).index,inplace=True)   

        for index in range(len(x)) :
            x.loc[index,'month_residual'] = x.loc[index,'year_residual'] - monthly_avg[int(x.loc[index,'Month'].item())-1]

        for index in range(len(x)) :
            x.loc[index,'day_residual'] = x.loc[index,'month_residual'] - daily_avg[int(date_to_nth_day(x.loc[index,'Year'].item(),x.loc[index,'Month'].item(),x.loc[index,'Day'].item())) -1 ]
        
        for index in range(len(x)) :
            if not x.loc[index,'isNight'] :
                x.loc[index, 'ResidueFORlstm'] = -time_avgs[ int(date_to_nth_day(x.loc[index,'Year'].item(),
                                                                                x.loc[index,'Month'].item(),
                                                                                x.loc[index,'Day'].item()))-1 ][ int(c_to_i(x.loc[index,'Hour'],x.loc[index,'Minute'])) ] + x.loc[index,'day_residual']
        
        x = x[['year_residual','month_residual','day_residual','ResidueFORlstm']]
        input_df = pd.concat([dat,x],axis=1,sort=False)
        input_df = input_df[1:]
        input_df = input_df[:-1]
        input_df = input_df[input_df['isNight']==False]
        test_df = input_df.reset_index(drop=True)
        test_df.to_csv(tar_dir+'Data_Files/Data'+str(year)+'.csv')

def __main__() :
    parser = argparse.ArgumentParser(description = 'Prepare the Dataset')
    parser.add_argument('--csv_prefix')
    parser.add_argument('--tar_dir')
    parser.add_argument('--tr_start_year', type='int')
    parser.add_argument('--tr_final_year', type='int')
    parser.add_argument('--t_start_year', type='int')
    parser.add_argument('-t_final_year', type='int')
    args=parser.parse_args()
    generate_data_files(args.csv_prefix, arg.tar_dir, args.tr_start_year, args.tr_final_year, args.t_start_year, args.t_final_year)

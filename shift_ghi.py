import pickle
import pandas as pd
import argparse

def reform(lis, level=0) :
    if type(lis[0])!=list or (level==1 and type(lis[0][0])!=list) :
        return lis
    new_lis = []
    for elem in lis :
        x=reform(elem)
        if type(x)==list :
            new_lis+=x
        else :
            new_lis.append(x)
    return new_lis

def date_to_nth_day(year, month, day):
    date = pd.Timestamp(year=year,month=month,day=day)
    new_year_day = pd.Timestamp(year=year, month=1, day=1)
    return (date - new_year_day).days + 1

def c_to_i(a,b) :
    if a==5 and b==30 : return 0
    if b==30 : return 2*(a-6)+2
    else : return 2*(a-6)+1  

root = '/content/drive/My Drive/SolarData'
all_mean_path = root+'/AllMean.pickle'
daily_residual_mean_path = root+'/daily_residual_means.pickle'
monthly_residual_mean_path  = root+'/monthly_residual_means.pickle'
time_residual_mean_path = root+'/time_residual_means.pickle'

with open(all_mean_path, 'rb') as f:
    all_mean = pickle.load(f)
with open(daily_residual_mean_path, 'rb') as f :
    daily_residual_mean_lis = pickle.load(f)
with open(monthly_residual_mean_path, 'rb') as f :
    monthly_residual_mean_lis = pickle.load(f)
with open(time_residual_mean_path, 'rb') as f :
    time_residual_mean_lis = pickle.load(f)

def shift_ghi(lis, times_lis) :
    lis_to_return = []
    for elem in zip(lis, times_lis) :
        y = elem[0]+all_mean+monthly_residual_mean_lis[elem[1][1]]+\
            daily_residual_mean_lis[date_to_nth_day(elem[1][0],elem[1][1],elem[1][2])]+\
            time_residual_mean_lis[date_to_nth_day(elem[1][0],elem[1][1],elem[1][2])][c_to_i(elem[1][3],elem[1][4])]
        lis_to_return.append(y)
    return lis_to_return

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--ghi_time_file', help='path to file having lists of ghi and times')
    parser.add_argument('--date_lis', nargs='*', type=int)
    parser.add_argument('--ghi',nargs='*',type=float)
    parser.add_argument('--write_to',help='Choose from <append|new_file_path|print>' )
    args = parser.parse_args()
    
    if args.ghi_time_file is not None :
        with open(args.ghi_time_file, 'rb') as f :
            dic = pickle.load(f)
        times_lis = dic['times_lis']
        times_lis = reform(times_lis, level=1)
        for k,v in dic.items() :
            if k!='times_lis' :
                dic[k] = reform(dic[k])
                dic[k] = shift_ghi(dic[k], times_lis)
                print(dic[k])
        if args.write_to == 'append' :
            with open(args.ghi_time_file, 'ab') as f :
                pickle.dump(dic,f)
        elif args.write_to == 'print' :
            print(dic)
        else :
            with open(args.ghi_time_file, 'wb+') as f:
                pickle.dump(dic,f)
                
    else :
        dates = []
        i=0
        while i+5<=len(args.date_lis) :
            dates.append(args.date_lis[i:i+5])
            i+=5
        print(shift_ghi(args.ghi, dates))

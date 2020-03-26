from torch.utils.data import Dataset
import pandas as pd
import torch 

class SRdata(Dataset) :
    def __init__(self, csv_paths, seq_len, transform = None, steps=1, final_len=1) :
        df = self.get_df(csv_paths)
        self.seq_len = seq_len
        self.in_df = df.drop(['isNight','GHI','DHI'],axis=1)
        self.out_df = df['ResidueFORlstm']
        self.steps = steps
        self.final_len = final_len

    def get_df(self, csv_paths) :
        df_lis = []
        for path in csv_paths :
            df_lis.append(pd.read_csv(path))
        return pd.concat(df_lis,ignore_index=True).drop(['Unnamed: 0'],axis=1)
    
    def __len__(self) :
        return len(self.in_df)-self.seq_len-self.steps+1-self.final_len

    def __getitem__(self,idx) :
        start_index = idx+self.seq_len+self.steps-1
        end_index = start_index+self.final_len-1
        return { 'in' : torch.tensor(self.in_df[idx : idx+self.seq_len].values,dtype=torch.float64), 
                'out' : torch.tensor(self.out_df.loc[start_index:end_index].values,dtype=torch.float64) }

    def __getonlyin__(self,idx) :
        return {'in' : torch.tensor(self.in_df[idx : idx+self.seq_len].values,dtype=torch.float64)}
    
    def getitem_by_date(self,date) :
        '''date:- list of form [year, month, day, hour, minute]'''
        index = self.in_df.index[(self.in_df['Year']==date[0])&(self.in_df['Month']==date[1]) \
                    &(self.in_df['Day']==date[2])&(self.in_df['Hour']==date[3]) \
                    &(self.in_df['Minute']==date[4])]
        index = index.tolist()[0]
        if index<=self.seq_len :
            print('Not Enough Information To Predict')
            exit(0)
        if index-self.seq_len>=self.__len__() :
            return self.__getonlyin__(index)
        return self.__getitem__(index-self.seq_len) 

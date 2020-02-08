from torch.utils.data import Dataset

class SRdata(Dataset) :
    def __init__(self, csv_paths, seq_len, transform = None, steps=1) :
        df = self.get_df(csv_paths)
        self.seq_len = seq_len
        self.in_df = df.drop(['isNight','GHI','DHI'],axis=1)
        self.out_df = df['ResidueFORlstm']
        self.steps = steps

    def get_df(self, csv_paths) :
        df_lis = []
        for path in csv_paths :
            df_lis.append(pd.read_csv(path))
        return pd.concat(df_lis,ignore_index=True).drop(['Unnamed: 0'],axis=1)
    
    def __len__(self) :
        return len(self.in_df)-self.seq_len-self.steps+1

    def __getitem__(self,idx) :
        return { 'in' : torch.tensor(self.in_df[idx : idx+self.seq_len].values,dtype=torch.float64), 
                'out' : torch.tensor(self.out_df.loc[idx+self.seq_len+self.steps-1],dtype=torch.float64) } 
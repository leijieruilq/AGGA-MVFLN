import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,configs):
        super(LSTM,self).__init__()
        self.seq_len = configs["seq_len"]
        self.pred_len = configs["pred_len"]
        self.channels = configs["c_out"]
        self.num_nodes = configs["num_nodes"]
        self.lstm = nn.LSTM(input_size=self.seq_len,
                            hidden_size=self.pred_len,
                            num_layers=configs["n_layers"],
                            dropout=configs["dropout"],
                            batch_first=True)
    
    def forward(self,x):
        x = x.reshape(-1,self.channels*self.num_nodes,self.seq_len)
        out,_ = self.lstm(x)
        return out.reshape(-1,self.channels,self.num_nodes,self.pred_len)
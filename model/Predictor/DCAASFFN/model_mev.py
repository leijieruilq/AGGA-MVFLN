import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .block import FreqConv, Indepent_Linear, stattn_layer, gated_mlp

class MEVModel(nn.Module):
    def __init__(self, configs, **args):
        nn.Module.__init__(self)
        self.params(configs)
        self.fconv1 = FreqConv(6, self.inp_len, self.inp_len, kernel_size=self.kernel_size, dilation=self.dilation, order=self.order)
        self.fconv2 = FreqConv(6, self.pred_len, self.pred_len, kernel_size=self.kernel_size, dilation=self.dilation, order=self.order)
        self.fc_idp = Indepent_Linear(self.inp_len, self.pred_len, self.channels, configs["share"], self.dp_rate)

        self.stattn_in = nn.ModuleList([stattn_layer(num_of_vertices=self.n_nodes, 
                               num_of_features=self.channels, 
                               num_of_timesteps=self.inp_len,
                               device = configs['device'],
                               emb_dim=self.emb_dim,
                               p=self.dropout) for i in range(configs["layers"])])
        
        self.gated_mlps_in = nn.ModuleList([gated_mlp(seq_in=self.inp_len,seq_out=self.inp_len,channels=self.channels,
                                                      use_update=configs["use_update"]
                                                      ) for i in range(configs["layers"])])
        
        self.stattn_out = nn.ModuleList([stattn_layer(num_of_vertices=self.n_nodes, 
                               num_of_features=self.channels, 
                               num_of_timesteps=self.pred_len,
                               device = configs['device'],
                               emb_dim=self.emb_dim,
                               p=self.dropout) for i in range(configs["layers"])])

        self.gated_mlps_out = nn.ModuleList([gated_mlp(seq_in=self.pred_len,seq_out=self.pred_len,channels=self.channels,
                                                       use_update=configs["use_update"]
                                                       ) for i in range(configs["layers"])])
        
        self.mask_in = nn.Parameter(torch.rand(1, self.channels, 1, int(self.inp_len/2)+1)) #自适应mask
        nn.init.xavier_normal_(self.mask_in)

        self.mask_out = nn.Parameter(torch.rand(1, self.channels, 1, int(self.pred_len/2)+1)) #自适应mask
        nn.init.xavier_normal_(self.mask_out)

        # self.mlp_in = nn.ModuleList([nn.Linear(self.inp_len,self.inp_len) for i in range(configs["layers"])])
        # self.mlp_out = nn.ModuleList([nn.Linear(self.pred_len,self.pred_len) for i in range(configs["layers"])])
        # self.conv_in = nn.Conv2d(self.c_in,self.c_out,kernel_size=3,padding=1)
        # self.conv_out = nn.Conv2d(self.c_in,self.c_out,kernel_size=3,padding=1)
        
    def freq_attn_in(self, x):
        freq = torch.fft.rfft(x)
        y = torch.fft.irfft((self.mask_in)*freq)
        return y
    
    def freq_attn_out(self, x):
        freq = torch.fft.rfft(x)
        y = torch.fft.irfft((self.mask_out)*freq)
        return y
        
    def params(self, configs):
        self.c_in = configs['c_in']
        self.order = configs['order']
        self.c_out = configs['c_out']
        self.channels = configs['c_in']
        self.c_date = configs['c_date']
        self.dp_rate = configs['dropout']
        self.n_nodes = configs['n_nodes']
        self.inp_len = configs['inp_len']
        self.pred_len = configs['pred_len']
        self.dilation = configs['dilation']
        self.emb_dim = configs['adp_dim']
        self.use_guide = configs['use_guide']
        self.dropout = configs["dropout"]
        self.kernel_size = configs["kernel_size"]
        self.device = configs['device']

    def forward(self, x, **args):
        b,c,n,t = x.shape
        if self.use_guide:
            x_t = self.freq_attn_in(x)
        else:
            x_t = torch.zeros((b,self.c_in,1,self.inp_len),device=x.device)
        x_c = x + x_t 
        for (layer,mlp) in zip(self.stattn_in,self.gated_mlps_in):
            x_c = mlp(x_c)
            x_c = torch.einsum("bcnt,btt->bcnt",[x_c,F.gelu(layer(x_c))]) + x_c

        # for (layer,mlp) in zip(self.stattn_in,self.mlp_in):
        #     x_c = mlp(x_c)
        #     x_c = torch.einsum("bcnt,btt->bcnt",[x_c,F.gelu(layer(x_c))]) + x_c
        # h_x = self.conv_in(x+x_t+x_c)
        h_x = self.fconv1(x, x_t, x_c)
        h_y = self.fc_idp(h_x)
        if self.use_guide:
            y_t = self.freq_attn_out(h_y)
        else:
            y_t = torch.zeros((b,self.c_in,1,self.pred_len),device=x.device)
        y_c = h_y + y_t
        i = 0 
        for (layer,mlp) in zip(self.stattn_out,self.gated_mlps_out):
            y_c = mlp(y_c)
            y_c = torch.einsum("bcnt,btt->bcnt",[y_c,F.gelu(layer(y_c))]) + y_c
        # for (layer,mlp) in zip(self.stattn_out,self.mlp_out):
        #     y_c = mlp(y_c)
        #     y_c = torch.einsum("bcnt,btt->bcnt",[y_c,F.gelu(layer(y_c))]) + y_c
        # y = self.conv_out(h_y+y_t+y_c)
        y = self.fconv2(h_y, y_t, y_c)
        loss = 0.0
        return y, loss
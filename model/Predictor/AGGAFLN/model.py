import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .block import FreqConv, Indepent_Linear, TimeEmbedding, stattn_layer, gated_mlp,self_adp_Linear

class Model(nn.Module):
    def __init__(self, configs, **args):
        nn.Module.__init__(self)
        self.params(configs)
        self.fconv1 = FreqConv(4, self.inp_len, self.inp_len, kernel_size=self.kernel_size, dilation=self.dilation, order=self.order)
        self.fconv2 = FreqConv(4, self.pred_len, self.pred_len, kernel_size=self.kernel_size, dilation=self.dilation, order=self.order)
        self.fc_idp = Indepent_Linear(self.inp_len, self.pred_len, self.channels, configs["share"], self.dp_rate)

        self.stattn_out = nn.ModuleList([stattn_layer(num_of_vertices=self.n_nodes, 
                               num_of_features=self.channels, 
                               num_of_timesteps=self.pred_len,
                               device = configs['device'],
                               emb_dim=self.emb_dim,
                               p=self.dropout) for i in range(configs["sta_layers"])])
        
        self.gated_mlps_out = nn.ModuleList([gated_mlp(seq_in=self.pred_len,seq_out=self.pred_len,channels=self.channels,
                                                      use_update=configs["use_update"]
                                                      ) for i in range(configs["sta_layers"])])
        self.stattn_in = nn.ModuleList([stattn_layer(num_of_vertices=self.n_nodes, 
                               num_of_features=self.channels, 
                               num_of_timesteps=self.inp_len,
                               device = configs['device'],
                               emb_dim=self.emb_dim,
                               p=self.dropout) for i in range(configs["sta_layers"])])
        
        self.gated_mlps_in = nn.ModuleList([gated_mlp(seq_in=self.inp_len,seq_out=self.inp_len,channels=self.channels,
                                                      use_update=configs["use_update"]
                                                      ) for i in range(configs["sta_layers"])])

        self.global_graph_c = nn.Parameter(torch.rand(self.channels,self.channels))
        nn.init.xavier_normal_(self.global_graph_c)
        self.latent_graph_c = nn.Parameter(torch.rand(self.channels,self.inp_len//2 + 1))
        nn.init.xavier_normal_(self.latent_graph_c)
        self.latent_global_c = nn.Parameter(torch.randn(self.inp_len//2 + 1,self.inp_len))
        nn.init.xavier_normal_(self.latent_global_c)

        self.global_graph_t = nn.Parameter(torch.rand(self.channels,self.channels))
        nn.init.xavier_normal_(self.global_graph_t)
        self.latent_graph_t = nn.Parameter(torch.rand(self.channels,self.inp_len//2 + 1))
        nn.init.xavier_normal_(self.latent_graph_t)
        self.latent_global_t = nn.Parameter(torch.randn(self.inp_len//2 + 1,self.inp_len))
        nn.init.xavier_normal_(self.latent_global_t)
        
    def freq_attn_in(self, x):
        freq = torch.fft.rfft(x)
        freq.real = torch.einsum("bcnh,cl,lh->blnh",freq.real,self.global_graph_c,self.latent_graph_c)
        freq.imag = torch.einsum("bcnh,cl,lh->blnh",freq.imag,self.global_graph_t,self.latent_graph_t)
        y = torch.fft.irfft(freq)
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
        self.emb_dim = configs['emb_dim']
        self.dropout = configs["dropout"]
        self.kernel_size = configs["kernel_size"]
        self.device = configs['device']

    def forward(self, x, x_mark, y_mark, **args):
        x_t = self.freq_attn_in(x)
        # np.save("x-ett.npy",x.cpu().detach().numpy())
        # np.save("x-t-ett.npy",x_t.cpu().detach().numpy())
        x_c = x + x_t
        loss = 0.0
        for (layer,mlp) in zip(self.stattn_in,self.gated_mlps_in):
            x_c = mlp(x_c)
            w,cadj,tadj= layer(x_c)
            x_c = torch.einsum("bcnt,btt->bcnt",[x_c,F.gelu(w)]) + x_c
            g1 = torch.einsum("cl,lh,ht->ct",self.global_graph_c,self.latent_graph_c,self.latent_global_c) + \
                 torch.einsum("cl,lh,ht->ct",self.global_graph_t,self.latent_graph_t,self.latent_global_t)
            g2 = torch.einsum("cl,th->ct",cadj,tadj)
            if self.training:
                loss += (torch.fft.rfft(g1, dim=-1) - torch.fft.rfft(g2, dim=-1)).abs().mean()
        h_x = self.fconv1(x,x_c)
        h_y = self.fc_idp(h_x)
        y_c = h_y
        for (layer,mlp) in zip(self.stattn_out,self.gated_mlps_out):
            y_c = mlp(y_c)
            w,_,_ = layer(y_c)
            y_c = torch.einsum("bcnt,btt->bcnt",[y_c,F.gelu(w)]) + y_c
        y = self.fconv2(h_y,y_c)
        return y, loss
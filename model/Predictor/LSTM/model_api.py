import torch
import torch.nn as nn
from .model import LSTM

class lstm_api(nn.Module):
    def __init__(self, configs, graph_generator, fixed_adjs):
        super(lstm_api, self).__init__()
        self.configs = self.load_configs(configs)
        self.adjs = fixed_adjs
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        # spatial temporal predictor
        self.model = LSTM(self.configs)
        self.model_init()
        
    def load_configs(self,configs):
        model_configs = configs['model']
        model_configs['c_out'] = configs['envs']['c_out']
        model_configs['seq_len'] = configs['envs']['inp_len']
        model_configs['pred_len'] = configs['envs']['pred_len']
        model_configs['dtype'] = configs['envs']['dtype']
        model_configs['device'] = configs['envs']['device']
        model_configs['num_nodes'] = configs['dataset']['n_nodes']
        return model_configs
    
    def model_init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark,**args):
        if self.with_GraphGen:
            adjs, graph_loss = self.graph_generator(seq_x,seq_x_mark)
            addi_loss = graph_loss
        else:
            adjs = self.adjs
            addi_loss = 0.0
        predicts = self.model(seq_x)
        return predicts, addi_loss
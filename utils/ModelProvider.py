from torch import cat
from model.Predictor.TCAFFN.model_api import fcn_api
from model.Predictor.DCAASFFN.model_api import dcaasffn_api
from model.Predictor.AGGAFLN.model_api import aggafln_api
from model.DGCDN.DGCGN_GraphGenerator import Graph_Generator


def get_model(configs, graph_generator=None, fixed_adjs=None):    
    models_dict = {
        'tcaffn':fcn_api,
        'dca-asffn':dcaasffn_api,
        'agga-fln':aggafln_api,
    }
    # abnorm
    model_name_list = list(models_dict.keys())
    model_name = configs['model']['name'].lower()
    print(model_name)
    if model_name not in model_name_list:
        raise NameError('Model Not Find.')
    # load model
    model = models_dict[model_name](configs, graph_generator, fixed_adjs)
    return model

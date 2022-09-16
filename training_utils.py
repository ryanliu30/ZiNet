import sys
import yaml
sys.path.append("~/machine_learning/ZiNet/")

from VAE_CNN.models.convnext import ZiCVAE

path = "~/machine_learning/ZiNet/"

def process_hparams(hparams):

    return hparams

def model_selector(model_name, sweep_configs = {}):
    if model_name == "ConvNext" or model_name == "1":
        with open(path + "VAE_CNN/configs/convnext.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
        model = ZiCVAE(process_hparams({**hparams, **sweep_configs}))    
    else:
        raise ValueError("Can't Find Model Name {}!".format(model_name))
        
    return model
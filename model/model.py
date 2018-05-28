





from .classification import Classification

from .stargan import StarGAN
from .cvaegan import CVAEGAN
from .vae import VAE


model_dict = {
    "cvaegan" : CVAEGAN,
    'vae' : VAE,
    'stargan' : StarGAN,
    'classification' : Classification
}


def get_model(modelname, modelparams):

    if modelname in model_dict:
        return model_dict[modelname](modelparams)
    else:
        raise "No model named " + modelname




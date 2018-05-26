






from .cvaegan import CVAEGAN
from .vae import VAE



model_dict = {
    "cvaegan" : CVAEGAN,
    'vae' : VAE,
}




def get_model(modelname, modelparams):

    if modelname in model_dict:
        return model_dict[modelname](modelparams)
    else:
        raise "No model named " + modelname




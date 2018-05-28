


import os
import sys
sys.path.append('../')

from .decoder_pixel import DecoderPixel
from .decoder_simple import DecoderSimple


decoder_dict = {
    'DecoderPixel' : DecoderPixel,
    'DecoderSimple' : DecoderSimple,
    # 'resn' : None,
    # 'vgg' : None,
}



decoder_params_dict = {
    'DecoderPixel' : {
        
    },
    'DecoderSimple' : {

    }
    # 'resn' : None,
    # 'vgg' : None,
}



def get_decoder(name, config, model_config, is_training):

    if name in decoder_dict : 
        return decoder_dict[name](config, model_config, is_training, **decoder_params_dict[name])
    else :
        raise Exception("None decoder named " + name)
    pass



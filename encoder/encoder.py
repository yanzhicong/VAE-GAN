
import os
import sys
# import 





from .encoder_pixel import EncoderPixel
from .encoder_simple import EncoderSimple


encoder_dict = {
    "EncoderVGG" : None,
    "EncoderResnet" : None,
    "EncoderPixel" : EncoderPixel,
    "EncoderSimple" : EncoderSimple
}


encoder_params_dict = {
    "EncoderVGG" : {
        
    },
    "EncoderResnet" : {

    },
    "EncoderPixel" : {

    },
    "EncoderSimple" : {

    }
}


def get_encoder(name, config, model_config, is_training):
    if name in encoder_dict : 
        return encoder_dict[name](config, model_config, is_training, **encoder_params_dict[name])
    else:
        raise Exception("None encoder named " + name)



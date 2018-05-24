
import os
import sys
# import 





from encoder_pixel import EncoderPixel



encoder_dict = {
    "EncoderVGG" : None,
    "EncoderResnet" : None,
    "EncoderPixel" : EncoderPixel
}


encoder_params_dict = {
    "EncoderVGG" : {
        
    },
    "EncoderResnet" : {

    },
    "EncoderPixel" : {

    }
}



def get_encoder(name, config, model_config):
    if name in encoder_dict : 
        return encoder_dict[name](config, model_config, **encoder_params_dict[name])

    else:
        raise Exception("None encoder named " + name)



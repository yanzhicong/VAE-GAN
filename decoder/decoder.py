


import os
import sys
sys.path.append('../')



def get_decoder(name, config, is_training):

    if name == "DecoderSimple" : 
        from .decoder_simple import DecoderSimple
        return DecoderSimple(config, is_training)
    else :
        raise Exception("None decoder named " + name)

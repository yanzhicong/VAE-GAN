


import os
import sys
sys.path.append('../')




def get_discriminator(name, config, is_training):
	if name == 'DiscriminatorSimple':
		from .discriminator_simple import DiscriminatorSimple
		return DiscriminatorSimple(config, is_training)
	else : 
		raise Exception("None discriminator named " + name)


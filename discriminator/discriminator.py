


import os
import sys
sys.path.append('../')




def get_discriminator(name, config, is_training):
	if name == 'DiscriminatorSimple':
		from .discriminator_simple import DiscriminatorSimple
		return DiscriminatorSimple(config, is_training)
	elif name == 'cifar10 discriminator' or name == 'discriminator_cifar10':
		from .discriminator_cifar10 import DiscriminatorCifar10
		return DiscriminatorCifar10(config, is_training)
	else : 
		raise Exception("None discriminator named " + name)


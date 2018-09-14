# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2018 ZhicongYan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


def get_model(model_name, model_params):
    if model_name == 'cvaegan':
        from .cvaegan import CVAEGAN
        return CVAEGAN(model_params)
    
    elif model_name == 'vae':
        from .vae import VAE
        return VAE(model_params)
    
    elif model_name == 'cvae':
        from .cvae import CVAE
        return CVAE(model_params)
    
    elif model_name == 'aae':
        from .aae import AAE
        return AAE(model_params)
    
    elif model_name == 'aae_ssl' or model_name == 'aae_semi':
        from .aae_ssl import AAESemiSupervised
        return AAESemiSupervised(model_params)

    elif model_name == 'classification':
        from .classification import Classification
        return Classification(model_params)
    
    elif model_name == 'segmentation':
        from .segmentation import Segmentation
        return Segmentation(model_params)
    
    elif model_name == 'stargan':
        from .stargan import StarGAN
        return StarGAN(model_params)
    
    elif model_name == 'semidgm':
        from .semi_dgm import SemiDeepGenerativeModel
        return SemiDeepGenerativeModel(model_params)
    
    elif model_name == 'semidgm2':
        from .semi_dgm2 import SemiDeepGenerativeModel2
        return SemiDeepGenerativeModel2(model_params)
    
    elif model_name == 'dcgan':
        from .dcgan import DCGAN
        return DCGAN(model_params)
    
    elif model_name == 'wgan_gp' or model_name == 'wgan':
        from .wgan_gp import WGAN_GP
        return WGAN_GP(model_params)
    
    elif model_name == 'improved_gan':
        from .improved_gan import ImprovedGAN
        return ImprovedGAN(model_params)

    else:
        raise Exception("None model named " + model_name)

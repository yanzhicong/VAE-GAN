
import os
import sys
sys.path.append("../")



from encoder.encoder import get_encoder
from decoder.decoder import get_decoder
from classifier.classifier import get_classifier
from discriminator.discriminator import get_discriminator


from .basemodel import BaseModel






class CVAEGAN(BaseModel):


    def __init__(self, config,
        # input_shape=(128, 128, 3),
        # num_attrs=2,
        # z_dims = 64,
        **kwargs
    ):
        super(CVAEGAN, self).__init__(input_shape=config['input_shape'], **kwargs)

        self.input_shape = config['input_shape']
        self.num_classes = config['nb_classes']
        self.z_dim = config['z_dim']


        self.config = config

        # self.f_enc = None
        # self.f_dec = None
        # self.f_dis = None
        # self.f_cls = None
        # self.enc_trainer = None
        # self.dec_trainer = None
        # self.dis_trainer = None
        # self.cls_trainer = None

        self.build_model()


    def build_model(self):

        self.encoder = get_encoder(self.config['encoder'], self.config['encoder params'], self.config)
        self.decoder = get_decoder(self.config['decoder'], self.config['decoder params'], self.config)
        self.classifier = get_classifier(self.config['classifier'], self.config['classifier params'], self.config)
        self.discriminator = get_discriminator(self.config['discriminator', self.config['discriminator params'], self.config])
        

        pass

    def train_on_batch_supervised(self, x_batch, y_batch):
        raise NotImplementedError


    def train_on_batch_unsupervised(self, x_batch):
        raise NotImplementedError   

    def predict(self, z_sample):
        raise NotImplementedError
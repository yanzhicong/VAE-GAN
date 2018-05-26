
class VAE(BaseModel):

    def __init__(self, config,
        # input_shape=(128, 128, 3),
        # num_attrs=2,
        # z_dims = 64,
        **kwargs
    ):

        super(CVAEGAN, self).__init__(input_shape=config['input_shape'], **kwargs)

        self.input_shape = config['input_shape']
        self.z_dim = config['z_dim']
        self.config = config

        self.build_model()
        

    def build_model(self):
        self.encoder = get_encoder(self.config['encoder'], self.config['encoder params'], self.config)
        self.decoder = get_decoder(self.config['decoder'], self.config['decoder params'], self.config)

        self.x_real = tf.placeholder(tf.float32, shape=[None, ] + self.input_shape, name='xinput')

        # self.label_real = tf.placeholder(tf.float32, shape=[None, self.num_classes,], name='cls')


        z_mean, z_log_var = self.encoder(self.x_real)


        eps = tf.placeholder(tf.float32, shape=[None,self.z_dim], name='eps')
        z_sample = z_mean + tf.exp(z_log_var / 2) * eps

        x_decode = self.decoder(z_sample)



        self.kl_loss = -0.5 * tf.sum(1 + z_log_var - tf.square(z_mean) - K)

        # z = get_sample(self.config['sample_func'], (z_avg, z_log_var))
        xent_loss = 

        kl_loss = -0.5 * tf.sum( 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)

        # self.x_fake = self.decoder(tf.concat([z, self.label_real], axis=1))
        # z_possible = tf.placeholder(tf.float32, shape=(None, self.z_dim))
        # c_possible = tf.placeholder(tf.float32, shape=(None, self.nb_classes))

        # x_possible = self.decoder(tf.concat([z_possible, c_possible],axis=1), reuse=True)

        # d_real, feature_disc_real = self.discriminator(self.x_real)
        # d_fake, feature_disc_fake = self.discriminator(x_fake, reuse=True)
        # d_possible, feature_disc_possible = self.discriminator(x_possible, reuse=True)

        # c_real, feature_clas_real = self.classifier(self.x_real)
        # c_fake, feature_clas_fake = self.classifier(self.x_fake)
        # c_possible, feature_clas_possible = self.classifier(self.x_possible)

        self.encoder_loss = 


        pass



    def train_on_batch_supervised(self, x_batch, y_batch):
        raise NotImplementedError


    def train_on_batch_unsupervised(self, x_batch):
        raise NotImplementedError   


    def predict(self, z_sample):
        raise NotImplementedError


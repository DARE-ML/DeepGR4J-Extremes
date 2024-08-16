import tensorflow as tf
from tqdm import tqdm_gui


class ProductionStorage(tf.keras.Model):
    
    def __init__(self, s_init=0.0, x1=None, mu=None, sigma=None):
        super(ProductionStorage, self).__init__()
        self.s_init = s_init
        if x1 is None:
            x1 = tf.random.uniform(shape=[], minval=100, maxval=1000)
        self.set_x1(x1)
        self.mu = mu
        self.sigma = sigma

    def set_x1(self, value):
        self.x1 = tf.Variable(value, dtype=tf.float32)
    
    def get_x1(self):
        return self.x1.numpy()

    def call(self, x, scale=True, include_x=False):
        # Precip and Evaporanspiration
        P = x[:, 0]
        E = x[:, 1]
        
        # Number of simulation timesteps
        num_timesteps = tf.shape(P)[0]
        
        # Unpack the model parameters
        x1 = self.x1

        # Production Storage
        p_n = tf.nn.relu(P - E)
        e_n = tf.nn.relu(E - P)

        p_s_list = []
        e_s_list = []
        perc_list = []
        s_store_list = []

        s_store = self.s_init * x1

        for t in range(num_timesteps):
            # calculate fraction of netto precipitation that fills
            #  production store (eq. 3)
            p_s = x1 * (1 - (s_store/ x1)**2) * tf.tanh(p_n[t]/x1) / (1 + (s_store / x1) * tf.tanh(p_n[t] / x1))

            # from the production store (eq. 4)
            e_s = s_store * (2 - s_store/x1) * tf.tanh(e_n[t]/x1) / (1 + (1 - s_store/x1) * tf.tanh(e_n[t] / x1))

            s_store = s_store + p_s - e_s

            # calculate percolation from actual storage level
            perc = s_store * (1 - (1 + (4/9 * s_store / x1)**4)**(-0.25))
            
            # final update of the production store for this timestep
            s_store = s_store - perc

            # Append updated values
            p_s_list.append(p_s)
            e_s_list.append(e_s)
            perc_list.append(perc)
            s_store_list.append(s_store)
        

        # Expand dim
        p_n = tf.expand_dims(p_n, axis=1)
        e_n = tf.expand_dims(e_n, axis=1)
        p_s = tf.stack(p_s_list)[:, None] 
        perc = tf.stack(perc_list)[:, None]
        s_store = tf.stack(s_store_list)[:, None] 
        
        # Concatenate
        if include_x:
            out = tf.concat([x, p_n, e_n, p_s, perc], axis=1)
        else:
            out = tf.concat([p_n, e_n, p_s, perc], axis=1)

        # Scale
        if scale:
            if self.mu is None and self.sigma is None:
                self.mu = tf.reduce_mean(out, axis=0)
                self.sigma = tf.math.reduce_std(out, axis=0)

            out = (out - self.mu) / self.sigma

        return out, s_store
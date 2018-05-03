import tensorflow as tf
from baselines.her.util import store_args, nn

class GaussianPolicy:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        # Networks.
        with tf.variable_scope('pi'):
            latent = tf.concat(axis=1, values=[o, g])
            for _ in range(self.layers):
                latent = tf.layers.dense(latent, self.hidden, activation=tf.nn.relu)
            self.mu_tf = tf.layers.dense(latent, dimu, activation=None)
            self.sigma_tf = tf.layers.dense(latent, dimu, activation=tf.nn.softplus)
            self.pi_tf = tf.distributions.Normal(loc=self.mu_tf, scale=self.sigma_tf)
            self.raw_tf = self.pi_tf.sample()
            self.a_tf = self.max_u * tf.tanh(self.raw_tf)
            # Deterministic action
            self.da_tf = self.max_u * tf.tanh(self.mu_tf)
            self.a_prob_tf = self.pi_tf.prob(self.u_tf)
            print(self.a_prob_tf)

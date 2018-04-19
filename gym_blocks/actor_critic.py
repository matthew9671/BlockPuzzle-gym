import tensorflow as tf
from baselines.her.util import store_args, nn

BLOCK_FEATURES = 25
FEATURE_SIZE = 64

class AttentionActorCritic:
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

        num_blocks = o.get_shape().as_list()[1] // BLOCK_FEATURES

        # (?, ?, n)
        obs = tf.reshape(o, [-1, num_blocks, BLOCK_FEATURES])

        block_mlp = [64]
        for num_hidden in block_mlp:
            obs = tf.layers.dense(obs, num_hidden, activation=tf.nn.relu)
        obs = tf.layers.dense(obs, FEATURE_SIZE, activation=None)

        # Add all the blocks together
        # (?, n)
        sum_blocks = tf.reduce_sum(obs, axis=1)
        sum_mlp = [64]
        for num_hidden in sum_mlp:
            sum_blocks = tf.layers.dense(sum_blocks, num_hidden, activation=tf.nn.relu)

        sum_blocks = tf.layers.dense(sum_blocks, FEATURE_SIZE, activation=None)

        # (?, 1, n)
        attention = tf.expand_dims(sum_blocks, 1)
        # (?, ?, n)
        attention = tf.tile(attention, [1, num_blocks, 1])
        # (?, ?)
        weights = tf.reduce_sum(attention * obs, axis=2)
        weights = tf.nn.softmax(weights, axis=1)
        # (?, ?, 1)
        weights = tf.expand_dims(weights, 2)
        # (?, ?, n)
        weights = tf.tile(weights, [1, 1, FEATURE_SIZE])
        weighted = weights * obs
        # (?, n)
        gated_obs = tf.reduce_sum(weighted, axis=1)

        input_pi = tf.concat(axis=1, values=[gated_obs, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[gated_obs, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[gated_obs, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

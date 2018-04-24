import tensorflow as tf
from baselines.her.util import store_args, nn

BLOCK_FEATURES = 15
ENV_FEATURES = 10
FEATURE_SIZE = 64
ATTENTION_CNT = 1

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

        num_blocks = (o.get_shape().as_list()[1] - ENV_FEATURES) // BLOCK_FEATURES

        obs_env = tf.slice(o, [0, 0], [-1, ENV_FEATURES])
        obs_blocks = tf.slice(o, [0, ENV_FEATURES], [-1, -1])

        # (?, ?, n)
        input_blocks = tf.reshape(obs_blocks, [-1, num_blocks, BLOCK_FEATURES])

        to_concat = []

        with tf.variable_scope('Q'):
            for _ in range(ATTENTION_CNT):
                block_mlp = [64, 64]
                obs_blocks = input_blocks
                for num_hidden in block_mlp:
                    obs_blocks = tf.layers.dense(obs_blocks, num_hidden, activation=tf.nn.relu)
                obs_blocks = tf.layers.dense(obs_blocks, FEATURE_SIZE, activation=None)

                # Add all the blocks together
                # (?, n)
                sum_blocks = tf.reduce_sum(obs_blocks, axis=1)
                sum_mlp = [64, 64]
                for num_hidden in sum_mlp:
                    sum_blocks = tf.layers.dense(sum_blocks, num_hidden, activation=tf.nn.relu)

                sum_blocks = tf.layers.dense(sum_blocks, FEATURE_SIZE, activation=None)

                # (?, 1, n)
                attention = tf.expand_dims(sum_blocks, 1)
                # (?, ?, n)
                attention = tf.tile(attention, [1, num_blocks, 1])
                attention = tf.nn.l2_normalize(attention, axis=2)
                # (?, ?)
                norm_block_emb = tf.nn.l2_normalize(obs_blocks, axis=2)
                weights = tf.reduce_sum(attention * norm_block_emb, axis=2)
                # weights = tf.nn.softmax(weights, axis=1)
                self.block_weights = weights
                # (?, ?, 1)
                weights = tf.expand_dims(weights, 2)
                # (?, ?, n)
                weights = tf.tile(weights, [1, 1, FEATURE_SIZE])
                weighted = weights * obs_blocks
                # (?, n)
                gated_obs = tf.reduce_sum(weighted, axis=1)
                to_concat.append(gated_obs)

            gated_obs = tf.concat(axis=1, values=to_concat)

            input_pi = tf.concat(axis=1, values=[obs_env, gated_obs, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[obs_env, gated_obs, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[obs_env, gated_obs, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

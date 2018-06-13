import tensorflow as tf
from baselines.her.util import store_args, nn

COLOR_FEATURES = 4
BLOCK_BASE_FEATURES = 15
BLOCK_FEATURES = BLOCK_BASE_FEATURES + COLOR_FEATURES
ENV_FEATURES = 10
FEATURE_SIZE = 128
ATTENTION_CNT = 1

class AttentionGaussianPolicy:
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

        o = self.o_tf

        env_size = tf.constant(ENV_FEATURES, tf.int32)
        block_size = tf.constant(BLOCK_FEATURES, tf.int32)
        batch_size = tf.shape(o)[0]
        obs_shape = tf.shape(o)[1]

        max_num_blocks = tf.cast((obs_shape - env_size) / block_size, tf.int32)
        # Number of blocks comes first in the observation
        num_blocks = tf.reshape(tf.slice(o, [0, 0], [-1, 1]), [-1,])
        num_blocks = tf.cast(num_blocks, tf.int32)
        o = tf.slice(o, [0, 1], [-1, -1])

        o = self.o_stats.normalize(o)

        obs_env = tf.slice(o, [0, 0], [-1, ENV_FEATURES])
        obs_blocks = tf.slice(o, [0, ENV_FEATURES], [-1, -1])
        #print('######################', obs_blocks)

        input_blocks = tf.reshape(obs_blocks, [-1, max_num_blocks, BLOCK_FEATURES])
        #print('######################', input_blocks)
        to_concat = []

        with tf.variable_scope('pi'):
            for _ in range(ATTENTION_CNT):
                block_mlp = [64]
                obs_blocks = input_blocks
                for num_hidden in block_mlp:
                    obs_blocks = tf.layers.dense(obs_blocks, num_hidden, activation=tf.nn.relu)
                    #print('###########', obs_blocks)

                obs_blocks = tf.layers.dense(obs_blocks, FEATURE_SIZE, activation=None)
                # rnn_input = tf.transpose(obs_blocks, perm=[1,0,2])
                RNN_HIDDEN = FEATURE_SIZE
                lstm = tf.contrib.rnn.LSTMCell(RNN_HIDDEN, state_is_tuple=True)
                # For loop doesn't work! Use tf.nn.dynamic_rnn instead!
                # https://stackoverflow.com/questions/43341374/tensorflow-dynamic-rnn-lstm-how-to-format-input
                blocks, _ = tf.nn.dynamic_rnn(lstm, obs_blocks, 
                    sequence_length=num_blocks, dtype=tf.float32)

                # Add all the blocks together
                # (?, n)
                sum_blocks = tf.reduce_sum(blocks, axis=1)

                sum_mlp = [64]
                for num_hidden in sum_mlp:
                    sum_blocks = tf.layers.dense(sum_blocks, num_hidden, activation=tf.nn.tanh)

                sum_blocks = tf.layers.dense(sum_blocks, FEATURE_SIZE, activation=None)
                print(sum_blocks)
                # (?, 1, n)
                attention = tf.expand_dims(sum_blocks, 1)
                #print('###########', attention)

                # (?, ?, n)
                attention = tf.tile(attention, [1, max_num_blocks, 1])
                #print('###########', attention)

                attention = tf.nn.l2_normalize(attention, axis=2)
                #print('###########', attention)

                # (?, ?)
                norm_block_emb = tf.nn.l2_normalize(blocks, axis=2)
                #print('###########', attention)
                #print('###########', norm_block_emb)


                weights = tf.reduce_sum(attention * norm_block_emb, axis=2)
                weights = tf.nn.softmax(weights, axis=1)
                print('###########', weights)
                sindex = tf.argmax(weights, axis=1, output_type=tf.int32)
                print('###########',sindex)

                findex = tf.range(tf.shape(sindex)[0])
                #index = tf.stack(tf.meshgrid(tf.range(0,batch_size), tf.range(0,batch_size)) + [ sindex ], axis=2)

                print('###########', findex)

                index = tf.stack([findex, sindex])
                index = tf.transpose(index, perm=[1, 0])
                #sind = tf.expand_dims(ind, axis=1)
                print('###########', index)
                chosen_block = tf.gather_nd(input_blocks, index)

                print('###########', chosen_block)


                self.block_weights = weights
                # (?, ?, 1)
                weights = tf.expand_dims(weights, 2)
                # (?, ?, n)
                weights = tf.tile(weights, [1, 1, FEATURE_SIZE])
                weighted = weights * blocks
                # (?, n)
                gated_obs = tf.reduce_sum(weighted, axis=1)
                to_concat.append(gated_obs)
                to_concat.append(chosen_block)
            gated_obs = tf.concat(axis=1, values=to_concat)
            input_pi = tf.concat(axis=1, values=[obs_env, gated_obs])  # for actor

            latent = input_pi
            # For debugging
            self.embedding = latent

            for _ in range(self.layers):
                latent = tf.layers.dense(latent, self.hidden, activation=tf.nn.relu)
            self.mu_tf = tf.layers.dense(latent, dimu, activation=None)
            self.sigma_tf = tf.layers.dense(latent, dimu, activation=tf.nn.softplus)
            self.pg_pi_tf = tf.distributions.Normal(loc=self.mu_tf, scale=self.sigma_tf)
            self.raw_tf = self.pg_pi_tf.sample()
            self.a_tf = self.max_u * tf.tanh(self.raw_tf)
            # Deterministic action
            self.da_tf = self.max_u * tf.tanh(self.mu_tf)
            self.a_prob_tf = self.pg_pi_tf.prob(self.u_tf)
            # print(self.a_prob_tf)

            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            self.saver = tf.train.Saver(vars)
            # Stop the gradient if we don't want to train the embbeding of the blocks
            input_pi = tf.stop_gradient(input_pi)

        # Actor critic Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[input_pi, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[input_pi, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

    def save_weights(self, sess, path):
        print("Saving weights!")
        self.saver.save(sess, "{}/weights".format(path))

    def load_weights(self, sess, path):
        print("Loading weights!")
        self.saver.restore(sess, path)


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
            latent = o#tf.concat(axis=1, values=[o, g])
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
            # print(self.a_prob_tf)

            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            self.saver = tf.train.Saver(vars)

    def save_weights(self, sess, path):
        print("Saving weights!")
        self.saver.save(sess, "{}/weights".format(path))

    def load_weights(self, sess, path):
        print("Loading weights!")
        self.saver.restore(sess, path)

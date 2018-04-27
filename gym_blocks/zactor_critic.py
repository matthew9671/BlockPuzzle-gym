import tensorflow as tf
from baselines.her.util import store_args, nn

BLOCK_FEATURES = 16
ENV_FEATURES = 10
FEATURE_SIZE = 128
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
        #print('######################', obs_blocks)	

        input_blocks = tf.reshape(obs_blocks, [-1, num_blocks, BLOCK_FEATURES])
        #print('######################', input_blocks)  	
        to_concat = []
        batch_size = tf.shape(obs_blocks)[0]

        with tf.variable_scope('Q'):
            for _ in range(ATTENTION_CNT):
                block_mlp = [64]
                obs_blocks = input_blocks
                for num_hidden in block_mlp:
                    obs_blocks = tf.layers.dense(obs_blocks, num_hidden, activation=tf.nn.relu)
                    #print('###########', obs_blocks)

                obs_blocks = tf.layers.dense(obs_blocks, FEATURE_SIZE, activation=None)
                rnn_input = tf.unstack(tf.transpose(obs_blocks, perm=[1,0,2])) 
                RNN_HIDDEN = 128
                lstm = tf.contrib.rnn.LSTMCell(RNN_HIDDEN, state_is_tuple=True)

                #print('###########batch', batch_size, RNN_HIDDEN)
                hid_state = tf.zeros([batch_size, RNN_HIDDEN])
                cell_state = tf.zeros([batch_size, RNN_HIDDEN])
                state = (hid_state, cell_state)
 
                #out = tf.scan(lambda a, x: lstm(x, a), rnn_input, initializer=hid_state) 
                #print('######out', out)

                blocks = [] 
                for block in rnn_input:
                    output, state = lstm(block, state)
                    blocks.append(output) 
                    #print('#####', output)
                    #print('#####', state)

                blocks = tf.stack(blocks)
                blocks = tf.transpose(blocks, perm=[1, 0, 2])

                # Add all the blocks together
                # (?, n)
                sum_blocks = tf.reduce_sum(blocks, axis=1)
                #attention_input = tf.concat(axis=2, values=[obs_blocks, sum_blocks])
                #print('$$$$$$$', attention_input)

                sum_mlp = [64]
                for num_hidden in sum_mlp:
                    sum_blocks = tf.layers.dense(sum_blocks, num_hidden, activation=tf.nn.tanh)

                sum_blocks = tf.layers.dense(sum_blocks, FEATURE_SIZE, activation=None)
                print(sum_blocks)
                # (?, 1, n)
                attention = tf.expand_dims(sum_blocks, 1)
                
                print('###########', attention)

                # (?, ?, n)
                attention = tf.tile(attention, [1, num_blocks, 1])
                print('###########', attention)


                attention = tf.nn.l2_normalize(attention, axis=2)
                print('###########', attention)

                # (?, ?)
                norm_block_emb = tf.nn.l2_normalize(blocks, axis=2)
                print('###########', attention)
                print('###########', norm_block_emb)


                weights = tf.reduce_sum(attention * norm_block_emb, axis=2)
                # weights = tf.nn.softmax(weights, axis=1)
                self.block_weights = weights
                # (?, ?, 1)
                weights = tf.expand_dims(weights, 2)
                # (?, ?, n)
                weights = tf.tile(weights, [1, 1, FEATURE_SIZE])
                weighted = weights * blocks
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

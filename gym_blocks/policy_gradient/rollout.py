from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from baselines.her.util import convert_episode_to_batch_major, store_args

class RolloutStudent:

    @store_args
    def __init__(self, make_env, policy, expert, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, gamma=None, 
                 beta_final=None, annealing_coeff=None, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

        # ------------
        self.gamma = gamma
        self.time = 0.0
        self.beta_final = beta_final
        self.annealing_coeff = annealing_coeff
        self.expert = expert
        # ------------

    def reset_rollout(self, i, test=False):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        if test:
            # Set difficulty to maximum
            obs = self.envs[i].unwrapped.set_test()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self, test=False):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i, test)

    def increase_difficulty(self):
        max_level = False
        for env in self.envs:
            max_level = env.unwrapped.increase_difficulty()
        if not max_level:
            return self.envs[0].unwrapped.get_difficulty()
        else:
            return None

    def trim(self, o, g, ag, dimo, dimg):
        # No need to trim
        if o.shape[-1] == dimo:
            return o, g, ag
        # If the shapes don't match, it means there are one extra block
        # we need to get rid of
        if len(o.shape) == 1:
            o_ = o[:dimo]
            g_, ag_ = []
            num_objs = (int)(dimg ** 0.5) + 1
            assert (int)(len(g) ** 0.5) == num_objs
            for i in range(len(g)):
                if (i // num_objs != (num_objs - 1) and
                    i % num_objs != (num_objs - 1)):
                    g_.append(g[i])
                    ag_.append(ag[i])
            g_ = np.asarray(g_)
            ag_ = np.asarray(ag_)
        else:
            o_ = o[:,:dimo]
            batch_size = o.shape[0]
            g_ = [[] for _ in range(batch_size)]
            ag_ = [[] for _ in range(batch_size)]
            num_objs = (int)(dimg ** 0.5) + 1
            assert (int)(g.shape[1] ** 0.5) == num_objs
            for i in range(g.shape[1]):
                if (i // num_objs != (num_objs - 1) and
                    i % num_objs != (num_objs - 1)):
                    for j in range(batch_size):
                        g_[j].append(g[j][i])
                        ag_[j].append(ag[j][i])
            g_ = np.asarray(g_)
            ag_ = np.asarray(ag_)
        return o_, g_, ag_

    def generate_rollouts(self, render=False, test=False, exploit=False):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts(test)

        # Annealing
        if self.expert != None:
            beta = self.beta()
        else:
            beta = 0

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes, returns, sigmas = [], [], [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        for t in range(self.T):
            if np.random.rand() < beta:
                o_, g_, ag_ = self.trim(o, self.g, ag, self.expert.dimo, self.expert.dimg)
                policy_output = self.expert.get_actions(o_, ag_, g_, compute_raw=True)
                u, raw = policy_output
            else:
                policy_output = self.policy.get_actions(
                    o, ag, self.g, exploit=exploit)
                u, raw, sigma = policy_output
            # We can't report sigma accurately when we are using the expert
            if self.expert != None:
                sigma = np.zeros((self.rollout_batch_size, self.dims['u']))

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)
                raw = raw.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # --------------
            r_new = np.zeros(self.rollout_batch_size)
            # --------------
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                # print(u[i])
                try:
                    # We don't ignore reward here 
                    # because we need to compute the return
                    curr_o_new, r, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    # --------------
                    r_new[i] = r
                    # --------------
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if render:
                        self.envs[i].render()
                except MujocoException as e:
                    self.logger.info(str(e))
                    self.logger.info('Exception thrown by Mujoco. Giving up on life...')
                    assert(False)
                    return self.generate_rollouts(render, test)

            if np.isnan(o_new).any():
                self.logger.info('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts(test)
                return self.generate_rollouts(render, test)

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(raw.copy())
            goals.append(self.g.copy())
            sigmas.append(sigma.copy())
            # ---------
            returns.append(r_new.copy())
            for t_ in range(t):
                r_new = r_new.copy()
                returns[t_] += self.gamma ** (t - t_) * r_new
            # ---------
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals,
                       # --------
                       G=returns,
                       sigma=sigmas)
                       # --------
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)

        self.success_history.append(success_rate)
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def beta(self):
        return (1.0 - self.beta_final) * np.exp(-self.time / self.annealing_coeff) + self.beta_final

    def anneal(self):
        self.time += 1.0
        if self.expert != None:
            self.logger.info("Beta = {}".format(self.beta()))

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def save_policy_weights(self, path):
        self.policy.save_weights(path)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)

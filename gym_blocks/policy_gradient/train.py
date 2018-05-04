import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI
import pickle

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import config
from rollout import RolloutStudent
from baselines.her.util import mpi_fork

import gym_blocks

SUCCESS_THRESHOLD = 0.7

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, examiner, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, render, level, curriculum, max_test, **kwargs):

    if curriculum and level > 0:
        l = level
        for i in range(l):
        
            level = rollout_worker.increase_difficulty()
            evaluator.increase_difficulty()
            examiner.increase_difficulty()
            if level != None:
                logger.info("Difficulty increased to level {}!".format(level))

    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1

    # # Warming up the replay memory
    # for _ in range(20):
    #     episode = rollout_worker.generate_rollouts()
    #     policy.store_episode(episode)

    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()

        # test
        logger.info("Start testing!")
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts(render=render, test=False, exploit=True)

        logger.info("Start final exams!")
        # final exam
        examiner.clear_history()
        for _ in range(n_test_rollouts):
            examiner.generate_rollouts(render=render, test=True, exploit=True)

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in examiner.logs('finals'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        rollout_worker.anneal()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(examiner.current_success_rate())
        test_success_rate = mpi_average(evaluator.current_success_rate())

        # Increase difficulty in curriculum learning
        if curriculum and test_success_rate >= SUCCESS_THRESHOLD:
            level = rollout_worker.increase_difficulty()
            evaluator.increase_difficulty()
            examiner.increase_difficulty()
            if level != None:
                logger.info("Difficulty increased to level {}!".format(level))

        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            rollout_worker.save_policy(best_policy_path)
            rollout_worker.save_policy(latest_policy_path)

            rollout_worker.save_policy_weights(logger.get_dir())
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            rollout_worker.save_policy(policy_path)

        # print("Saving and loading!!!")

        # with open(best_policy_path, 'rb') as f:
        #     policy = pickle.load(f)

        # evaluator.policy = policy
        # evaluator.clear_history()
        # for _ in range(n_test_rollouts):
        #     evaluator.generate_rollouts(render=render, test=False, exploit=True)

        # print("success rate: {}".format(mpi_average(evaluator.current_success_rate())))

        # assert False

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env_name, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,
    override_params={}, save_policies=True, render=False, max_test=True, expert_file="", policy_file="", level=0, curriculum=True
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' + 
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)

    if policy_file == "":
        policy = config.configure_pggd(dims=dims, params=params, clip_return=clip_return)
    else:
        # Load policy.
        with open(policy_file, 'rb') as f:
            policy = pickle.load(f)
        fn = config.configure_her(params)
        # print(fn)
        policy.set_sample_transitions(fn)
        # print(dir(policy))
        policy.set_obs_size(dims)

    if expert_file != "":
        with open(expert_file, 'rb') as f:
            expert = pickle.load(f)
    else:
        expert = None

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        'beta_final': params['beta_final'],
        'annealing_coeff': params['annealing_coeff']
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': False,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutStudent(params['make_env'], policy, expert, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    examiner = RolloutStudent(params['make_env'], policy, None, dims, logger, **eval_params)
    examiner.seed(rank_seed)

    evaluator = RolloutStudent(params['make_env'], policy, None, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker, examiner=examiner,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies,
        render=render, level=level, curriculum=curriculum, max_test=max_test)


@click.command()
@click.option('--env_name', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=500, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--expert_file', type=str, default='', help='the path of the pre-learned expert policy')
@click.option('--policy_file', type=str, default='', help='the path of the pre-learned policy to fine tune')
@click.option('--render/--no-render', default=False)
@click.option('--max_test/--no_max_test', default=True)
@click.option('--level', type=int, default=0, help='starting difficulty')
@click.option('--curriculum/--no_curriculum', default=False)
def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()

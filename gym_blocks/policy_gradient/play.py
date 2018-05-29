import click
import numpy as np
import pickle

from baselines import logger
from baselines.common import set_global_seeds
import config
from rollout import RolloutStudent
from pggd import PGGD

import gym_blocks

@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--env_name', type=str, default='')
@click.option('--render', type=int, default=1)
@click.option('--level', type=int, default=0)
@click.option('--dimo', type=int, default=40)
def main(policy_file, seed, n_test_rollouts, render, level, dimo, env_name):
    set_global_seeds(seed)

    PGGD.DIMO = dimo
    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    if env_name == '':
        env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': False,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]
    
    evaluator = RolloutStudent(params['make_env'], policy, None, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    # Set the evaluator to the corresponding difficulty level
    for _ in range(level):
        evaluator.increase_difficulty()

    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts(render=True, test=True, exploit=True)

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()

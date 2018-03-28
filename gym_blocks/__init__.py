import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='BlocksTest-v0',
    entry_point='gym_blocks.envs:BlocksEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Banana-v0',
    entry_point='gym_banana.envs:BananaEnv',
)

register(
    id='haha-v0',
    entry_point='gym_banana.envs:BlocksEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

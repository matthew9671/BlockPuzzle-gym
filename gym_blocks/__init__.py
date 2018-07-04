import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='GripperTouch-v0',
    entry_point='gym_blocks.envs:GripperTouchEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='BlocksTouch-v0',
    entry_point='gym_blocks.envs:BlocksTouchEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='ToppleTower-v0',
    entry_point='gym_blocks.envs:ToppleTowerEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='BlocksTouchCurriculum-v0',
    entry_point='gym_blocks.envs:BlocksTouchCurriculum',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='BlocksTouchChoose-v0',
    entry_point='gym_blocks.envs:BlocksTouchChooseEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='BlocksTouchChooseCurriculum-v0',
    entry_point='gym_blocks.envs:BlocksTouchChooseCurriculum',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='BlocksTouchVariation-v0',
    entry_point='gym_blocks.envs:BlocksTouchVariationEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)
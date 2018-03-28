from gym import utils
from gym_banana.envs import fetch_env

# class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
#     def __init__(self, reward_type='sparse'):
#         initial_qpos = {
#             'robot0:slide0': 0.405,
#             'robot0:slide1': 0.48,
#             'robot0:slide2': 0.0,
#             'table0:slide0': 1.05,
#             'table0:slide1': 0.4,
#             'table0:slide2': 0.0,
#             'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
#         }
#         fetch_env.FetchEnv.__init__(
#             self, 'fetch/pick_and_place.xml', has_object=True, block_gripper=False, n_substeps=20,
#             gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
#             obj_range=0.15, target_range=0.15, distance_threshold=0.05,
#             initial_qpos=initial_qpos, reward_type=reward_type)
#         utils.EzPickle.__init__(self)

class BlocksEnv(fetch_env.FetchBlocksEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.25, 0.55, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.7, 0.4, 1., 0., 0., 0.],
            # 'object2:joint': [1.25, 0.85, 0.4, 1., 0., 0., 0.],
            # 'object3:joint': [1.25, 0.1, 0.4, 1., 0., 0., 0.]
        }
        fetch_env.FetchBlocksEnv.__init__(
            self, 'fetch/blocks.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
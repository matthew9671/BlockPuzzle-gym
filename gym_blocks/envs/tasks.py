from gym import utils
from gym_blocks.envs import fetch_env

class GripperTouchEnv(fetch_env.GripperTouchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.25, 0.55, 0.4, 1., 0., 0., 0.]
        }
        fetch_env.GripperTouchEnv.__init__(
            self, 'fetch/1block.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

class BlocksTouchEnv(fetch_env.BlocksTouchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.25, 0.55, 0.46, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.85, 0.46, 1., 0., 0., 0.]
        }
        fetch_env.BlocksTouchEnv.__init__(
            self, curriculum=False, model_path='fetch/2blocks.xml', has_object=True, 
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

class BlocksTouchCurriculum(fetch_env.BlocksTouchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.25, 0.55, 0.46, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.58, 0.46, 1., 0., 0., 0.]
        }
        fetch_env.BlocksTouchEnv.__init__(
            self, curriculum=True, model_path='fetch/2blocks.xml', has_object=True, 
            block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

class BlocksTouchChooseEnv(fetch_env.BlocksTouchChooseEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.25, 0.55, 0.46, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.6, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.65, 0.46, 1., 0., 0., 0.]
        }
        fetch_env.BlocksTouchChooseEnv.__init__(
            self, curriculum=False, model_path='fetch/3blocks.xml', has_object=True, 
            block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

class BlocksTouchChooseCurriculum(fetch_env.BlocksTouchChooseEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.275, 0.6, 0.46, 1., 0., 0., 0.],
            'object1:joint': [1.3, 0.75, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.075, 0.425, 0.46, 1., 0., 0., 0.]
        }
        fetch_env.BlocksTouchChooseEnv.__init__(
            self, curriculum=True, model_path='fetch/3blocks.xml', has_object=True, 
            block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

class ToppleTowerEnv(fetch_env.ToppleTowerEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.25, 0.6, 0.46, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.6, 0.485, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.6, 0.51, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.6, 0.535, 1., 0., 0., 0.]
        }
        fetch_env.ToppleTowerEnv.__init__(
            self, 'fetch/4blocks.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
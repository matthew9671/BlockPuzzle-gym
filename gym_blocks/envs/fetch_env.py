import os
import copy
import numpy as np

from gym.envs.robotics import rotations, utils
from gym_blocks.envs import robot_env
import mujoco_py
from mujoco_py.modder import TextureModder

# Code for the colors used in puzzle solving
GREY = 0
RED = 1
GREEN = 2
BLUE = 3
NUM_COLORS = 4

COLORS_RGB = [(100, 100, 100), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

BLOCK_SIZE = 0.05
MIN_BLOCK_DIST = 1.5 * BLOCK_SIZE

TABLE_W = 0.25
TABLE_H = 0.35
TABLE_X = 1.05 + TABLE_W
TABLE_Y = 0.40 + TABLE_H

TABLE_W -= BLOCK_SIZE / 2
TABLE_H -= BLOCK_SIZE / 2

def out_of_table(pos):
    # print(pos)
    return abs(pos[0] - TABLE_X) > TABLE_W or abs(pos[1] - TABLE_Y) > TABLE_H

def one_hot_color(c):
    result = np.zeros(NUM_COLORS)
    result[c] = 1
    return result

class BlocksEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, num_blocks
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        # This should always be sparse
        self.reward_type = reward_type
        # The number of objects present in the scene
        # Now this number is fixed
        self.num_objs = num_blocks + 2
        self.obj_colors = self._sample_colors()
        # This is the goal vector that we define for this task
        self.achieved_goal = -1 * np.ones([self.num_objs, self.num_objs])

        self.has_succeeded = False
        self.id2obj = None
        # For curriculum learning
        self.difficulty = 0
        super(BlocksEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    def _sample_from_table(self):
        return np.asarray([TABLE_X + self.np_random.uniform(-TABLE_W, TABLE_W),
                           TABLE_Y + self.np_random.uniform(-TABLE_H, TABLE_H)])

    # For curriculum learning
    def increase_difficulty(self):
        raise NotImplementedError()

    def get_difficulty(self):
        return self.difficulty

    # Configure the environment for testing
    def set_test(self):
        raise NotImplementedError()

    def _sample_colors(self):
        raise NotImplementedError()

    def _geom2objid(self, i):
        name = self.sim.model.geom_id2name(i)
        # First object is reserved for the gripper
        # and the second object is reserved for the table
        if name != None:
            if "finger" in name:
                return 0
            elif name == "table":
                return 1
            elif "object" in name:
                return int(name[6:]) + 2
        return None

    def _check_goal(self):
        for i in range(self.num_objs):
            for j in range(self.num_objs):
                if (self.achieved_goal[i][j] != self.achieved_goal[j][i]):
                    return False
        return True

    # GoalEnv methods
    # ----------------------------

    # Goal is defined as a matrix of touching conditions
    # -1 for never touched, 1 for is touching, 0 for touched but not touching
    # In the desired goal we simply input 1 for the blocks we want touching
    # and -1 for blocks we never want touch, and 0 for the blocks we don't care
    # In this setting, if the achieved goal aligns with the desired goal
    # their dot product should equal to the number of non-zero elements
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        goal_len = min(achieved_goal.shape[0], goal.shape[0])
        achieved_goal = achieved_goal[:goal_len]
        goal = goal[:goal_len]

        d = np.sum(achieved_goal * goal, axis=-1)
        c = np.count_nonzero(goal, axis=-1)
        return (1.0-(d != c).astype(np.float32))

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()
        # Update achieved goal
        for i in range(self.num_objs):
            for j in range(self.num_objs):
                if (self.achieved_goal[i][j] == 1):
                    self.achieved_goal[i][j] = 0
        d = self.sim.data
        for i in range(d.ncon):
            con = d.contact[i]
            i1 = con.geom1
            i2 = con.geom2
            obj1 = self.id2obj[i1]
            obj2 = self.id2obj[i2]
            if (obj1 != None and obj2 != None):
                self.achieved_goal[obj1][obj2] = 1
                self.achieved_goal[obj2][obj1] = 1

    # No changes here
    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        object_pos, object_rot, object_velp, object_velr, object_rel_pos = [], [], [], [], []

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = []

        for i in range(self.num_objs-2):
            obj_name = 'object{}'.format(i)
            temp_pos = self.sim.data.get_site_xpos(obj_name)
            # rotations
            temp_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_name))
            # velocities
            temp_velp = self.sim.data.get_site_xvelp(obj_name) * dt
            temp_velr = self.sim.data.get_site_xvelr(obj_name) * dt
            # gripper state
            temp_rel_pos = temp_pos - grip_pos
            temp_velp -= grip_velp

            block_obs = np.concatenate([temp_pos.ravel(), 
                temp_rel_pos.ravel(), temp_rot.ravel(),
                temp_velp.ravel(), temp_velr.ravel()])

            obs = np.concatenate([obs, block_obs])

        assert(self._check_goal())
        achieved_goal = self.achieved_goal.copy().ravel()

        obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel, obs])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        # sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        # site_id = self.sim.model.site_name2id('target0')
        # self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        # self.sim.forward()
        pass

    def _reset_sim(self, test=False):
        self.sim.set_state(self.initial_state)
        # Randomize colors for each object
        self.obj_colors = self._sample_colors()
        # Randomize start position of each object
        self._randomize_objects(test)
        self.sim.forward()
        self.has_succeeded = False
        return True

    def _randomize_objects(self, test=False):
        raise NotImplementedError()

    def _sample_goal(self):
        C = self.obj_colors
        goal = []
        for i in range(self.num_objs):
            for j in range(self.num_objs):
                if ((C[i] == RED and C[j] == BLUE) or 
                   (C[i] == BLUE and C[j] == RED)):
                    goal.append(-1)
                elif ((C[i] == GREEN and C[j] == BLUE) or 
                     (C[i] == BLUE and C[j] == GREEN)):
                    goal.append(1)
                else:
                    goal.append(0)
        return np.asarray(goal)

    def _is_success(self, achieved_goal, desired_goal):
        # Override whatever input it gives
        achieved_goal, desired_goal = self.achieved_goal.ravel(), self.goal
        r = self.compute_reward(achieved_goal, desired_goal, None)
        if r == 1:
            self.has_succeeded = True
        return self.has_succeeded

    def _env_setup(self, initial_qpos):
        self.id2obj = [self._geom2objid(i) for i in range(self.sim.model.ngeom)]

        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        if not hasattr(self, 'initial_gripper_xpos'):
            self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

        # Change the colors to match the success conditions
        self.color_modder = TextureModder(self.sim)
        for i in range(self.sim.model.ngeom):
            obj_id = self.id2obj[i]
            if obj_id != None:
                name = self.sim.model.geom_id2name(i)
                if 'table' in name:
                    color = np.asarray(COLORS_RGB[self.obj_colors[obj_id]])
                    self.color_modder.set_rgb(name, color * 2)
                else:
                    self.color_modder.set_rgb(name, 
                        COLORS_RGB[self.obj_colors[obj_id]])

class GripperTouchEnv(BlocksEnv):
    """A very simple environment in which the gripper has to touch the block"""
    def __init__(self, *args, **kwargs):
        super(GripperTouchEnv, self).__init__(*args, **kwargs, num_blocks=1)
        # utils.EzPickle.__init__(self)
    
    def _sample_colors(self):
        #     Gripper Table Block0
        #        |      |     |
        return [BLUE, GREY, GREEN]

    def _randomize_objects(self, test=False):
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = (self.initial_gripper_xpos[:2] + 
                self.np_random.uniform(-self.obj_range, self.obj_range, size=2))
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

class BlocksTouchEnv(BlocksEnv):
    """A simple environment in which the gripper has to make two blocks touch"""
    def __init__(self, curriculum, *args, **kwargs):
        super(BlocksTouchEnv, self).__init__(*args, **kwargs, num_blocks=2)
        if curriculum:
            self.obj_range = 0.08
            self.obj_range_step = 0.025
            self.max_obj_range = 0.2
        else:
            self.max_obj_range = self.obj_range
            self.obj_range_step = 0

    # Returns true if maximum difficulty reached
    def increase_difficulty(self):
        self.obj_range += self.obj_range_step
        if self.obj_range > self.max_obj_range:
            self.obj_range = self.max_obj_range
            return True
        else:
            self.difficulty += 1
            return False

    def _sample_colors(self):
        #     Gripper Table Block0 Block1
        #        |      |     |      |
        return [GREY, GREY, GREEN, BLUE]

    def set_test(self):
        self._randomize_objects(True)
        self.goal = self._sample_goal().copy()
        return self._get_obs()

    def _randomize_objects(self, test=False):
        object_xpos = self.initial_gripper_xpos[:2]
        # Set the position of the first block
        if test:
            obj_range = self.max_obj_range
        else:
            obj_range = self.obj_range

        # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        object_xpos = (self.initial_gripper_xpos[:2] + 
            self.np_random.uniform(-obj_range/2, obj_range/2, size=2))
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        object0_pos = object_xpos
        # Set the position of the second block
        do = True
        while do:
            direction = np.random.normal(size=2)
            direction = direction / np.linalg.norm(direction)
            mag = np.random.uniform(MIN_BLOCK_DIST, obj_range)
            object_xpos = (object0_pos + direction * mag)
            do = out_of_table(object_xpos)
        # print(object_xpos)
        object_qpos = self.sim.data.get_joint_qpos('object1:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object1:joint', object_qpos)

class BlocksTouchChooseEnv(BlocksEnv):
    """An environment in which the gripper has to make 
    blocks of the right colors touch"""
    def __init__(self, curriculum, challenge=False, *args, **kwargs):
        super(BlocksTouchChooseEnv, self).__init__(*args, **kwargs, num_blocks=3)
        # utils.EzPickle.__init__(self)
        if curriculum:
            self.obj_range = 0.08
            self.obj_range_step = 0.025
            self.wrong_obj_range = 0.2
            self.wrong_obj_range_step = 0.02
            self.max_obj_range = 0.3
        else:
            self.wrong_obj_range = 0
            self.max_obj_range = 0.2

        self.challenge = challenge

    def increase_difficulty(self):
        self.obj_range += self.obj_range_step
        self.wrong_obj_range -= self.wrong_obj_range_step
        
        if self.obj_range > self.max_obj_range:
            self.obj_range = self.max_obj_range
            if self.wrong_obj_range < 0:
                self.wrong_obj_range = 0
                return True
        else:
            if self.wrong_obj_range < 0:
                self.wrong_obj_range = 0
        self.difficulty += 1
        return False

    def _sample_colors(self):
        #              Block0 Block1 Block2
        #                |      |     |
        blockcolors = [GREEN, BLUE, GREY]
        #np.random.shuffle(blockcolors)
        #     Gripper Table 
        #        |      |   
        return [GREY, GREY] + blockcolors

    def set_test(self):
        self._randomize_objects(True)
        self.goal = self._sample_goal().copy()
        return self._get_obs()

    def _randomize_objects(self, test=False):
        object_xpos = self.initial_gripper_xpos[:2]
        # Get the relevant parameters for setting up the environment
        # If we are testing, then use the hardest set of parameters 
        if test or self.challenge:
            obj_range = self.max_obj_range
            wrong_obj_range = 0
        else:
            obj_range = self.obj_range
            wrong_obj_range = self.wrong_obj_range
        if self.challenge:
            max_wrong_obj_range = 0.04
            min_obj_range = 0.15
        else:
            min_obj_range = MIN_BLOCK_DIST
            max_wrong_obj_range = self.max_obj_range
        # Find the block indices corresponding to each color
        for i in range(3):
            if self.obj_colors[i+2] == BLUE:
                blue = i
            elif self.obj_colors[i+2] == GREEN:
                green = i
            else:
                # In the choose environment this is the grey block
                # and in the fail environment this is the red block
                wrong = i
        # Set the position of the first block (blue)
        do = True
        while do:
            #print(self.initial_gripper_xpos[:2])
            object_xpos = (self.initial_gripper_xpos[:2] + 
                self.np_random.uniform(-obj_range/2, obj_range/2, size=2))
            do = out_of_table(object_xpos)
        object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(blue))
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object{}:joint'.format(blue), object_qpos)

        object0_pos = object_xpos
        # Set the position of the second block (green)
        do = True
        while do:
            direction = np.random.normal(size=2)
            direction = direction / np.linalg.norm(direction)
            mag = np.random.uniform(min_obj_range, obj_range)
            object_xpos = (object0_pos + direction * mag)
            do = out_of_table(object_xpos)
        object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(green))
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object{}:joint'.format(green), object_qpos)
        # Set the position of the third block (grey/red)
        # For the first difficulty level we are going to keep third block fixed in a corner
        # if self.difficulty == 0:
        #     return
        object1_pos = object_xpos
        center_pos = (object0_pos + object_xpos) / 2.0
        do = True
        while do:
            direction = np.random.normal(size=2)
            direction = direction / np.linalg.norm(direction)
            mag = np.random.uniform(wrong_obj_range, max_wrong_obj_range)
            object_xpos = (center_pos + direction * mag)
            do = (out_of_table(object_xpos) 
                or np.linalg.norm(object_xpos - object0_pos) < MIN_BLOCK_DIST
                or np.linalg.norm(object_xpos - object1_pos) < MIN_BLOCK_DIST)
        object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(wrong))
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object{}:joint'.format(wrong), object_qpos)

class BlocksTouchVariationEnv(BlocksEnv):
    """An environment in which the gripper has to make 
    blocks of the right colors touch"""
    def __init__(self, *args, **kwargs):
        self.max_num_blocks = 4
        self.initial_qpos = kwargs['initial_qpos']

        super(BlocksTouchVariationEnv, self).__init__(*args, **kwargs, 
            num_blocks=self.max_num_blocks)
        list_models = ['fetch/2blocks.xml', 
                       'fetch/3blocks.xml', 
                       'fetch/4blocks.xml']
        self.sims = []
        self.initial_states = []
        self.sim_id = 0
        self.viewers = [None for _ in list_models]
        self.set_up = [False for _ in list_models]

        def valid_key(key, num_grey):
            if "object" in key:
                index = int(key[6:key.find(":")])
                return index < (num_grey + 2)
            else:
                return True

        for i,model_path in enumerate(list_models):

            if model_path.startswith('/'):
                fullpath = model_path
            else:
                fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
            if not os.path.exists(fullpath):
                raise IOError('File {} does not exist'.format(fullpath))

            model = mujoco_py.load_model_from_path(fullpath)
            sim = mujoco_py.MjSim(model, nsubsteps=kwargs['n_substeps'])
            self.sims.append(sim)
            self.sim = sim
            initial_qpos = {key:value for key, value in self.initial_qpos.items() if valid_key(key, i)}
            self._env_setup(initial_qpos=initial_qpos)
            self.initial_states.append(copy.deepcopy(sim.get_state()))

        self.obj_range = 0.08
        self.obj_range_step = 0.025
        self.max_obj_range = 0.2

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        object_pos, object_rot, object_velp, object_velr, object_rel_pos = [], [], [], [], []

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        num_blocks = self.num_objs - 2
        obs = [num_blocks]
        block_features = None
        for i in range(num_blocks):
            obj_name = 'object{}'.format(i)
            temp_pos = self.sim.data.get_site_xpos(obj_name)
            # rotations
            temp_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_name))
            # velocities
            temp_velp = self.sim.data.get_site_xvelp(obj_name) * dt
            temp_velr = self.sim.data.get_site_xvelr(obj_name) * dt
            # gripper state
            temp_rel_pos = temp_pos - grip_pos
            temp_velp -= grip_velp

            block_obs = np.concatenate([temp_pos.ravel(), 
                temp_rel_pos.ravel(), temp_rot.ravel(),
                temp_velp.ravel(), temp_velr.ravel(), 
                one_hot_color(self.obj_colors[i+2])
                ])

            block_features = block_obs.shape[0]

            obs = np.concatenate([obs, block_obs])

        padding = np.zeros(block_features * (self.max_num_blocks - num_blocks))
        obs = np.concatenate([obs, padding])

        assert(self._check_goal())
        achieved_goal = self.achieved_goal.copy().ravel()
        max_goal_len = (self.max_num_blocks + 2) ** 2
        achieved_goal = np.append(achieved_goal, 
            np.zeros(max_goal_len - achieved_goal.shape[0]))
        desired_goal = np.append(self.goal.copy(), 
            np.zeros(max_goal_len - self.goal.shape[0]))

        obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel, obs])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy(),
        }

    def increase_difficulty(self):
        self.obj_range += self.obj_range_step
        if self.obj_range > self.max_obj_range:
            self.obj_range = self.max_obj_range
            return True
        else:
            self.difficulty += 1
            return False

    def _sample_colors(self):
        #              Block0 Block1 
        #                |      |   
        blockcolors = [GREEN, BLUE] + [GREY] * (self.num_objs - 4)
        #np.random.shuffle(blockcolors)
        #     Gripper Table 
        #        |      |   
        return [GREY, GREY] + blockcolors

    def set_test(self):
        # self._randomize_objects(True)
        # self.goal = self._sample_goal().copy()
        return self._get_obs()

    def _reset_sim(self, test=False):
        num_grey = self.np_random.randint(len(self.sims))
        # 2 colored blocks + gripper and table + grey blocks
        self.num_objs = 4 + num_grey
        self.sim = self.sims[num_grey]
        self.id2obj = [self._geom2objid(i) for i in range(self.sim.model.ngeom)]
        self.sim_id = num_grey

        def valid_key(key):
            if "object" in key:
                index = int(key[6:key.find(":")])
                return index < (num_grey + 2)
            else:
                return True

        initial_qpos = {key:value for key, value in self.initial_qpos.items() if valid_key(key)}
        
        self.achieved_goal = -1 * np.ones([self.max_num_blocks + 2, self.max_num_blocks + 2])
        # Randomize colors for each object
        self.obj_colors = self._sample_colors()
        # if not self.set_up[num_grey]:
        #     self._env_setup(initial_qpos=initial_qpos)
        #     self.set_up[num_grey] = True

        self.sim.set_state(copy.deepcopy(self.initial_states[num_grey]))
        # Randomize start position of each object
        self._randomize_objects(test)
        self.sim.forward()

        if self.viewer != None:
            self.viewer.update_sim(self.sim)

        self.has_succeeded = False
        return True

    def _randomize_objects(self, test=False):
        object_xpos = self.initial_gripper_xpos[:2]

        num_blocks = self.num_objs - 2

        if test:
            obj_range = self.max_obj_range
        else:
            obj_range = self.obj_range
        
        min_obj_range = MIN_BLOCK_DIST

        # Find the block indices corresponding to each color
        for i in range(num_blocks):
            if self.obj_colors[i+2] == BLUE:
                blue = i
            elif self.obj_colors[i+2] == GREEN:
                green = i
            elif self.obj_colors[i+2] == RED:
                bad = i

        # Set the position of the first block (blue)
        do = True
        while do:
            #print(self.initial_gripper_xpos[:2])
            object_xpos = (self.initial_gripper_xpos[:2] + 
                self.np_random.uniform(-obj_range/2, obj_range/2, size=2))
            do = out_of_table(object_xpos)

        object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(blue))
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object{}:joint'.format(blue), object_qpos)
        object0_pos = object_xpos
        # Set the position of the second block (green)
        do = True
        while do:
            direction = np.random.normal(size=2)
            direction = direction / np.linalg.norm(direction)
            mag = np.random.uniform(min_obj_range, obj_range)
            object_xpos = (object0_pos + direction * mag)
            do = out_of_table(object_xpos)
        object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(green))
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object{}:joint'.format(green), object_qpos)
        obj_positions = [object0_pos, object_xpos]
        for i in range(num_blocks):
            if blue == i or green == i:
                continue

            do = True
            while do:
                do = False
                object_xpos = self._sample_from_table()
                # The for-else statement!
                for object_pos in obj_positions:
                    if np.linalg.norm(object_xpos - object_pos) < MIN_BLOCK_DIST:
                        do = True
                        break
                else:
                    do = out_of_table(object_xpos) 
        
            object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)
            obj_positions.append(object_xpos)

class ToppleTowerEnv(BlocksEnv):
    """A very simple environment in which the gripper has to touch the block"""
    def __init__(self, *args, **kwargs):
        super(ToppleTowerEnv, self).__init__(*args, **kwargs, num_blocks=4)
        # utils.EzPickle.__init__(self)
    
    def _sample_colors(self):
        #     Gripper Table Block0 Block1 Block2 Block3
        #        |      |     |      |      |     |
        return [RED, GREEN, GREY,  GREY,  GREY,  BLUE]

    def _randomize_objects(self, test=False):
        object_xpos = self.initial_gripper_xpos[:2]
        # Set the position of the first block
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = (self.initial_gripper_xpos[:2] + 
                self.np_random.uniform(-self.obj_range, self.obj_range, size=2))
        for i in range(4):
            object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)
        
        

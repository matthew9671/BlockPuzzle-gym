import numpy as np

from gym.envs.robotics import rotations, utils
from gym_blocks.envs import robot_env
from mujoco_py.modder import TextureModder

# Code for the colors used in puzzle solving
GREY = 0
RED = 1
GREEN = 2
BLUE = 3
COLORS_RGB = [(50, 50, 50), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

MIN_BLOCK_DIST = 0.075

TABLE_X = 0.25
TABLE_Y = 0.35

def out_of_table(pos):
    return abs(pos[0]) > TABLE_X or abs(pos[1]) > TABLE_Y

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
        super(BlocksEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # For curriculum learning
    def increase_difficulty(self):
        raise NotImplementedError()

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
        d = np.sum(achieved_goal * goal, axis=-1)
        c = np.count_nonzero(goal, axis=-1)
        return -(d != c).astype(np.float32)

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

            object_pos = np.concatenate([object_pos, temp_pos.ravel()])
            object_rot = np.concatenate([object_rot, temp_rot.ravel()])
            object_velp = np.concatenate([object_velp, temp_velp.ravel()])
            object_velr = np.concatenate([object_velr, temp_velr.ravel()])
            object_rel_pos = np.concatenate([object_rel_pos, temp_rel_pos.ravel()])
    
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        # achieved_goal = np.squeeze(object_pos.copy())
        assert(self._check_goal())
        achieved_goal = self.achieved_goal.copy().ravel()

        # Does order matter here? I think not ...
        # obs = np.concatenate([
        #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        #     object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        # ])
        obs = np.concatenate([
            grip_pos, gripper_state, grip_velp, gripper_vel, object_pos.ravel(), 
            object_rel_pos.ravel(), object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(),
        ])

        # print(obs.shape)

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
        # for i in range(self.num_objs):
        #     object_xpos = self.initial_gripper_xpos[:2]
        #     while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #         object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        #     object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos
        #     self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)
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
        r = self.compute_reward(achieved_goal, desired_goal, None)
        if r == 0:
            self.has_succeeded = True
        return self.has_succeeded

    def _env_setup(self, initial_qpos):
        if self.id2obj == None:
            # Store the ids of all the object geoms by name
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
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

        # Change the colors to match the success conditions
        self.color_modder = TextureModder(self.sim)
        for i in range(self.sim.model.ngeom):
            obj_id = self.id2obj[i]
            if obj_id != None:
                name = self.sim.model.geom_id2name(i)
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

    def increase_difficulty(self):
        self.obj_range += self.obj_range_step
        self.obj_range = min(self.max_obj_range, self.obj_range)

    def _sample_colors(self):
        #     Gripper Table Block0 Block1
        #        |      |     |      |
        return [GREY, GREY, GREEN, BLUE]

    def set_test(self):
        self._randomize_objects(True)
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
    def __init__(self, curriculum, *args, **kwargs):
        super(BlocksTouchEnv, self).__init__(*args, **kwargs, num_blocks=2)
        # utils.EzPickle.__init__(self)
        if curriculum:
            self.obj_range = 0.08
            self.obj_range_step = 0.025
            self.wrong_obj_range = 0.1
            self.wrong_obj_range_step = 0.01
            self.max_obj_range = 0.2
        else:
            self.wrong_obj_range = 0
            self.max_obj_range = 0.2

    def increase_difficulty(self):
        self.obj_range += self.obj_range_step
        self.obj_range = min(self.max_obj_range, self.obj_range)
        self.wrong_obj_range -= self.wrong_obj_range_step
        self.wrong_obj_range = max(0, self.wrong_obj_range)

    def _sample_colors(self):
        #              Block0 Block1 Block2
        #                |      |     |
        blockcolors = [GREEN, BLUE, GREY]
        np.random.shuffle(blockcolors)
        #     Gripper Table 
        #        |      |   
        return [GREY, GREY] + blockcolors

    def set_test(self):
        self._randomize_objects(True)
        return self._get_obs()

    def _randomize_objects(self, test=False):
        object_xpos = self.initial_gripper_xpos[:2]
        # Get the relevant parameters for setting up the environment
        # If we are testing, then use the hardest set of parameters 
        if test:
            obj_range = self.max_obj_range
            wrong_obj_range = 0
        else:
            obj_range = self.obj_range
            wrong_obj_range = self.wrong_obj_range
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
            object_xpos = (self.initial_gripper_xpos[:2] + 
                self.np_random.uniform(-obj_range, obj_range, size=2))
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
            mag = np.random.uniform(MIN_BLOCK_DIST, obj_range)
            object_xpos = (object0_pos + direction * mag)
            do = out_of_table(object_xpos)
        object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(green))
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object{}:joint'.format(green), object_qpos)
        # Set the position of the second block (grey/red)
        center_pos = (object0_pos + object_xpos) / 2.0
        do = True
        while do:
            direction = np.random.normal(size=2)
            direction = direction / np.linalg.norm(direction)
            mag = np.random.uniform(wrong_obj_range, self.max_obj_range)
            object_xpos = (center_pos + direction * mag)
            do = out_of_table(object_xpos)
        object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(wrong))
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object{}:joint'.format(wrong), object_qpos)

class BlocksTouchAttentionEnv(BlocksTouchEnv):
    # The observation is given block by block
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
                temp_velp.ravel(), temp_velr.ravel(),
                ])

            obs = np.concatenate([obs, block_obs])

            # object_pos = np.concatenate([object_pos, temp_pos.ravel()])
            # object_rot = np.concatenate([object_rot, temp_rot.ravel()])
            # object_velp = np.concatenate([object_velp, temp_velp.ravel()])
            # object_velr = np.concatenate([object_velr, temp_velr.ravel()])
            # object_rel_pos = np.concatenate([object_rel_pos, temp_rel_pos.ravel()])

        # achieved_goal = np.squeeze(object_pos.copy())
        assert(self._check_goal())
        achieved_goal = self.achieved_goal.copy().ravel()

        obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel, obs])
        # Does order matter here? I think not ...
        # obs = np.concatenate([
        #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        #     object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        # ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

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
        
        
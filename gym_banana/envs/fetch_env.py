import numpy as np

from gym.envs.robotics import rotations, utils
from gym_banana.envs import robot_env

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


# class FetchEnv(robot_env.RobotEnv):
#     """Superclass for all Fetch environments.
#     """

#     def __init__(
#         self, model_path, n_substeps, gripper_extra_height, block_gripper,
#         has_object, target_in_the_air, target_offset, obj_range, target_range,
#         distance_threshold, initial_qpos, reward_type,
#     ):
#         """Initializes a new Fetch environment.

#         Args:
#             model_path (string): path to the environments XML file
#             n_substeps (int): number of substeps the simulation runs on every call to step
#             gripper_extra_height (float): additional height above the table when positioning the gripper
#             block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
#             has_object (boolean): whether or not the environment has an object
#             target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
#             target_offset (float or array with 3 elements): offset of the target
#             obj_range (float): range of a uniform distribution for sampling initial object positions
#             target_range (float): range of a uniform distribution for sampling a target
#             distance_threshold (float): the threshold after which a goal is considered achieved
#             initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
#             reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
#         """
#         self.gripper_extra_height = gripper_extra_height
#         self.block_gripper = block_gripper
#         self.has_object = has_object
#         self.target_in_the_air = target_in_the_air
#         self.target_offset = target_offset
#         self.obj_range = obj_range
#         self.target_range = target_range
#         self.distance_threshold = distance_threshold
#         self.reward_type = reward_type

#         super(FetchEnv, self).__init__(
#             model_path=model_path, n_substeps=n_substeps, n_actions=4,
#             initial_qpos=initial_qpos)

#     # GoalEnv methods
#     # ----------------------------

#     def compute_reward(self, achieved_goal, goal, info):
#         # Compute distance between goal and the achieved goal.
#         d = goal_distance(achieved_goal, goal)
#         if self.reward_type == 'sparse':
#             return -(d > self.distance_threshold).astype(np.float32)
#         else:
#             return -d

#     # RobotEnv methods
#     # ----------------------------

#     def _step_callback(self):
#         if self.block_gripper:
#             self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
#             self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
#             self.sim.forward()

#     def _set_action(self, action):
#         assert action.shape == (4,)
#         action = action.copy()  # ensure that we don't change the action outside of this scope
#         pos_ctrl, gripper_ctrl = action[:3], action[3]

#         pos_ctrl *= 0.05  # limit maximum change in position
#         rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
#         gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
#         assert gripper_ctrl.shape == (2,)
#         if self.block_gripper:
#             gripper_ctrl = np.zeros_like(gripper_ctrl)
#         action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

#         # Apply action to simulation.
#         utils.ctrl_set_action(self.sim, action)
#         utils.mocap_set_action(self.sim, action)

#     def _get_obs(self):
#         # positions
#         grip_pos = self.sim.data.get_site_xpos('robot0:grip')
#         dt = self.sim.nsubsteps * self.sim.model.opt.timestep
#         grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
#         robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
#         if self.has_object:
#             object_pos = self.sim.data.get_site_xpos('object0')
#             # rotations
#             object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
#             # velocities
#             object_velp = self.sim.data.get_site_xvelp('object0') * dt
#             object_velr = self.sim.data.get_site_xvelr('object0') * dt
#             # gripper state
#             object_rel_pos = object_pos - grip_pos
#             object_velp -= grip_velp
#         else:
#             object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
#         gripper_state = robot_qpos[-2:]
#         gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

#         if not self.has_object:
#             achieved_goal = grip_pos.copy()
#         else:
#             achieved_goal = np.squeeze(object_pos.copy())
#         obs = np.concatenate([
#             grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
#             object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
#         ])

#         return {
#             'observation': obs.copy(),
#             'achieved_goal': achieved_goal.copy(),
#             'desired_goal': self.goal.copy(),
#         }

#     def _viewer_setup(self):
#         body_id = self.sim.model.body_name2id('robot0:gripper_link')
#         lookat = self.sim.data.body_xpos[body_id]
#         for idx, value in enumerate(lookat):
#             self.viewer.cam.lookat[idx] = value
#         self.viewer.cam.distance = 2.5
#         self.viewer.cam.azimuth = 132.
#         self.viewer.cam.elevation = -14.

#     def _render_callback(self):
#         # Visualize target.
#         sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
#         site_id = self.sim.model.site_name2id('target0')
#         self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
#         self.sim.forward()

#     def _reset_sim(self):
#         self.sim.set_state(self.initial_state)

#         # Randomize start position of object.
#         if self.has_object:
#             object_xpos = self.initial_gripper_xpos[:2]
#             while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
#                 object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
#             object_qpos = self.sim.data.get_joint_qpos('object0:joint')
#             assert object_qpos.shape == (7,)
#             object_qpos[:2] = object_xpos
#             self.sim.data.set_joint_qpos('object0:joint', object_qpos)

#         self.sim.forward()
#         return True

#     def _sample_goal(self):
#         if self.has_object:
#             goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
#             goal += self.target_offset
#             goal[2] = self.height_offset
#             if self.target_in_the_air and self.np_random.uniform() < 0.5:
#                 goal[2] += self.np_random.uniform(0, 0.45)
#         else:
#             goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
#         return goal.copy()

#     def _is_success(self, achieved_goal, desired_goal):
#         d = goal_distance(achieved_goal, desired_goal)
#         return (d < self.distance_threshold).astype(np.float32)

#     def _env_setup(self, initial_qpos):
#         for name, value in initial_qpos.items():
#             self.sim.data.set_joint_qpos(name, value)
#         utils.reset_mocap_welds(self.sim)
#         self.sim.forward()

#         # Move end effector into position.
#         gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
#         gripper_rotation = np.array([1., 0., 1., 0.])
#         self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
#         self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
#         for _ in range(10):
#             self.sim.step()

#         # Extract information for sampling goals.
#         self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
#         if self.has_object:
#             self.height_offset = self.sim.data.get_site_xpos('object0')[2]

NUM_BLOCKS = 2

class FetchBlocksEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
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
        # The number of blocks present in the scene
        # Now this number is fixed
        self.num_blocks = NUM_BLOCKS
        # This is the goal vector that we define for this task
        self.achieved_goal = -1 * np.ones([self.num_blocks, self.num_blocks])

        super(FetchBlocksEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

        # Store the ids of all the object geoms by name
        self.id2obj = [self._geom2objid(i) for i in range(self.sim.model.ngeom)]
        # print(self.id2obj)

    def _geom2objid(self, i):
        name = self.sim.model.geom_id2name(i)
        if name != None and "object" in name:
            return int(name[6:])
        else:
            return None

    def _check_goal(self):
        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
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
        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
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

        for i in range(self.num_blocks):
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

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of each object.
        # Actually maybe we want to initialize them in a fixed configuration
        # for i in range(self.num_blocks):
        #     object_xpos = self.initial_gripper_xpos[:2]
        #     while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #         object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        #     object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos
        #     self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

        # for name, value in initial_qpos.items():
        #     if name == 
        #     self.sim.data.set_joint_qpos(name, value)

        self.sim.forward()
        return True

    def _sample_goal(self):
        # r, g, b = np.random.choice(self.num_blocks, 3, replace=False)
        # Sanity check: two blocks! No failure conditions!
        g, b = np.random.choice(self.num_blocks, 2, replace=False)
        r = -1

        goal = []
        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                if (i == r and j == b) or (i == b and j == r):
                    goal.append(-1)
                elif (i == g and j == b) or (i == b and j == g):
                    goal.append(1)
                else:
                    goal.append(0)
        return np.asarray(goal)

    def _is_success(self, achieved_goal, desired_goal):
        r = self.compute_reward(achieved_goal, desired_goal, None)
        return r == 0

    def _env_setup(self, initial_qpos):
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

import numpy as np
import os
import itertools

from gym_object_manipulation import rotations, robot_env, utils

ASYMMETRIC = False


def objects_distance(x, y):
    """
    A function that returns the euclidean distance between two objects x and y
    """
    assert x.shape == y.shape
    return np.linalg.norm(x - y)


def signed_distance(x, y):
    """
    A function that returns the signed distance between two objects x and y
    """
    """assert x.shape == y.shape
    if -0.03  < x[2] - y[2] < 0.03:
        return np.linalg.norm(x[:2] - y[:2]) * np.sign(x[1] - y[1])  # Positive if x on the right of y
    else: 
        return -np.linalg.norm(x[:2] - y[:2])"""
    return np.linalg.norm(x[:2] - y[:2]) * np.sign(x[1] - y[1])


def is_grasped(body, obj):
    """
    A temporary function that determines whether obj is grasped by the gripper or not
    Testing a supplementary unary predicate to learn how to grasp before arranging
    """
    if np.linalg.norm(body[:2] - obj[:2]) <= 0.005 and body[2] - obj[2] <= 0.03 and body[3] + body[4] <= 0.06:
        return 1
    return 0


def above(x, y):
    """
    A function that returns whether the object x is above y
    eps is a tolerance parameter that tackles the simulation precision
    """
    """eps = 0.001
    above = 0.05 - eps < (x[2] - y[2]) < 0.05 + eps
    return above and 0 <= np.linalg.norm(x[:2] - y[:2]) < 0.025 * np.sqrt(2) - eps"""
    return np.linalg.norm(x[:2] - y[:2]) < 0.04 and 0.06 > x[2] - y[2] > 0.01
    #return np.linalg.norm(x[:2] - y[:2]) < 0.07 and 0.06 > np.abs(x[2] - y[2]) > 0.01


def below(x, y):
    return np.linalg.norm(x[:2] - y[:2]) < 0.07 and x[2] < y[2]


def in_the_air(x):
    """
    A function that returns whether the object x is in the air or not
    :param x: array of object cartezian coordinates
    :return: Boolean
    """
    return (x[2] - 0.425 >= 0.04)


def inside_zone(x, zone):
    """
    A function that returns whether an object x is inside zone or not
    :param x:
    :param zone:
    :return:
    """
    return abs(x[0]- zone[0]) < 0.07 and abs(x[1] - zone[1]) < 0.1


class FetchManipulateEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, num_blocks, n_substeps, gripper_extra_height, block_gripper,
        target_in_the_air, target_offset, obj_range, target_range, predicate_threshold,
            initial_qpos, reward_type, predicates,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            num_blocks: number of block objects in environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        
        self.target_goal = None
        self.num_blocks = num_blocks
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.predicate_threshold = predicate_threshold
        self.predicates = predicates
        self.num_predicates = len(self.predicates)
        self.guide = False
        self.reward_type = reward_type

        self.goal_size = 0

        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]

        self.location_record = None
        self.location_record_write_dir = None
        self.location_record_prefix = None
        self.location_record_file_number = 0
        self.location_record_steps_recorded = 0
        self.location_record_max_steps = 2000

        super(FetchManipulateEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # Heatmap Generation
    # ----------------------------

    def set_location_record_name(self, write_dir, prefix):
        if not os.path.exists(write_dir):
            os.makedirs(write_dir, exist_ok=True)

        self.flush_location_record()
        # if not self.record_write_dir:
        #     self.save_heatmap_picture(os.path.join(write_dir,'level.png'))
        self.location_record_write_dir = write_dir
        self.location_record_prefix = prefix
        self.location_record_file_number = 0

        return True

    def flush_location_record(self, create_new_empty_record=True):
        if self.location_record is not None and self.location_record_steps_recorded > 0:
            write_file = os.path.join(self.location_record_write_dir,"{}_{}".format(self.location_record_prefix,
                                                                           self.location_record_file_number))
            np.save(write_file, self.location_record[:self.location_record_steps_recorded])
            self.location_record_file_number += 1
            self.location_record_steps_recorded = 0

        if create_new_empty_record:
            self.location_record = np.empty(shape=(self.location_record_max_steps, 3), dtype=np.float32)

    def log_location(self, location):
        if self.location_record is not None:
            self.location_record[self.location_record_steps_recorded] = location
            self.location_record_steps_recorded += 1

            if self.location_record_steps_recorded >= self.location_record_max_steps:
                self.flush_location_record()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        assert self.reward_type in ['sparse', 'incremental']
        if self.reward_type == 'incremental':
            # Using incremental reward for each correct predicate
            reward = np.sum(achieved_goal == goal).astype(np.float32) - self.num_blocks
        else:
            # Using sparse reward
            reward = (achieved_goal == goal).all().astype(np.float32)
        return reward

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.log_location(location=self.sim.data.get_site_xpos('robot0:grip'))
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

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
    
    def _get_configuration(self, body, positions):
        close_config = np.array([])
        above_config = np.array([])
        below_config = np.array([])
        right_close_config = np.array([])
        left_close_config = np.array([])
        if "close" in self.predicates:
            if not ASYMMETRIC:
                object_combinations = itertools.combinations(positions, 2)
            else:
                if self.num_blocks == 3:
                    object_combinations = [(positions[0], positions[1]), (positions[1], positions[0]),
                                           (positions[0], positions[2]),
                                           (positions[2], positions[0]), (positions[1], positions[2]),
                                           (positions[2], positions[1])]
                else:
                    object_combinations = [(positions[0], positions[1]), (positions[1], positions[0])]
            object_rel_distances = np.array([objects_distance(obj[0], obj[1]) for obj in object_combinations])

            close_config = np.array([(distance <= self.predicate_threshold).astype(np.float32)
                                     for distance in object_rel_distances])
            #close_config = np.array([is_grasped(body, position) for position in positions])
        if "above" in self.predicates:
            if self.num_blocks == 3:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0]), (positions[0], positions[2]),
                                       (positions[2], positions[0]), (positions[1], positions[2]), (positions[2], positions[1])]
                #object_permutations = itertools.combinations(positions, 2)
            else:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0])]

            above_config = np.array([int(above(obj[0], obj[1])) for obj in object_permutations]).astype(np.float32) # Changed is_on to close for testing
            #above_config = np.zeros(6)
            #above_config = np.array([(self.predicate_threshold >= objects_distance(obj[0], obj[1]) > 0).astype(np.float32)
            #                         - (objects_distance(obj[0], obj[1]) < 0 or objects_distance(obj[0], obj[1]) > self.predicate_threshold).astype(np.float32)
            #                         for obj in object_permutations])
        if "below" in self.predicates:
            if self.num_blocks == 3:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0]), (positions[0], positions[2]),
                                       (positions[2], positions[0]), (positions[1], positions[2]), (positions[2], positions[1])]
            else:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0])]

            below_config = np.array([int(below(obj[0], obj[1])) for obj in object_permutations]).astype(np.float32)
        if "right_close" in self.predicates:
            if self.num_blocks == 3:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0]),
                                       (positions[0], positions[2]), (positions[2], positions[0]),
                                       (positions[1], positions[2]), (positions[2], positions[1])]
            else:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0])]
            right_close_config = np.array(
                [(self.predicate_threshold >= signed_distance(obj[0], obj[1]) > 0).astype(np.float32)
                 for obj in object_permutations])
        if "left_close" in self.predicates:
            if self.num_blocks == 3:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0]),
                                       (positions[0], positions[2]), (positions[2], positions[0]),
                                       (positions[1], positions[2]), (positions[2], positions[1])]
            else:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0])]
            left_close_config = np.array([(self.predicate_threshold >= signed_distance(obj[1], obj[0]) > 0).astype(np.float32)
                                     for obj in object_permutations])
        
        res = np.concatenate([close_config, right_close_config, left_close_config, above_config, below_config])
        #res = np.array([in_the_air(obj).astype(np.float32) for obj in positions])
        return res

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])

        objects_positions = []

        for i in range(self.num_blocks):

            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            # gripper state
            object_i_rel_pos = object_i_pos - grip_pos
            object_i_velp -= grip_velp

            obs = np.concatenate([
                obs,
                object_i_pos.ravel(),
                object_i_rel_pos.ravel(),
                object_i_rot.ravel(),
                object_i_velp.ravel(),
                object_i_velr.ravel()
            ])

            objects_positions = np.concatenate([
                objects_positions, object_i_pos.ravel()
            ])

        objects_positions = objects_positions.reshape(self.num_blocks, 3)
        object_combinations = itertools.combinations(objects_positions, 2)
        object_rel_distances = np.array([objects_distance(obj[0], obj[1]) for obj in object_combinations])

        self.goal_size = len(object_rel_distances)
        
        achieved_goal = self._get_configuration(obs, objects_positions)
        """if self.num_predicates == 2:
            achieved_goal = self._get_configuration(objects_positions)
        else:    
            achieved_goal = np.array([(distance <= self.predicate_threshold).astype(np.float32) for distance in object_rel_distances])"""

        achieved_goal = np.squeeze(achieved_goal)

        obs = np.concatenate([obs, achieved_goal])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.target_goal.copy(),
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
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        # print("sites offset: {}".format(sites_offset[0]))
        #for i in range(self.num_blocks):

            #site_id = self.sim.model.site_name2id('target{}'.format(i))
            #self.sim.model.site_pos[site_id] = self.goal[i*3:(i+1)*3] - sites_offset[i]

        self.sim.forward()

    def _grasp(self, obs, target_idx):
        success = False
        itr = 0
        observation = obs['observation']
        # Make sure to get high enough not to hit objects
        for _ in range(10):
            next_obs, r, d, info = self.step([0, 0, 1, 1])
            observation = next_obs['observation']
        # Reach object and grasp it
        while not success and itr < 20:
            action = np.concatenate((7 * (-observation[:3] + observation[10 + 15 * target_idx:13 + 15 * target_idx]), np.ones(1)))
            if np.linalg.norm(-observation[:3] + observation[10 + 15 * target_idx:13 + 15 * target_idx]) < 0.005:
                for _ in range(15):
                    action = [0., 0., 0.4, -1]
                    next_obs, r, d, info = self.step(action)
                    observation = next_obs['observation']
                    itr += 1
                success = True
            else:
                next_obs, r, d, info = self.step(action)
                observation = next_obs['observation']
                itr += 1
        obs['observation'] = observation
        return obs

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.

        self.target_goal = self._sample_goal()

        self.sim.set_state(self.initial_state)

        # temp = [[-1, 0], [1, 0]]
        # a = [0.425, 0.475, 0.525]
        
        # Guide learning by generating a stack in initialization
        if "above" in self.predicates or "right_close" in self.predicates:
            self.guide = True

        if self.guide:
            stack_level_proba = np.random.uniform()
            if stack_level_proba > 0.8:
                stack = list(np.random.choice([i for i in range(self.num_blocks)], 3, replace=False))
                z_stack = [0.525, 0.475, 0.425]
            else:
                stack = list(np.random.choice([i for i in range(self.num_blocks)], 2, replace=False))
                z_stack = [0.475, 0.425]
            k = np.random.uniform()
            if k < 0.9:
                for i, obj_name in enumerate(self.object_names):
                    object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                    assert object_qpos.shape == (7,)
                    object_qpos[2] = 0.425
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range,
                                                                                         size=2)
                    object_qpos[:2] = object_xpos

                    self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
            else:
                temp_rand = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                for i, obj_name in enumerate(self.object_names):
                    object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                    assert object_qpos.shape == (7,)
                    if i in stack:
                        object_qpos[2] = z_stack[stack.index(i)]
                        object_xpos = self.initial_gripper_xpos[:2] + temp_rand
                        object_qpos[:2] = object_xpos

                    else:
                        object_qpos[2] = 0.425
                        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                             self.obj_range,
                                                                                             size=2)
                        object_qpos[:2] = object_xpos

                    self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
                
        else:
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                object_qpos[2] = 0.425
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
                object_qpos[:2] = object_xpos

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
        self.sim.forward()

        obs = self._get_obs()

        return obs

    def _get_valid(self, l1, l2):
        for i in range(len(l2) // 2):
            if l2[2 * i] == 1. or l2[2 * i + 1] == 1.:  # If two object above each other then certainly close
                l1[i] = 1.
            try:
                if l2[2 * i] == 1. and l2[2 * i + 2] == 1.:
                    l1 = np.array([1., 1., 1.])
            except IndexError:
                pass
            continue
        return l2, l1

    def _generate_valid_goal(self):
        above_config = np.array([])
        close_config = np.array([])
        right_close_config = np.array([])
        left_close_config = np.array([])
        below_config = np.array([])
        zone_config = np.array([])
        """if "above" in self.predicates:
            # For 3 objects
            if self.num_blocks == 3:
                valid_list = [0, 1, 2, 4, 5, 6, 8, 9, 16, 17, 18, 24, 32, 34, 36, 40]
                above_config = np.array(tuple('{0:06b}'.format(np.random.choice(valid_list)))).astype(np.float32)[::-1]
            #above_config = np.zeros(6)
            # For 2 objects
            else:
                valid_list = [1, 2]
                above_config = np.array(tuple('{0:02b}'.format(np.random.choice(valid_list)))).astype(np.float32)[::-1]"""
        if "close" in self.predicates:
            # ATTEMPT WITH INDEPENDENT PREDICATES
            #close_config = np.random.randint(2, size=self.num_blocks).astype(np.float32)
            close_config = np.array(tuple('{0:06b}'.format(np.random.choice([0, 3, 12, 48, 15, 51, 60, 63])))).astype(np.float32)[::-1]
            """k = np.random.uniform()
            if k < 0.5:
                list1 = [1, 2, 3, 4, 5, 6, 7]
                close_config = np.array(tuple('{0:03b}'.format(np.random.choice(list1)))).astype(np.float32)[::-1]
                above_config = np.zeros(6)
            else:
                close_config = np.zeros(3)
                list2 = [1, 2, 4, 5, 6, 8, 9, 16, 17, 18, 24, 32, 34, 36, 40]
                above_config = np.array(tuple('{0:06b}'.format(np.random.choice(list2)))).astype(np.float32)[::-1]"""
            """if not ASYMMETRIC:
                close_config = np.random.randint(2, size=self.num_blocks).astype(np.float32)
            else:
                close_config = np.array([]).astype(np.float32)
                for _ in range(self.num_blocks):
                    temp = np.random.randint(2)
                    close_config = np.concatenate([close_config, [temp, temp]])
                    del temp
            if "above" in self.predicates and "below" in self.predicates:
                above_config = np.zeros(self.num_blocks * (self.num_blocks - 1))
                below_config = np.zeros(self.num_blocks * (self.num_blocks - 1))
                for i in range(self.num_blocks * (self.num_blocks - 1) // 2):
                    if close_config[2*i] == 1.:
                        proba = np.random.uniform()
                        above_config[2*i] = 1.0 * (proba < 0.5)
                        below_config[2*i + 1] = 1.0 * (proba < 0.5)
                        above_config[2*i + 1] = 1.0 * (1 - proba < 0.5)
                        below_config[2*i] = 1.0 * (1 - proba < 0.5)
            elif "above" in self.predicates:
                if "close" in self.predicates:
                    above_config = np.zeros(self.num_blocks * (self.num_blocks - 1))
                    for i in range(self.num_blocks * (self.num_blocks - 1) // 2):
                        if close_config[i] == 1.:
                            proba = np.random.uniform()
                            above_config[2*i] = 1.0 * (proba < 0.5)
                            above_config[2*i + 1] = 1.0 * (1 - proba < 0.5)"""
        elif "above" in self.predicates:
            above_list = [0, 1, 2, 4, 5, 8, 16, 18, 32, 40, 37, 21, 41, 42, 22, 26]
            above_config = np.array(tuple('{0:06b}'.format(np.random.choice(above_list)))).astype(np.float32)[::-1]
        if "right_close" in self.predicates:
            if self.num_blocks == 3:
                if "close" in self.predicates:  # Build on the close generated goal predicate
                    right_close_config = np.zeros(self.num_blocks * (self.num_blocks - 1))
                    left_close_config = np.zeros(self.num_blocks * (self.num_blocks - 1))
                    above_config = np.zeros(self.num_blocks * (self.num_blocks - 1))
                    below_config = np.zeros(self.num_blocks * (self.num_blocks - 1))
                    for i in range(self.num_blocks * (self.num_blocks - 1) // 2):
                        if close_config[2*i] == 1.:
                            proba = np.random.uniform()
                            right_close_config[2*i] = 1.0 * (proba < 0.25)
                            left_close_config[2*i + 1] = 1.0 * (proba < 0.25)
                            right_close_config[2*i + 1] = 1.0 * (0.25 < proba < 0.5)
                            left_close_config[2*i] = 1.0 * (0.25 < proba < 0.5)
                            above_config[2*i] =  1.0 * (0.5 < proba < 0.75)
                            below_config[2*i + 1] = 1.0 * (0.5 < proba < 0.75)
                            above_config[2*i + 1] = 1.0 * (0.75 < proba )
                            below_config[2*i] = 1.0 * (0.75 < proba)
                else:
                    valid_list = [0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 32, 33, 34, 24, 40, 20, 36, 41, 21, 37, 26, 42, 22]
                    right_close_config = np.array(tuple('{0:06b}'.format(np.random.choice(valid_list)))).astype(np.float32)[::-1]
            # For 2 objects
            else:
                valid_list = [0, 1, 2]
                right_close_config = np.array(tuple('{0:02b}'.format(np.random.choice(valid_list)))).astype(np.float32)[::-1]

        res = np.concatenate([close_config, right_close_config, left_close_config, above_config, below_config])
        return res

    def _sample_goal(self):
        """if self.num_predicates == 2:
            self.target_goal = self._generate_valid_goal()
        else:
            self.target_goal = np.random.randint(2, size=self.goal_size).astype(np.float32)"""
        #self.target_goal = self._generate_valid_goal()
        self.target_goal = np.random.randint(2, size=9).astype(np.float32)
        #self.target_goal = np.concatenate([np.array(tuple('{0:02b}'.format(np.random.choice([0, 1, 2])))).astype(np.float32) for _ in range(self.num_blocks)])
        # self.target_goal = np.array([0.0, 0.0, 0.0])
        
        return self.target_goal

    def set_goal(self, goal):
        self.target_goal = goal

    def _is_success(self, achieved_goal, desired_goal):
        return (achieved_goal == desired_goal).all()

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
    
    '''def reset_goal(self, goal, bucket=0, guide_proba=0, eval=True):
        
        # self.target_goal = self._sample_goal()
        self.target_goal = goal

        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        # prev_x_positions = [goal[:2] for goal in goals]  # Avoids blocks randomly being in goals
        # prev_x_positions = [goals[0][:2]]
        # temp = [[-1, -1], [1, 0], [-1, 1]]

        # Guide learning by generating a stack in initialization
        # stage = np.random.choice(np.arange(5), p=p)

        self.guide = np.random.uniform() < guide_proba

        proba_stack_3 = 0  # P(stack 3 | stack)
        proba_stack = 0
        
        if not eval:
            if bucket == 3 and self.guide:
                proba_stack = 1
                proba_stack_3 = 1
            if bucket == 1 and self.guide:
                proba_stack = 0
            if bucket in [0, 2, 4]:
                proba_stack_3 = 0.7  # P(stack 3 | stack)
                proba_stack = 0.3

        idx_grasp = np.random.choice([i for i in range(self.num_blocks)])

        if np.random.uniform() < proba_stack_3:
            stack = list(np.random.choice([i for i in range(self.num_blocks)], 3, replace=False))
            z_stack = [0.525, 0.475, 0.425]
        else:
            stack = list(np.random.choice([i for i in range(self.num_blocks)], 2, replace=False))
            z_stack = [0.475, 0.425]
        
        if np.random.uniform() < 1 - proba_stack:
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                object_qpos[2] = 0.425
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                     self.obj_range,
                                                                                     size=2)
                object_qpos[:2] = object_xpos

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)

        else:
            temp_rand = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                if i in stack:
                    object_qpos[2] = z_stack[stack.index(i)]
                    object_xpos = self.initial_gripper_xpos[:2] + temp_rand
                    object_qpos[:2] = object_xpos

                else:
                    object_qpos[2] = 0.425
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range,
                                                                                         size=2)
                    object_qpos[:2] = object_xpos

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
            if len(stack) == self.num_blocks:
                idx_grasp = stack[0]
        self.sim.forward()

        obs = self._get_obs()
        if bucket in [0, 2, 3] and self.guide and not eval:
            obs = self._grasp(obs, idx_grasp)
        
        return obs'''

    def reset_goal(self, goal, biased_init=False, eval=True):
        self.target_goal = goal

        self.sim.set_state(self.initial_state)

        # If biased_init==False or during evaluation time, generate blocks on the table with no stacks
        if not biased_init or eval:
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                object_qpos[2] = 0.425
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                     self.obj_range,
                                                                                     size=2)
                object_qpos[:2] = object_xpos

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)

            self.sim.forward()
            obs = self._get_obs()

            return obs

        stack_level_proba = np.random.uniform()
        if stack_level_proba > 0.7:
            stack = list(np.random.choice([i for i in range(self.num_blocks)], 3, replace=False))
            z_stack = [0.525, 0.475, 0.425]
        else:
            stack = list(np.random.choice([i for i in range(self.num_blocks)], 2, replace=False))
            z_stack = [0.475, 0.425]
        k = np.random.uniform()

        idx_grasp = np.random.choice([i for i in range(self.num_blocks)])
        if k < 0.7:
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                object_qpos[2] = 0.425
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                     self.obj_range,
                                                                                     size=2)
                object_qpos[:2] = object_xpos

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
        else:
            temp_rand = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                if i in stack:
                    object_qpos[2] = z_stack[stack.index(i)]
                    object_xpos = self.initial_gripper_xpos[:2] + temp_rand
                    object_qpos[:2] = object_xpos

                else:
                    object_qpos[2] = 0.425
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range,
                                                                                         size=2)
                    object_qpos[:2] = object_xpos

                    idx_grasp = i

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
            if len(stack) == self.num_blocks:
                idx_grasp = stack[0]

        self.sim.forward()
        obs = self._get_obs()
        
        if np.random.uniform() < 0.6:
            obs = self._grasp(obs, idx_grasp)
        return obs

    def reset_init(self, init_config, target_goal):
        self.target_goal = target_goal

        self.sim.set_state(self.initial_state)
        possible_stacks = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
        actual_stacks = possible_stacks[np.where(np.array(init_config[-6:]) == 1.)]
        blocks_set = [False for i in range(self.num_blocks)]
        positions = [None for i in range(self.num_blocks)]
        if actual_stacks.shape[0] > 0:
            # There is at least one stack
            if actual_stacks.shape[0] == 1.:
                # There exactly one stack
                stack = actual_stacks[0]
                z_stack = [0.475, 0.425]
                offset_on_table = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                for i in stack:
                    object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
                    object_qpos[2] = z_stack[np.where(stack == i)[0][0]]
                    object_xpos = self.initial_gripper_xpos[:2] + offset_on_table
                    object_qpos[:2] = object_xpos
                    blocks_set[i] = True
                    positions[i] = object_qpos[:3]
                    self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), object_qpos)
            if actual_stacks.shape[0] > 1.:
                # There are two stacks
                if actual_stacks[0][0] == actual_stacks[1][0]:
                    # One block above two blocks (pyramid)
                    bot_block_1 = actual_stacks[0][1]
                    bot_block_2 = actual_stacks[1][1]
                    top_block = actual_stacks[0][0]
                    pos_on_table_1 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                            size=2)
                    pos_on_table_2 = pos_on_table_1 + [0, 0.05]
                    pos_on_table_3 = pos_on_table_1 + [0, 0.025]
                    z = [0.425, 0.425, 0.475]
                    xy = [pos_on_table_1, pos_on_table_2, pos_on_table_3]
                    for i, block in enumerate([bot_block_1, bot_block_2, top_block]):
                        object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[block]))
                        object_qpos[2] = z[i]
                        object_xpos = xy[i]
                        object_qpos[:2] = object_xpos
                        blocks_set[block] = True
                        positions[i] = object_qpos[:3]
                        self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[block]), object_qpos)
                else:
                    # A stack of 3 blocks
                    if actual_stacks[0][1] == actual_stacks[1][0]:
                        stack = [actual_stacks[0][0], actual_stacks[0][1], actual_stacks[1][1]]
                    else:
                        stack = [actual_stacks[1][0], actual_stacks[1][1], actual_stacks[0][1]]
                    z_stack = [0.525, 0.475, 0.425]
                    offset_on_table = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                    for i in stack:
                        object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
                        object_qpos[2] = z_stack[stack.index(i)]
                        object_xpos = self.initial_gripper_xpos[:2] + offset_on_table
                        object_qpos[:2] = object_xpos
                        blocks_set[i] = True
                        positions[i] = object_qpos[:3]
                        self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), object_qpos)
            # Getting remaining blocks which were not placed
            remain_block = [i for i, x in enumerate(blocks_set) if not x]
            if len(remain_block) > 0:
                if sum(init_config[:3]) == 1:
                    remain_block = remain_block[0]
                    object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[remain_block]))
                    object_qpos[2] = 0.425
                    # Added center to make sure the block stays on the table
                    center = positions[np.where(blocks_set)[0][0]][:2] - self.initial_gripper_xpos[:2]
                    object_xpos = positions[np.where(blocks_set)[0][0]][:2] + \
                                  np.array([-np.sign(center[0]) * 0.1, -np.sign(center[1]) * 0.1])
                    object_qpos[:2] = object_xpos
                    blocks_set[remain_block] = True
                    positions[remain_block] = object_qpos[:3]
                    self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[remain_block]), object_qpos)
                else:
                    # The remaining block is close to the stack
                    remain_block = remain_block[0]
                    object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[remain_block]))
                    object_qpos[2] = 0.425
                    # Added center to make sure the block stays on the table
                    center = positions[np.where(blocks_set)[0][0]][:2] - self.initial_gripper_xpos[:2]
                    object_xpos = positions[np.where(blocks_set)[0][0]][:2] + np.array([0, -np.sign(center[1]) * 0.05])
                    object_qpos[:2] = object_xpos
                    blocks_set[remain_block] = True
                    positions[remain_block] = object_qpos[:3]
                    self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[remain_block]), object_qpos)

        else:
            if sum(init_config[:3]) == 0:
                # All blocks are far
                offset = self.np_random.uniform(0.1, self.obj_range)
                offset1 = self.np_random.uniform(0.1, self.obj_range)
                offsets = [[offset, -offset1], [-offset, offset1], [-offset1, -offset]]
                np.random.shuffle(offsets)
                for i, obj_name in enumerate(self.object_names):
                    object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                    assert object_qpos.shape == (7,)
                    object_qpos[2] = 0.425
                    object_xpos = self.initial_gripper_xpos[:2] + offsets[i]
                    object_qpos[:2] = object_xpos
                    blocks_set[i] = True
                    positions[i] = object_qpos[:3]
                    self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)

            else:
                possible_pairs = np.array([[0, 1], [0, 2], [1, 2]])
                actual_close_pairs = possible_pairs[np.where(np.array(init_config[:3]) == 1.)]
                if len(actual_close_pairs) == 3:
                    # All blocks are close to each other
                    pos_on_table_1 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                            size=2)
                    pos_on_table_2 = pos_on_table_1 + [0, 0.05]
                    pos_on_table_3 = pos_on_table_1 + [0.025, 0.05]
                    xy = [pos_on_table_1, pos_on_table_2, pos_on_table_3]
                    for i, obj_name in enumerate(self.object_names):
                        object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                        object_qpos[2] = 0.425
                        object_xpos = xy[i]
                        object_qpos[:2] = object_xpos
                        blocks_set[i] = True
                        positions[i] = object_qpos[:3]
                        self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
                elif len(actual_close_pairs) == 2:
                    pivot = list(set(actual_close_pairs[0]) & set(actual_close_pairs[1]))[0]
                    indexes = [i for i in itertools.chain(actual_close_pairs[0], actual_close_pairs[1]) if i != pivot]
                    indexes = [pivot] + indexes
                    pos_on_tab_1 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,size=2)
                    pos_on_tab_2 = pos_on_tab_1 + np.array([0., 0.07])
                    pos_on_tab_3 = pos_on_tab_1 - np.array([0., 0.07])
                    xy = [pos_on_tab_1, pos_on_tab_2, pos_on_tab_3]
                    for i, pos in zip(indexes, xy):
                        object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
                        object_qpos[2] = 0.425
                        object_xpos = pos
                        object_qpos[:2] = object_xpos
                        blocks_set[i] = True
                        positions[i] = object_qpos[:3]
                        self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), object_qpos)
                else:
                    for pair in actual_close_pairs:
                        if not blocks_set[pair[0]]:
                            pos_on_table_1 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                                    size=2)
                        else:
                            pos_on_table_1 = positions[pair[0]][:2]
                        pos_on_table_2 = pos_on_table_1 + np.array([0, 0.07])
                        xy = [pos_on_table_1, pos_on_table_2]
                        for i, pos in zip(pair, xy):
                            object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
                            object_qpos[2] = 0.425
                            object_xpos = pos
                            object_qpos[:2] = object_xpos
                            blocks_set[i] = True
                            positions[i] = object_qpos[:3]
                            self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), object_qpos)
                    if sum(blocks_set) != len(blocks_set):
                        remain_block = np.where(np.invert(blocks_set))[0][0]
                        object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[remain_block]))
                        object_qpos[2] = 0.425
                        # Added center to make sure the block stays on the table
                        center = positions[np.where(blocks_set)[0][0]][:2] - self.initial_gripper_xpos[:2]
                        object_xpos = positions[np.where(blocks_set)[0][0]][:2] + \
                                      np.array([-np.sign(center[0])*0.1, -np.sign(center[1])*0.1])
                        object_qpos[:2] = object_xpos
                        blocks_set[remain_block] = True
                        positions[remain_block] = object_qpos[:3]
                        self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[remain_block]), object_qpos)

        self.sim.forward()
        obs = self._get_obs()

        return obs




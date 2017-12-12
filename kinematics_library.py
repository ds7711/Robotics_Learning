# import required modules
import numpy as np


def solve_quadratic_func(a, b, c):
    """
    solve the quadratic function
    :param a:
    :param b:
    :param c:
    :return:
    """
    delta = (b ** 2) - (4 * a * c)
    if delta >= 0:
        solution1 = (-b - np.sqrt(delta)) / (2 * a)
        solution2 = (-b + np.sqrt(delta)) / (2 * a)
        return(solution1, solution2)
    else:
        return(np.inf, np.inf)


def ball2hoop(pos, vel, hoop_pos, gravity):
    """
    calculate the distance between the ball and the hoop
    :param pos:
    :param vel:
    :param hoop_pos:
    :param gravity:
    :return:
    """
    z_diff = hoop_pos[2] - pos[2]
    z_vel = vel[2]
    distance = np.inf
    if z_diff > 0:  # if hoop is above the ball
        if z_vel <= 0:  # if ball is going downward
            xy_diff = hoop_pos - pos
            distance = np.linalg.norm(xy_diff)
        else:
            t_max = -z_vel / gravity  # if the ball can not reach the hoop level
            z_max = pos[2] + z_vel * t_max + 0.5 * gravity * (t_max ** 2)
            if z_max < hoop_pos[2]:
                # return the distance between the initial ball pos and target
                xy_diff = hoop_pos - pos
                distance = np.linalg.norm(xy_diff)
                # the following code encourages throwing from the bottom
                # xy_max = pos[:2] + t_max * vel[:2]
                # xy_diff = hoop_pos[:2] - xy_max
                # distance = np.linalg.norm(xy_diff)
    if distance == np.inf:
        a = 0.5 * gravity
        b = z_vel
        c = -z_diff
        t1, t2 = solve_quadratic_func(a, b, c)
        xy_distance = []
        for tmp_t in [t1, t2]:
            if tmp_t >= 0:
                xy_max = pos[:2] + tmp_t * vel[:2]
                xy_diff = hoop_pos[:2] - xy_max
                tmp_distance = np.linalg.norm(xy_diff)
                xy_distance.append(tmp_distance)
        distance = np.min(xy_distance)
    return(distance)


class BallObj(object):

    def __init__(self, pos, vel, env_obj):
        """
        initialize the ball object
        :param pos:
        :param vel:
        :param env_obj:
        """
        self.pos = np.copy(np.asarray(pos, dtype=np.double))
        self.vel = np.copy(np.asarray(vel, dtype=np.double))
        self.gravity = env_obj.gravity

    def update(self):
        t = 0.001  # set the time interval between two updates
        g = self.gravity  # gravity acceleration
        ground = 0.0  # the ground
        ball_trajectory = []
        while self.pos[2] > ground:
            # we assume the only acceleration caused by gravity is along z axis
            # so only z_dot changes

            ball_trajectory.append(np.copy(self.pos))  # store the trajectory of the ball

            # update position
            self.pos[0] += self.vel[0] * t
            self.pos[1] += self.vel[1] * t
            self.pos[2] += self.vel[2] * t

            self.vel[2] += g * t

        return(np.asarray(ball_trajectory))


class RobotArm(object):

    def __init__(self, link_lengthes, initial_angles, intial_angular_velocities, time_step=1e-3):
        """
        create the robotic manipulator object
        :param link_lengthes:
        :param initial_angles:
        :param intial_angular_velocities:
        :param time_step:
        """
        self.link_lengthes = np.asarray(link_lengthes)
        self.joint_relative_locations = np.asarray([[0, 0, self.link_lengthes[0], 1],
                                                    [0, 0, self.link_lengthes[1], 1],
                                                    [0, self.link_lengthes[2], 0, 1],
                                                    [0, self.link_lengthes[3], 0, 1],
                                                    [0, self.link_lengthes[4], 0, 1],
                                                    [self.link_lengthes[5], 0, 0, 1],
                                                    [self.link_lengthes[6], 0, 0, 1]
                                                    ])
        self.rotation_axises = ["z", "x", "x", "z", "y", "y"]
        self.initial_joint_angles = np.asarray(initial_angles, dtype=np.double)
        self.joint_angles = np.zeros(len(initial_angles), dtype=np.double)
        # self.joint_angles = np.asarray(initial_angles, dtype=np.double)
        self.angular_velocities = np.asarray(intial_angular_velocities, dtype=np.double)
        self.release = 0
        self.num_joints = len(initial_angles)
        self.time = 0
        self.time_step = time_step
        self.joint_angle_limit = 2 * np.pi
        self.joint_vel_limit = 2 * np.pi

        # a list of homogeneous transformation matrix functions
        def r10(q, idx=0):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, 0],
                             [0, 0, 1, l],
                             [0, 0, 0, 1]])
            return (hm)

        def r21(q, idx=1):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), 0],
                             [0, np.sin(q), np.cos(q), l],
                             [0, 0, 0, 1]])
            return (hm)

        def r32(q, idx=2):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), l],
                             [0, np.sin(q), np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r43(q, idx=3):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, l],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r54(q, idx=4):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), 0, -np.sin(q), 0],
                             [0, 1, 0, l],
                             [np.sin(q), 0, np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r65(q, idx=5):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), 0, -np.sin(q), l],
                             [0, 1, 0, 0],
                             [np.sin(q), 0, np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        self.ht_list = [r10, r21, r32, r43, r54, r65]
        self.joint_abs_locations = self.loc_joints(qs=[0] * len(self.rotation_axises))

    @staticmethod
    def _simple_forward_kinematics(qs, new_x, transform_list):
        """
        routine for calculating the absolute location
        :param new_x:
        :param transform_list:
        :param qs:
        :return:
        """
        for hm, q in zip(transform_list, qs):
            tmp_x = np.dot(hm(q), new_x)
            new_x = tmp_x
        return(new_x)

    def forward_kinematics(self, qs, x, reference_frame=6):
        """
        convert the relative loctions in the joint frame into the world frame
        :param qs: joint angles
        :param x: location in the joint frame
        :param reference_frame: frame in which the x is specified
        :return:
        """
        transform_list = self.ht_list[:reference_frame]
        transform_list = transform_list[::-1]

        qs = qs[:reference_frame]
        qs = qs[::-1]

        new_x = self._simple_forward_kinematics(qs, x, transform_list)
        return(new_x)

    def loc_joints(self, qs=None):
        """
        if qs is not specified, calculate the absolute positions of the joints under the current configurations
        :param qs:
        :return:
        """
        if qs is None:  # if joint angles were not specified, use the current joint angles
            qs = self.joint_angles
        joint_abs_locations = []
        for idx, rel_loc in enumerate(self.joint_relative_locations):
            tmp_loc = self.forward_kinematics(qs, rel_loc, reference_frame=idx)
            joint_abs_locations.append(tmp_loc)
        self.joint_abs_locations = np.asarray(joint_abs_locations)
        return (self.joint_abs_locations)

    def configure_robots(self, qs):
        """
        configure the robot to specified rotation angles
        :param qs: specify the angles of each joints
        :return:
        """
        self.joint_angles = qs
        self.joint_abs_locations = self.loc_joints(self.joint_angles)

    def _update_angular_velocities(self, action):
        """
        update the angular velocities and hold/release state
        :param action:
        :return:
        """
        self.angular_velocities += action[:-1] * self.time_step
        self.release = action[-1]
        if np.any(np.abs(self.angular_velocities) > self.joint_vel_limit):
            self.release = 1

    def _update_joint_angles(self):
        """
        update the joint angles
        :return:
        """
        self.joint_angles += self.angular_velocities * self.time_step
        if np.any(np.abs(self.joint_angles) > self.joint_angle_limit):
            self.release = 1

    def update(self, action):
        """
        update the robot to the next time step
        update the robot's joint angles based on the velocity from previous time step and \
            angular velocity based on current action (specified as acceleration)
        :param action: acceleration at each joint and whether to release the ball
        :return:
        """
        self._update_joint_angles()
        self._update_angular_velocities(action)
        self.time += self.time_step

    def _jacobian_matrix(self, x, qs=None, reference_frame=6, delta=1e-10):
        """
        calculate the value of the Current Jacobian matrix values
        :return:
        """
        if qs is None:
            qs = self.joint_angles
        new_qs = qs[:reference_frame]
        new_qs = new_qs[::-1]

        transform_list = self.ht_list[:reference_frame]
        transform_list = transform_list[::-1]

        new_x = self._simple_forward_kinematics(new_qs, x, transform_list)

        jacobian = np.zeros((4, self.num_joints))
        for i in range(self.num_joints):
            delta_qs = np.zeros(len(transform_list))
            delta_qs[i] = delta
            tmp_qs = new_qs + delta_qs
            tmp_x = self._simple_forward_kinematics(tmp_qs, x, transform_list)
            tmp_jacobian = (tmp_x - new_x) / delta
            jacobian[:, self.num_joints-i-1] = tmp_jacobian

        return(jacobian)

    def cal_ee_speed(self):
        """
        calculate the speed of the end effector in the world frame
        :return:
        """
        # pdb.set_trace()
        x = self.joint_relative_locations[-1]
        jacobian = self._jacobian_matrix(x)
        v_dot = np.dot(jacobian, self.angular_velocities)
        return(v_dot)

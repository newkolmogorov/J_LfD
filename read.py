import os, pickle, numpy as np


class Demonstration(object):

    def __init__(self, demo_dir):
        self.demo_dir = demo_dir

        goal_data = pickle.load(open(os.path.join(demo_dir, 'pcae.pk'), 'rb'))[1:]
        robot_data = pickle.load(open(os.path.join(demo_dir, 'robot_states.pk'), 'rb'))[1:]
        keyframe_times = np.load(os.path.join(demo_dir, 'keyframes.npy'))

        self.action_keyframes = np.array([self.get_closest(kf, robot_data)[4] for kf in keyframe_times])
        self.goal_keyframes = np.array([self.get_closest(kf, goal_data)[1] for kf in keyframe_times])

        self.all_action = np.array([r[4] for r in robot_data])
        self.all_goal = np.array([g[1] for g in goal_data])

    def get_closest(self, time, data):
        time_diffs = list(map(lambda x: np.abs(time-x[0]), data))
        closest_idx = np.argsort(time_diffs)[0]
        return data[closest_idx]


class Skill(object):
    def __init__(self, skill_dir):
        self.skill_dir = skill_dir

        self.demo_dirs = map(
            lambda x: os.path.join(self.skill_dir, x),
            sorted(os.listdir(self.skill_dir), key=lambda x: int(x)))

        self.demos = []

        for demo_dir in self.demo_dirs:
            self.demos.append(Demonstration(demo_dir))
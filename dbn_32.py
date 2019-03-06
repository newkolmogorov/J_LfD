from utils import *


class DBN:

    def __init__(self, nstates_action, nstates_goal, demo_ls, obs_action, obs_goal, keys, num_demo):

        self.n_states_action = nstates_action
        self.n_states_goal = nstates_goal
        self.demo_ls = demo_ls
        self.obs_action = obs_action
        self.obs_goal = obs_goal
        self.keyframe_nums = keys
        self.num_demo = num_demo

        self.startprob_action, self.means_action, self.covars_action = init_w_hmm(self.n_states_action,
                                                                                  self.obs_action,
                                                                                  self.keyframe_nums)

        self.startprob_goal, self.means_goal, self.covars_goal = init_w_hmm(self.n_states_goal,
                                                                            self.obs_goal,
                                                                            self.keyframe_nums)

        self.transprob_action, self.transprob_goal = transition_prob(self.demo_ls,
                                                                     self.means_action, self.covars_action,
                                                                     self.means_goal, self.covars_goal)

    def belief(self):

        belief_a = [np.empty((self.keyframe_nums[r], self.n_states_action)) for r in range(self.num_demo)]
        belief_g = [np.empty((self.keyframe_nums[r], self.n_states_goal)) for r in range(self.num_demo)]

        for r in range(self.num_demo):
            for t in range(self.keyframe_nums[r]):

                if t == 0:
                    belief_a[r][t, :] = self.startprob_action
                    belief_g[r][t, :] = self.startprob_goal

                else:

                    belief_a[r][t, :] = sum(
                        [belief_a[r][t - 1, i] * belief_g[r][t - 1, j] * self.transprob_action[i, j, :]
                         for i in range(self.n_states_action) for j in range(self.n_states_goal)]
                    )

                    belief_g[r][t, :] = sum(
                        [belief_a[r][t - 1, i] * belief_g[r][t - 1, j] * self.transprob_goal[i, j, :]
                         for i in range(self.n_states_action) for j in range(self.n_states_goal)]
                    )

                sum_normalize(belief_a[r][t, :])
                sum_normalize(belief_g[r][t, :])

        return belief_a, belief_g

    def log_alpha(self, belief_a, belief_g):

        log_a = [np.zeros(shape=(self.keyframe_nums[r], self.n_states_action)) for r in range(self.num_demo)]
        log_g = [np.zeros(shape=(self.keyframe_nums[r], self.n_states_goal)) for r in range(self.num_demo)]

        for r in range(self.num_demo):
            for t in range(self.keyframe_nums[r]):

                if t == 0:

                    log_a[r][t, :] = np.log(self.startprob_action + 1e-6) + \
                                     log_pdf(self.demo_ls[r][0], self.means_action, self.covars_action)[t, :]
                    log_g[r][t, :] = np.log(self.startprob_goal + 1e-6) + \
                                     log_pdf(self.demo_ls[r][1], self.means_goal, self.covars_goal)[t, :]

                else:

                    log_a[r][t, :] = log_pdf(self.demo_ls[r][0], self.means_action, self.covars_action)[t, :] + \
                                     np.log(
                                         np.sum(
                                             np.array(
                                                 [np.exp(log_a[r][t - 1, i]) * belief_g[r][t - 1, j] *
                                                  self.transprob_action[i, j, :]
                                                  for i in range(self.n_states_action)
                                                  for j in range(self.n_states_goal)
                                                  ]
                                             ),
                                             axis=0
                                         ) + 1e-6
                                     )

                    log_g[r][t, :] = log_pdf(self.demo_ls[r][1], self.means_goal, self.covars_goal)[t, :] + \
                                     np.log(
                                         np.sum(
                                             np.array(
                                                 [np.exp(log_g[r][t - 1, j])*belief_a[r][t - 1, i] *
                                                  self.transprob_goal[i, j, :]
                                                  for i in range(self.n_states_action)
                                                  for j in range(self.n_states_goal)
                                                  ]
                                             ),
                                             axis=0
                                         ) + 1e-6
                                     )

                idx_a, idx_g = np.where(log_a[r][t, ] < -700)[0], np.where(log_g[r][t, ] < -700)[0]
                log_a[r][t, idx_a] = -700
                log_g[r][t, idx_g] = -700

        return log_a, log_g

    def log_beta(self, bel_a, bel_g,):

        log_a = [np.empty((self.keyframe_nums[r], self.n_states_action)) for r in range(self.num_demo)]
        log_g = [np.empty((self.keyframe_nums[r], self.n_states_goal)) for r in range(self.num_demo)]

        for r in range(self.num_demo):
            for t in range(self.keyframe_nums[r] - 1, -1, -1):

                if t == self.keyframe_nums[r] - 1:

                    log_a[r][t, :] = np.array([0] * self.n_states_action)
                    log_g[r][t, :] = np.array([0] * self.n_states_goal)

                else:

                    for i in range(self.n_states_action):
                        log_a[r][t, i] = np.log(
                            np.sum(
                                np.sum(np.array([self.transprob_action[i, j, :] * bel_g[r][t, j]
                                                 for j in range(self.n_states_goal)]), axis=0) *
                                np.exp(log_pdf(self.demo_ls[r][0], self.means_action, self.covars_action)[t + 1, :]) *
                                np.exp(log_a[r][t+1, :])
                            )+1e-6
                        )

                    for j in range(self.n_states_goal):
                        log_g[r][t, j] = np.log(
                            np.sum(
                                np.sum(np.array([self.transprob_goal[i, j, :] * bel_a[r][t, i]
                                                 for i in range(self.n_states_action)]), axis=0) *
                                np.exp(log_pdf(self.demo_ls[r][1], self.means_goal, self.covars_goal)[t + 1, :]) *
                                np.exp(log_g[r][t + 1, :])
                            )+1e-6
                        )

        return log_a, log_g

    def log_phi(self, log_alpha_a, log_alpha_g, log_beta_a, log_beta_g):

        log_phi_a = [np.empty((self.keyframe_nums[r] - 1, self.n_states_action, self.n_states_goal,
                               self.n_states_action))
                     for r in range(self.num_demo)]
        log_phi_g = [np.empty((self.keyframe_nums[r] - 1, self.n_states_action, self.n_states_goal,
                               self.n_states_goal))
                     for r in range(self.num_demo)]

        for r in range(self.num_demo):
            for t in range(self.keyframe_nums[r]-1):

                den1 = np.sum(np.exp(log_alpha_a[r][t, ]))
                den2 = np.sum(np.exp(log_alpha_g[r][t, ]))
                for i in range(self.n_states_action):
                    for j in range(self.n_states_goal):

                        log_phi_a[r][t, i, j, :] = log_alpha_a[r][t, i] + log_alpha_g[r][t, j] + \
                            np.log(self.transprob_action[i, j, :]+1e-6) + \
                            log_pdf(self.demo_ls[r][0], self.means_action, self.covars_action)[t+1, ] +\
                            log_beta_a[r][t+1, ] + log_beta_g[r][t, j] - np.log(den1+1e-6) - np.log(den2+1e-6)

                        log_phi_g[r][t, i, j, :] = log_alpha_a[r][t, i] + log_alpha_g[r][t, j] + \
                            np.log(self.transprob_goal[i, j, :] + 1e-6) + \
                            log_pdf(self.demo_ls[r][1], self.means_goal, self.covars_goal)[t + 1, ] + \
                            log_beta_a[r][t, i] + log_beta_g[r][t+1, ] - np.log(den1+1e-6) - np.log(den2+1e-6)

        return log_phi_a, log_phi_g

    def transition_update(self, log_phi_a, log_phi_g):

        phi_a = [np.exp(log_phi_a[r]) for r in range(self.num_demo)]
        phi_g = [np.exp(log_phi_g[r]) for r in range(self.num_demo)]

        for i in range(self.n_states_action):
            for j in range(self.n_states_goal):
                for k in range(self.n_states_action):

                    self.transprob_action[i, j, k] = sum(
                        [
                            np.sum(phi_a[r][:, i, j, k])
                            for r in range(self.num_demo)
                        ]
                    )/sum(
                        [
                            np.sum(phi_a[r][:, i, j, :])
                            for r in range(self.num_demo)
                        ]
                    )
                for k in range(self.n_states_goal):

                    self.transprob_goal[i, j, k] = sum(
                        [
                            np.sum(phi_g[r][:, i, j, k])
                            for r in range(self.num_demo)
                        ]
                    ) / sum(
                        [
                            np.sum(phi_g[r][:, i, j, :])
                            for r in range(self.num_demo)
                        ]
                    )

    def c_cal(self, log_alpha, log_beta):

        n_state = log_alpha[0].shape[1]
        c = [np.zeros(shape=(self.keyframe_nums[r], n_state)) for r in range(self.num_demo)]

        for r in range(self.num_demo):
            for t in range(self.keyframe_nums[r]):
                for i in range(n_state):
                    c[r][t, i] = np.exp(
                        log_alpha[r][t, i] + log_beta[r][t, i]
                    ) / sum(
                        [
                            np.exp(
                                log_alpha[r][t, j] + log_beta[r][t, j]
                            )
                            for j in range(n_state)
                        ]
                    )
        return c

    def mu_update(self, c_a, c_g):

        for i in range(self.n_states_action):
            self.means_action[i, :] = sum(
                [
                    c_a[r][t, i] * self.demo_ls[r][0][t, :]
                    for r in range(self.num_demo) for t in range(self.keyframe_nums[r])
                ]
            ) / sum(
                [
                    c_a[r][t, i]
                    for r in range(self.num_demo) for t in range(self.keyframe_nums[r])
                ]
            )

        for i in range(self.n_states_goal):
            self.means_goal[i, :] = sum(
                [
                    c_g[r][t, i] * self.demo_ls[r][1][t, :]
                    for r in range(self.num_demo) for t in range(self.keyframe_nums[r])
                ]
            ) / sum(
                [
                    c_g[r][t, i]
                    for r in range(self.num_demo) for t in range(self.keyframe_nums[r])
                ]
            )

    def cov_update(self, c_a, c_g):

        for i in range(self.n_states_action):
            nominator, denominator = 0, 0
            for r in range(self.num_demo):
                for t in range(self.keyframe_nums[r]):
                    x1 = np.matrix(self.demo_ls[r][0][t, :] - self.means_action[i, :])  # shape = (1, 7)
                    x2 = np.transpose(x1)  # shape = (7, 1)
                    nominator += np.matmul(x2, x1) * c_a[r][t, i]
                    denominator += c_a[r][t, i]
            self.covars_action[i, :] = nominator / denominator

        for i in range(self.n_states_goal):
            nominator, denominator = 0, 0
            for r in range(self.num_demo):
                for t in range(self.keyframe_nums[r]):
                    x1 = np.matrix(self.demo_ls[r][1][t, :] - self.means_goal[i, :])  # shape = (1, 7)
                    x2 = np.transpose(x1)  # shape = (7, 1)
                    nominator += np.matmul(x2, x1) * c_g[r][t, i]
                    denominator += c_g[r][t, i]
            self.covars_goal[i, :] = nominator / denominator

    def prior_update(self, log_alpha_a, log_alpha_g, log_beta_a, log_beta_g):

        for i in range(self.n_states_action):
            self.startprob_action[i] = sum(
                [
                    np.exp(log_alpha_a[r][0, i] + log_beta_a[r][0, i])
                    for r in range(self.num_demo)
                ]
            ) / sum(
                [
                    np.exp(log_alpha_a[r][0, j] + log_beta_a[r][0, j])
                    for r in range(self.num_demo)
                    for j in range(self.n_states_action)
                ]
            )

        for i in range(self.n_states_goal):
            self.startprob_goal[i] = sum(
                [
                    np.exp(log_alpha_g[r][0, i] + log_beta_g[r][0, i])
                    for r in range(self.num_demo)
                ]
            ) / sum(
                [
                    np.exp(log_alpha_g[r][0, j] + log_beta_g[r][0, j])
                    for r in range(self.num_demo)
                    for j in range(self.n_states_goal)
                ]
            )

    def fit(self, max_iter):

        iter_no = 0

        while iter_no < max_iter:

            b1, b2 = self.belief()
            a1, a2 = self.log_alpha(b1, b2)
            c1, c2 = self.log_beta(b1, b2)
            log_phi_1, log_phi_2 = self.log_phi(a1, a2, c1, c2)

            self.prior_update(a1, a2, c1, c2)
            self.transition_update(log_phi_1, log_phi_2)

            c_a, c_g = self.c_cal(a1, c1), self.c_cal(a2, c2)
            self.mu_update(c_a, c_g)
            self.cov_update(c_a, c_g)

            print('iteration ', str(iter_no), ' is completed')
            iter_no += 1

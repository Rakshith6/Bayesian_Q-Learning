# Independent learners

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import copy
import Game

# Bayesian agent
# Agents maintain PDF's for the reward (parameter) for each action.
# Agents sample reward for each action from the PDF's, and selects action that has the highest sampled reward.
# Agents update PDF using (action, reward) tuples.
class agent_BQ:
    def __init__(self, actions):
        self.action_centers = actions
        self.ALFA = 0.1
        self.payoff_history = []
        self.action_history = []
        self.action = 0.0
        self.action_idx = 0.0

        self.actionReward_dist = [joint_dist(np.array([-0.0,0.0,2.0,2.0])) for i in range(len(self.action_centers))]
        self.dist_params = np.zeros((len(self.action_centers), 4))

        self.rewardFn_param = np.arange(-5, 15, 0.2)
        self.true_rewardFn = np.zeros((len(self.action_centers), len(self.rewardFn_param)))

    def sample_action(self,_):
        parameter_draw = np.zeros(len(self.action_centers))
        for i in range(len(parameter_draw)):
            parameter_draw[i] = self.actionReward_dist[i].sample_mu()

        self.action_idx = np.argmax(parameter_draw)
        self.action = self.action_centers[self.action_idx]

        return self.action

    # agent associates the applied action to the nearest reference action i.e. interval centre
    def update_values(self, action, payoff, _):
        reward_bin = np.digitize(payoff, self.rewardFn_param)

        self.true_rewardFn[self.action_idx, reward_bin] += 1

        self.actionReward_dist[self.action_idx].update_params(np.array([payoff]))
        self.dist_params[self.action_idx,:] = self.actionReward_dist[self.action_idx].param

# Joint distribution class
class joint_dist:
    def __init__(self, params):
        self.sig_step = 0.05
        self.sig_start = 0.1
        self.sig_end = 20
        self.sig_linspace = np.arange(self.sig_start, self.sig_end, self.sig_step)
        self.sig_CDF = np.zeros(len(self.sig_linspace))
        self.sig_PDF = np.zeros(len(self.sig_linspace))
        self.mu_step = 0.5
        self.mu_start = -100
        self.mu_end = 100
        self.mu_linspace = np.arange(self.mu_start, self.mu_end, self.mu_step)
        self.mu_CDF = np.zeros(len(self.mu_linspace))
        self.mu_PDF = np.zeros(len(self.mu_linspace))
        self.param = params # parameters of the joint distribution

        self.update_no = 0
        self.update_dist()

    # inverse-gamma distribution for the variance parameter
    def sig_invGammaDist(self, sig2):
        return sig2 ** -(1 + self.param[2] / 2) * np.exp(-self.param[3] / 2 / sig2)

    # normal distribution for the mean parameter
    def mu_normalDist(self, mu, sig_sample):
        return sig_sample ** -0.5 * np.exp(-self.param[1] * (mu - self.param[0]) ** 2/(2*sig_sample))

    def update_dist(self):
        self.update_sigDist()
        self.update_muDist()

    # update the distribution based on current parameter values
    def update_sigDist(self):
        self.sig_PDF= self.sig_invGammaDist(self.sig_linspace)
        self.sig_PDF/= sum(self.sig_PDF)

        self.sig_CDF= np.cumsum(self.sig_PDF)

    def update_muDist(self):
        sig_sample = self.sample_sig() # sample the sigma square and then find the conditional mean distribution

        self.mu_PDF= self.mu_normalDist(self.mu_linspace, sig_sample)
        self.mu_PDF/= sum(self.mu_PDF)

        self.mu_CDF = np.cumsum(self.mu_PDF)

    # sample a sigma square value from the inv-gamma distribution
    def sample_sig(self):
        r = np.random.random()
        return self.sig_linspace[np.where(self.sig_CDF >= r)[0][0]]

    # sample a mean parameter value from the normal distribution
    def sample_mu(self):
        r = np.random.random()
        return self.mu_linspace[np.where(self.mu_CDF >= r)[0][0]]

    def update_params(self, data):
        n = len(data)
        data_mean = np.mean(data)

        hparam_old = copy.copy(self.param)

        self.param[0] = (hparam_old[1]*hparam_old[0] + n*data_mean)/(hparam_old[1] + n)
        self.param[1] += n
        self.param[2] += n
        self.param[3] = hparam_old[3] + np.var(data) + hparam_old[1]*n/(hparam_old[1]+n)*(hparam_old[0] - data_mean)**2

        try:
            self.update_dist()
            self.update_no += 1
        except:
            print('params',self.param, '\nUpdate no',self.update_no)

def run_simulation():
    game_solutions = np.zeros((TOTAL_GAMES, 3, EPISODES))
    game_payoffs = np.zeros((TOTAL_GAMES, 2, EPISODES))
    game_result = np.zeros((TOTAL_GAMES,3))

    for game_no in range(TOTAL_GAMES):
        if game_no%10 == 0:
            print('Running game number {}'.format(game_no))
        game = Game.Game()
        game.determine_payoffPlanes()

        AgentA = agent_BQ(action_vector)
        AgentB = agent_BQ(action_vector)

        agent_scores = np.zeros((2, EPISODES), dtype=int)
        for trial in range(EPISODES):
            actionA = AgentA.sample_action(trial)
            actionB = AgentB.sample_action(trial)

            AgentA.action_history.append(actionA)
            AgentB.action_history.append(actionB)

            payA, payB = game.find_payoff(actionA, actionB)

            AgentA.payoff_history.append(payA)
            AgentB.payoff_history.append(payB)

            AgentA.update_values(actionA, payA, trial)
            AgentB.update_values(actionB, payB, trial)

            game_solutions[game_no,game.determine_solutionType(actionA, actionB), trial] = 1

            agent_scores[:, trial] = np.array([payA, payB])


        game_payoffs[game_no,:,:] = np.array([AgentA.payoff_history, AgentB.payoff_history])
        game_result[game_no,:] = np.sum(game_solutions[game_no, :, EPISODES - int(0.1 * EPISODES):-1], axis = 1) / int(0.1 * EPISODES)

        plot = True if TOTAL_GAMES < 5 else False
        if plot:
            plt.figure(100+2*game_no)

            plt.subplot(211)
            plt.plot(AgentA.action_history, 'tab:blue')
            plt.plot(AgentB.action_history, 'tab:orange')
            plt.xlabel('Episode')
            plt.ylabel('Agent Actions')

            plt.subplot(212)
            plt.plot(AgentA.payoff_history, 'tab:blue', label = 'Agent A')
            plt.plot(AgentB.payoff_history, 'tab:orange', label = 'Agent B')
            plt.xlabel('Episode')
            plt.ylabel('Agent Reward')
            plt.legend()

            plt.show()


    # plot of the mean payoffs vs trials across games
    game_pay_mean = np.mean(game_payoffs, axis = 0)
    game_pay_std = np.std(game_payoffs, axis = 0)
    game_pay_upBand = game_pay_mean + game_pay_std
    game_pay_lowBand = game_pay_mean - game_pay_std
    plt.figure(0)
    plt.xlabel('Episode')
    plt.ylabel('Agent Reward')
    plt.ylim((0, 10))
    plt.plot(game_pay_mean[0,:], 'tab:blue', label = 'Agent A')
    plt.fill_between(np.arange(1, EPISODES + 1), game_pay_upBand[0, :], game_pay_lowBand[0, :], color ='tab:blue', alpha = 0.3, linewidth = 0)
    plt.plot(game_pay_mean[1,:], 'tab:orange', label = 'Agent B')
    plt.fill_between(np.arange(1, EPISODES + 1), game_pay_upBand[1, :], game_pay_lowBand[1, :], color ='tab:orange', alpha = 0.3, linewidth = 0)
    plt.legend()

    # plotting the mean strategy proportion vs episodes across games
    game_sol_mean = np.mean(game_solutions, axis = 0)
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Strategy Proportion')
    plt.ylim((-0.2, 1.2))
    plt.plot(game_sol_mean[1,:], 'tab:blue', label = 'Cooperation')
    plt.plot(game_sol_mean[0,:], 'tab:orange', label = 'Defection')
    plt.legend()

TOTAL_GAMES = 20 # Number of dyads simulated
EPISODES = 500 # Total learning episodes in each game
action_vector = np.arange(0, 1.1, 0.1) # Dsicretized action space

plt.rcParams.update({'font.size': 14})
run_simulation()







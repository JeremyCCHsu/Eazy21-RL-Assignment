# codings = utf8

'''
Specifically, write a function, named `step`, 
    which takes as input 
    a state s (dealer's first card 1-10 and the player's sum 1-21), and 
    an action a (hit or stick), and 
    returns 
        a sample of the next state s0 
        (which may be terminal if the game is finished)
        and reward r. 

We will be using this environment for model-free reinforcement learning,
and you should not explicitly represent the transition matrix for the MDP.

*There is no discounting (gamma = 1).
*You should treat the dealer's moves as part of the environment, 
i.e. calling step with a stick action will play out the
dealer's cards and return the final reward and terminal state.

*eps: prob of taking the random action in an eps-greedy policy
'''

# import tensorflow as tf
import random
import sys
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from util import plot
from game import Easy21


def epsilonGreedy(eps, actions):
    ''' return action index '''
    roll = np.random.uniform()
    if roll > eps:  # act greedily
        a = np.argmax(actions)
        # print('Greedy act: (p={:d}, d={:d}) => a={:d}; Exp = {}; [{}, {}]'.format(
        #   p + 1, d + 1, a, self.Nsa[p, d, a], self.Qsa[p, d, 0], self.Qsa[p, d, 1]))
        # self.nsa[p, d, a] += 1
        # return self.action[a]
    else:           # act randomly; the first move is always random
        a = random.choice([0, 1])
        # print('Random act: (p={:d}, d={:d}) => a={:d}; Exp = {}'.format(
        #   p + 1, d + 1, a, self.Nsa[p, d, a]))
    return a


class MonteCarloPlayer(object):
    def __init__(self, N0=1000, lam=1.0, gamma=1.0):
        self.Qsa = np.zeros([21, 10, 2])
        self.Nsa = np.zeros([21, 10, 2])  # visit count throughout the whole history
        self.nsa = np.zeros([21, 10, 2])  # visit count in this episode
        self.action = ['hit', 'stick']
        self.N0 = N0        # discounting factor for eligibility
        self.lam = lam      # useless in MC;
        self.gamma = gamma  # useless in MC; discounting factor for reward

    def act(self, state, reward=None):
        '''
        Remeber to convert state into zero-indexing format
        *reward is useless in Monte Carlo
        '''
        p, d = state  # player_sum, dealer_show
        a = self.epsilonGreedy(state=(p - 1, d - 1))
        _ = self._update_a_step(state_action=(p - 1, d - 1, a))
        return self.action[a]

    def epsilonGreedy(self, state):
        ''' return action index '''
        Nst = np.sum(self.Nsa[state])
        eps = self.N0 / (self.N0 + Nst)
        actions = self.Qsa[state]
        return epsilonGreedy(eps, actions)

    def _reset_count(self):
        self.nsa *= 0 # np.zeros([21, 10, 2])
        return None

    def _update_a_step(self, state_action):
        ''' In contrast to Episode-wise contrast '''
        self.nsa[state_action] += 1
        return None

    def update(self, reward):
        ''' Episode-wise MC can only update off-line '''
        index = np.nonzero(self.nsa)
        self.Nsa += self.nsa
        self.Qsa[index] += \
            1. / self.Nsa[index] * (reward - self.Qsa[index])
        self._reset_count()


class SarsaLambdaPlayer(MonteCarloPlayer):
    '''
    Notes:
        Backward view
        Q += a[R + rQ' - Q]
    '''
    def __init__(self, N0=100, lam=1.0, gamma=1.0):
        '''
        # use 'self.nsa' as Eligibility trace
        '''
        super(SarsaLambdaPlayer, self).__init__(N0, lam, gamma)
        self.prev_state_action = 0, 0, 0

    def act_initially(self, state):
        p, d = state  # player_sum, dealer_show
        a = self.epsilonGreedy(state=(p - 1, d - 1))

        s_t = p -1, d - 1, a

        self.prev_state_action = s_t
        # self.prev_2nd_state_action = s_t
        return self.action[a]

    def act(self, state, reward):
        p, d = state  # player_sum, dealer_show
        a = self.epsilonGreedy(state=(p - 1, d - 1))
        _ = self._update_a_step(
            state_action=(p - 1, d - 1, a),
            reward=reward)
        return self.action[a]

    def update(self, reward):
        s_t = self.prev_state_action

        self.Nsa[s_t] += 1
        self.nsa = self.nsa * self.gamma * self.lam
        self.nsa[s_t] += 1

        # [TODO] I use a different update for the last step. (not sure if this is correct)
        alpha = 1. / self.Nsa[s_t]
        delta = reward - self.Qsa[s_t]
        self.Qsa = self.Qsa + alpha * delta * self.nsa  # [TODO] Why propagate for all (s, a)?

        self._reset_count()


    def _update_a_step(self, state_action, reward):
        s_t = state_action
        s_t_1 = self.prev_state_action

        self.Nsa[s_t_1] += 1
        self.nsa = self.nsa * self.gamma * self.lam
        self.nsa[s_t_1] += 1

        alpha = 1. / self.Nsa[s_t_1]  # [TODO] should I use [p, d, a] or [i, j, k] ?
        delta = reward + self.gamma * self.Qsa[s_t] - self.Qsa[s_t_1]        
        self.Qsa = self.Qsa + alpha * delta * self.nsa  # [TODO] Why propagate for all (s, a)?

        self.prev_state_action = s_t
        return None

    def _reset_count(self):
        self.nsa = 0. * self.nsa


def test_mc(N_ITER):
    try:
        mcplayer = MonteCarloPlayer()
        for it in range(N_ITER):
            print('Episode {:8d}'.format(it))
            game = Easy21()
            reward = None
            while True:
                if game.isEnded():
                    break
                state = game.get_state()
                action = mcplayer.act(state)
                reward, state = game.step(action)

            mcplayer.update(reward)
        plot(mcplayer, 'MC-as-standard')
    except KeyboardInterrupt:
        print('Done')
    finally:
        plot(mcplayer)


def test_sarsa_lambda(N_ITER, N_ITER_SARSA):
    try:
        Qsa = list()

        mcplayer = MonteCarloPlayer()
        for it in range(N_ITER):
            print('Episode {:8d}'.format(it))
            game = Easy21()
            while True:
                if game.isEnded():
                    break
                else:
                    state = game.get_state()
                    action = mcplayer.act(state)
                    reward, state = game.step(action)

            mcplayer.update(reward)

        plot(mcplayer, 'MC-as-standard')
        Qsa.append(np.expand_dims(mcplayer.Qsa, 0))

        n_lambda = 11
        mse = np.zeros((n_lambda,))
        for i in range(0, n_lambda):
            player = SarsaLambdaPlayer(lam=i*0.1)
            for it in range(N_ITER_SARSA):
                print('Episode {:8d}'.format(it))
                game = Easy21()
                action = player.act_initially(game.get_state())
                reward, state = game.step(action)
                while True:
                    if game.isEnded():
                        break
                    else:
                        state = game.get_state()
                        action = player.act(state, reward)
                        reward, state = game.step(action)

                player.update(reward)

            plot(player, 'sarsa-lam-{}'.format(i))
            mse[i] = np.mean(np.square(mcplayer.Qsa - player.Qsa))
            Qsa.append(np.expand_dims(player.Qsa, 0))

        with open('Qsa.npf', 'w') as f:
            Qsa = np.concatenate(Qsa, 0)
            Qsa.tofile(f)
    except KeyboardInterrupt:
        print('Done')

    finally:
        plot(player)
        plt.figure()
        plt.plot(np.arange(11) / 10., mse, 'o-')
        plt.xlabel('lambda')
        plt.ylabel('Mean Squared Error')
        plt.savefig('MSE.png')
        plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        N_ITER = int(sys.argv[1])
    else:
        N_ITER = 1000

    if len(sys.argv) > 2:
        N_ITER_SARSA = int(sys.argv[2])
    else:
        N_ITER_SARSA = 1000
    
    test_sarsa_lambda(N_ITER, N_ITER_SARSA)
    # test_mc(N_ITER)

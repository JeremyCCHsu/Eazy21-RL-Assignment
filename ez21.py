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
# import matplotlib as mpl
# mpl.use('Agg')

# import matplotlib.pyplot as plt
import numpy as np
from util import plot
from game import Easy21

N_ITER = 1  # num of episodes



def epsilonGreedy(eps, actions):
    ''' return action index '''
    roll = np.random.uniform(1)
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
    def __init__(self, N0=100, lam=1.0, gamma=1.0):
        self.Qsa = np.zeros([21, 10, 2])
        self.Nsa = np.zeros([21, 10, 2])  # the whole history count
        self.nsa = np.zeros([21, 10, 2])  # this episode
        self.action = ['hit', 'stick']
        self.N0 = N0        # discounting factor for eligibility
        self.lam = lam      # useless in MC; discounting factor for reward
        self.gamma = gamma  # useless in MC; stickness to random action for epsilon

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
        Nst = np.sum(self.nsa[state])
        eps = self.N0 / (self.N0 + Nst)
        actions = self.Qsa[state]
        return epsilonGreedy(eps, actions)

    def _reset_count(self):
        self.nsa = np.zeros([21, 10, 2])
        return None

    def _update_a_step(self, state_action):
        ''' In contrast to Episode-wise contrast '''
        self.nsa[state_action] += 1
        return None

    def update(self, reward):
        ''' Episode-wise MC can only update off-line '''
        i, j, k = np.nonzero(self.nsa)
        self.Nsa += self.nsa
        self.Qsa[i, j, k] += \
            1. / self.Nsa[i, j, k] * (reward - self.Qsa[i, j, k])
        self._reset_count()


class SarsaLambdaPlayer(MonteCarloPlayer):
    '''
    Backward view
        Q += a[R + rQ' - Q]
    '''
    def __init__(self, N0=100, lam=1.0, gamma=1.0):
        super(SarsaLambdaPlayer, self).__init__(N0, lam, gamma)
        # self.nsa = np.zeros([21, 10, 2])
        # use 'self.nsa' as Eligibility trace
        self.prev_state_action = 0, 0, 0
        self.prev_2nd_state_action = 0, 0, 0
        # i, j, k = self.prev_state_action
        # self.Nsa[i, j, k] = 1

    def act_initially(self, state):
        p, d = state  # player_sum, dealer_show
        a = self.epsilonGreedy(state=(p - 1, d - 1))

        s_t = p -1, d - 1, a
        # self.Nsa[s_t] += 1
        # self.nsa[s_t] += 1

        self.prev_state_action = s_t
        self.prev_2nd_state_action = s_t
        return self.action[a]

    def act(self, state, reward):
        p, d = state  # player_sum, dealer_show
        a = self.epsilonGreedy(state=(p - 1, d - 1))
        _ = self._update_a_step(
            state_action=(p - 1, d - 1, a),
            reward=reward)
        return self.action[a]

    def _update_a_step(self, state_action, reward):
        s_t = state_action
        s_t_1 = self.prev_state_action

        self.Nsa[s_t_1] += 1
        self.nsa = self.nsa * self.gamma * self.lam
        self.nsa[s_t_1] += 1

        alpha = 1. / self.Nsa[s_t_1]  # [TODO] should I use [p, d, a] or [i, j, k] ?
        delta = reward + self.gamma * self.Qsa[s_t] - self.Qsa[s_t_1]
        
        self.Qsa = self.Qsa + alpha * delta * self.nsa  # [TODO] Why propagate for all (s, a)?
        
        # I have to update Nsa here; otherwise the next step will explode when computing alpha.
        # self.Nsa[s_t] += 1
        self.prev_state_action = s_t
        self.prev_2nd_state_action = s_t_1
        # self.Nsa[s_t_1] += 1
        return None

    def _reset_count(self):
        self.nsa = 0. * self.nsa

    def update(self, reward):
        s_t = self.prev_state_action
        # self.Nsa[s_t] += 1

        self.Nsa[s_t] += 1
        self.nsa = self.nsa * self.gamma * self.lam
        self.nsa[s_t] += 1

        alpha = 1. / self.Nsa[s_t]
        delta = reward - self.Qsa[s_t]
        self.Qsa = self.Qsa + alpha * delta * self.nsa  # [TODO] Why propagate for all (s, a)?
        # self.nsa *= self.gamma * self.lam

        self._reset_count()




def test_mc(N_ITER):
    try:
        player = MonteCarloPlayer()
        for it in range(N_ITER):
            print('Episode {:8d}'.format(it))
            game = Easy21()
            reward = None
            # while reward is None:
            while True:
                if game.isEndGame():
                    break
                state = game.get_state()
                action = player.act(state)
                reward, state = game.step(action)

            player.update(reward)
    except KeyboardInterrupt:
        print('Done')
    finally:
        plot(player)


def test_sarsa_lambda(N_ITER):
    try:
        # mcPlayer = MonteCarloPlayer()
        # game = Easy21()
        # for it in range(N_ITER):
        #     print('Episode {:8d}'.format(it))
        #     game = Easy21()
        #     reward = None
        #     while reward is None:
        #         state = game.get_state()
        #         action = mcPlayer.act(state)
        #         reward, state = game.step(action)

        #     mcPlayer.update(reward)
        
        for i in range(5, 6):
            player = SarsaLambdaPlayer(lam=i*0.1)
            for it in range(N_ITER):
                print('Episode {:8d}'.format(it))
                game = Easy21()
                # pdb.set_trace()
                action = player.act_initially(game.get_state())
                reward, state = game.step(action)
                # reward = None
                # while reward is None:
                #     if reward is None:
                #         reward = 0
                while True:
                    if game.isEndGame():
                        break
                    else:
                        state = game.get_state()
                        action = player.act(state, reward)
                        reward, state = game.step(action)
                # pdb.set_trace()

                # state_action = state + (player.action.index(action),)
                # player.update(state_action, reward)
                player.update(reward)
            # pdb.set_trace()
            plot(player, 'sarsa-lam-{}'.format(i))
        
    except KeyboardInterrupt:
        print('Done')
    finally:
        plot(player)
        # print(player.Qsa)


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        N_ITER = int(sys.argv[1])
    # game = Easy21()
    test_sarsa_lambda(N_ITER)
    # test_mc(N_ITER)

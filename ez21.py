# codings = utf8

# import tensorflow as tf
import numpy as np
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import sys

N_ITER = 1  # num of episodes

class Easy21(object):
    ''' 
    point: between [1, 10]
    color: determined during card drawing procedures
    '''
    def __init__(self, p=1./3):
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
        self.p = p  # prob of red (minus) card

        # player sum v. dealer showing
        self.dealer_sum = self.dealer_show = self.draw_1st_card()
        self.player_sum = self.draw_1st_card()

        # print('Initially:\n' +
        #   '  Player: {:d}\n'.format(self.player_sum) + 
        #   '  Dealer: {:d}\n'.format(self.dealer_show))


    def get_state(self):
        return [self.player_sum, self.dealer_show]


    def draw(self, person):
        sign = np.random.binomial(1, self.p)
        card = np.random.randint(low=1, high=11)
        if sign > 0.5:
            card = - card

        # msg = '  Player: {:>2d}'.format(self.player_sum)
        # if person == 'player':    
        #   msg += ', {:>2d}'.format(card)
        # msg += '\n  Dealer: {:>2d}'.format(self.dealer_sum)
        # if person == 'dealer':
        #   msg += ', {:>2d}'.format(card)
        # print(msg)

        return card
            

    def draw_1st_card(self):
        point = np.random.randint(low=1, high=11)
        return point


    def isEndGame(self):
        if self.player_sum < 1 or self.player_sum > 21:
            return True
        if self.dealer_sum < 1 or self.dealer_sum > 21:
            return True
        else:
            return False

    def reward(self):
        if self.player_sum < 1 or self.player_sum > 21:
            # print('Player busted')
            return -1
        if self.dealer_sum < 1 or self.dealer_sum > 21:
            # print('Dealer busted')
            return 1
        if self.player_sum < self.dealer_sum:
            # print('Player lost')
            return -1
        if self.player_sum > self.dealer_sum:
            # print('Player won')
            return 1
        if self.player_sum == self.dealer_sum:
            # print('A draw')
            return 0
        else:
            raise ValueError(
                'Impossible: {:d} {:d}'.format(
                    self.player_sum,
                    self.dealer_sum))

    def step(self, action):
        # player's move: stick or draw
        # detection of end-game
        # dealer's move: (criterion)
        # detection of end-game
        if action == 'stick':
            while self.dealer_sum < 17:
                self.dealer_sum += self.draw('dealer')
                if self.isEndGame():
                    break
            return self.reward(), (None, None)

        else:
            self.player_sum += self.draw('player')
            if self.isEndGame():
                return self.reward(), (None, None)
            else:
                self.dealer_sum += self.draw('dealer')
                if self.isEndGame():
                    return self.reward(), (None, None)

        return None, (self.player_sum, self.dealer_show)


class Player(object):
    def __init__(self):
        self.Qsa = np.zeros([21, 10, 2])
        self.Nsa = np.zeros([21, 10, 2])  # the whole history count
        self.nsa = np.zeros([21, 10, 2])  # this episode
        self.action = ['hit', 'stick']

    def act(self, state, N0=100):
        p, d = state  # player_sum, dealer_show
        p, d = p - 1, d - 1  # convert into zero-started index for numpy

        Nst = np.sum(self.nsa[p, d])
        eps = N0 / (N0 + Nst)  # eps: random, 1 - eps: best

        roll = np.random.uniform(1)
        if roll > eps:  # act greedily
            a = np.argmax(self.Qsa[p, d])
            # print('Greedy act: (p={:d}, d={:d}) => a={:d}; Exp = {}; [{}, {}]'.format(
            #   p + 1, d + 1, a, self.Nsa[p, d, a], self.Qsa[p, d, 0], self.Qsa[p, d, 1]))
            # self.nsa[p, d, a] += 1
            # return self.action[a]
        else:           # act randomly; the first move is always random
            a = random.choice([0, 1])
            # print('Random act: (p={:d}, d={:d}) => a={:d}; Exp = {}'.format(
            #   p + 1, d + 1, a, self.Nsa[p, d, a]))
        self.nsa[p, d, a] += 1
        return self.action[a]

    def _reset_count(self):
        self.nsa = np.zeros([21, 10, 2])

    def update(self, reward):
        ''' MC can only update off-line '''
        i, j, k = np.nonzero(self.nsa)
        self.Nsa += self.nsa
        self.Qsa[i, j, k] += \
            1. / self.Nsa[i, j, k] * (reward - self.Qsa[i, j, k])
        self._reset_count()


class SarsaPlayer(Player):
    '''
    Implement Sarsa(λ) in 21s.
    Initialise the value function to zero.
    Use the same step-size and exploration schedules as in the previous section.

    Run the algorithm with parameter values λ ∈ {0, 0.1, 0.2, ..., 1}.
    Stop each run after 1000 episodes and report the mean-squared error 

        sum s,a(Q(s, a) − Q∗(s, a))^2 over all states s and actions a,

    comparing the true values Q∗(s, a) computed in the previous section with the estimated values 
    Q(s, a) computed by Sarsa. Plot the meansquared error against λ.
    For λ = 0 and λ = 1 only, plot the learning curve of mean-squared error against episode number.

    1. Q += a[R + rQ' - Q]

    First I should impl SARSA-TD1
    '''
    def __init__(self):
        super(SarsaPlayer, self).__init__()
        self.prev_state_action = [0, 0, 0]  # p, d; zero indexed
        # self.Nsa[0, 0, 0] = 1

    def update(self, reward):
        ''' I should update online '''
        pass

    #   i, j, k = np.nonzero(self.nsa)
    #   self.Nsa += self.nsa
    #   self.Qsa

    def act(self, state, reward, N0=100, gamma=1):
        ''' I should update during action? 
        gamma: discount
        '''

        # update
        p, d = state  # player_sum, dealer_show
        p, d = p - 1, d - 1  # convert i

        # Q += a[R + rQ' - Q]nto zero-started index for numpy
        # self.Qsa += 

        # act
        Nst = np.sum(self.nsa[p, d])
        eps = N0 / (N0 + Nst)  # eps: random, 1 - eps: best

        roll = np.random.uniform(1)
        if roll > eps:  # act greedily
            a = np.argmax(self.Qsa[p, d])
            # print('Greedy act: (p={:d}, d={:d}) => a={:d}; Exp = {}; [{}, {}]'.format(
            #   p + 1, d + 1, a, self.Nsa[p, d, a], self.Qsa[p, d, 0], self.Qsa[p, d, 1]))
            # self.nsa[p, d, a] += 1
            # return self.action[a]
        else:           # act randomly; the first move is always random
            a = random.choice([0, 1])
            # print('Random act: (p={:d}, d={:d}) => a={:d}; Exp = {}'.format(
            #   p + 1, d + 1, a, self.Nsa[p, d, a]))
        # self.nsa[p, d, a] += 1

        i, j, k = self.prev_state_action
        # New: [p, d, a],  Old: [i, j, k]
        self.Nsa[p, d, a] += 1
        self.Qsa[i, j, k] += \
            1. / self.Nsa[p, d, a] * \
            (reward + gamma * self.Qsa[p, d, a] - self.Qsa[i, j, k])
        # self._reset_count()
        self.prev_state_action = [p, d, a]

        return self.action[a]

    def update(self, reward):
        ''' The last step of SARSA is to update off-line '''
        # i, j, k = np.nonzero(self.nsa)
        # self.Nsa += self.nsa
        i, j, k = self.prev_state_action
        self.Qsa[i, j, k] += \
            1. / self.Nsa[i, j, k] * (reward - self.Qsa[i, j, k])
        # self._reset_count()


class SarsaLambdaPlayer(SarsaPlayer):
    ''' Forward view implementation '''
    def __init__(self, l):
        super(SarsaLambdaPlayer, self).__init__()
        self.l = l


def plot(player):
    plt.figure(figsize=[12, 18])
    plt.subplot(221)
    plt.imshow(np.max(player.Qsa, 2), vmin=-1, vmax=1, interpolation='none')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(np.log10(np.sum(player.Nsa, 2) + 1), interpolation='none')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(player.Qsa[:,:,0], vmin=-1, vmax=1, interpolation='none')
    plt.colorbar()
    plt.title('hit')
    plt.subplot(224)
    plt.imshow(player.Qsa[:,:,1], vmin=-1, vmax=1, interpolation='none')
    plt.title('stick')
    plt.colorbar()
    plt.savefig('test.png')


def test_mc():
    try:
        player = Player()
        for it in range(N_ITER):
            print('Episode {:8d}'.format(it))
            game = Easy21()
            reward = None
            while reward is None:
                state = game.get_state()
                action = player.act(state)
                reward, state = game.step(action)

            player.update(reward)
    except KeyboardInterrupt:
        print('Done')
    finally:
        plot(player)


def test_sarsa(N_ITER):
    try:
        player = SarsaPlayer()
        for it in range(N_ITER):
            print('Episode {:8d}'.format(it))
            game = Easy21()
            reward = None
            while reward is None:
                if reward is None:
                    reward = 0
                state = game.get_state()
                action = player.act(state, reward)
                reward, state = game.step(action)

            player.update(reward)
    except KeyboardInterrupt:
        print('Done')
    finally:
        plot(player)

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        N_ITER = int(sys.argv[1])
    # game = Easy21()
    test_sarsa(N_ITER)

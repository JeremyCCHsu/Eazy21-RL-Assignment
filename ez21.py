# import tensorflow as tf
import numpy as np
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random


class Easy21(object):
	''' 
	As 2-dim data:
	point: [1, 10]
	color: [0, 1], 0 as black


	def step(state, action):
		
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
		self.p = p

		# player sum v. dealer showing
		self.dealer_sum = self.dealer_show = self.draw_1st_card()
		self.player_sum = self.draw_1st_card()

		print('Initially:\n' +
			'  Player: {:d}\n'.format(self.player_sum) + 
			'  Dealer: {:d}\n'.format(self.dealer_show))
		# state = [self.player_sum, self.dealer_show]


	def get_state(self):
		return [self.player_sum, self.dealer_show]


	def draw(self, person):
		sign = np.random.binomial(1, self.p)
		card = np.random.randint(low=1, high=11)
		if sign > 0.5:
			card = - card

		msg = '  Player: {:>2d}'.format(self.player_sum)
		if person == 'player':	
			msg += ', {:>2d}'.format(card)
		msg += '\n  Dealer: {:>2d}'.format(self.dealer_sum)
		if person == 'dealer':
			msg += ', {:>2d}'.format(card)
		print(msg)

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
			print('Player busted')
			return -1
		if self.dealer_sum < 1 or self.dealer_sum > 21:
			print('Dealer busted')
			return 1
		if self.player_sum < self.dealer_sum:
			print('Player lost')
			return -1
		if self.player_sum > self.dealer_sum:
			print('Player won')
			return 1
		if self.player_sum == self.dealer_sum:
			print('A draw')
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
		self.Nsa = np.zeros([21, 10, 2])
		self.Qsa = np.zeros([21, 10, 2])
		self.Nsa_ = np.zeros([21, 10, 2])
		self.action = ['hit', 'stick']
		# self.action = dict(hit=0, stick=1)

	def act(self, state):
		N0 = 100.0

		p, d = state  # player_sum, dealer_show
		p -= 1
		d -= 1

		Nst = np.sum(self.Nsa_[p, d])
		eps = N0 / (N0 + Nst)  # eps: random, 1 - eps: best

		roll = np.random.uniform(1)
		if roll > eps:	# act greedily
			a = np.argmax(self.Qsa[p, d])
			print('Greedy act: (p={:d}, d={:d}) => a={:d}; Exp = {}; [{}, {}]'.format(
				p + 1, d + 1, a, self.Nsa[p, d, a], self.Qsa[p, d, 0], self.Qsa[p, d, 1]))
			self.Nsa_[p, d, a] += 1
			return self.action[a]
		else:			# act randomly; the first move is always random
			a = random.choice([0, 1])
			print('Random act: (p={:d}, d={:d}) => a={:d}; Exp = {}'.format(
				p + 1, d + 1, a, self.Nsa[p, d, a]))
			self.Nsa_[p, d, a] += 1
			return self.action[a]

	def reset_count(self):
		self.Nsa_ = np.zeros([21, 10, 2])


	def update(self, reward):
		index = np.nonzero(self.Nsa_)
		self.Nsa += self.Nsa_
		self.Qsa[index[0], index[1], index[2]] += \
			1. / self.Nsa[index[0], index[1], index[2]] * \
			(reward - self.Qsa[index[0], index[1], index[2]])
		self.reset_count()

N_ITER = 1000

def main():
	player = Player()

	for it in range(N_ITER):
		game = Easy21()
		# player_tmp = Player()
		# player_tmp.Qsa = player.Qsa
		reward = None
		while reward is None:
			state = game.get_state()
			# action = player_tmp.act(state)
			action = player.act(state)
			reward, state = game.step(action)
		# pdb.set_trace()

		player.update(reward)
		# index = np.nonzero(player_tmp.Nsa)
		# player.Qsa[index[0], index[1], index[2]] += \
		# 	1. / player_tmp.Nsa[index[0], index[1], index[2]] * \
		# 	(reward - player_tmp.Qsa[index[0], index[1], index[2]])

		# player.Nsa += player_tmp.Nsa


		# player.reset_count()


	# pdb.set_trace()
	
	plt.figure(figsize=[12, 18])
	plt.subplot(221)
	plt.imshow(np.max(player.Qsa, 2), vmin=-1, vmax=1, interpolation='none')
	plt.colorbar()
	plt.subplot(222)
	plt.imshow(np.log10(np.sum(player.Nsa, 2)), interpolation='none')
	plt.colorbar()
	plt.subplot(223)
	plt.imshow(player.Qsa[:,:,0], interpolation='none')
	plt.colorbar()
	plt.title('hit')
	plt.subplot(224)
	plt.imshow(player.Qsa[:,:,1], interpolation='none')
	plt.title('stick')
	plt.colorbar()
	plt.savefig('test.png')






if __name__ == '__main__':
	# game = Easy21()
	main()

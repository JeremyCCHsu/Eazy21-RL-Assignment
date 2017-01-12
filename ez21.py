# import tensorflow as tf
import numpy as np
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



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
		self._p_red_ = p

		# player sum v. dealer showing
		self.dealer_show = self.draw_1st_card()
		self.dealer_sum = self.dealer_show
		self.player_sum = self.draw_1st_card()

		print('Initially:\n' +
			'  Player: {:d}\n'.format(self.player_sum) + 
			'  Dealer: {:d}\n'.format(self.dealer_show))
		state = [self.dealer_show, self.player_sum]


	def get_state(self):
		return [self.player_sum, self.dealer_show]


	def draw(self, person):
		sign = np.random.binomial(1, self._p_red_)
		card = np.random.randint(low=1, high=11)
		if sign > 0.5:
			card = - card

		msg = '  Player: {:d}'.format(self.player_sum)
		if person == 'player':	
			msg += ', {:d}'.format(card)
		msg += '\n  Dealer: {:d}'.format(self.dealer_sum)
		if person == 'dealer':
			msg += ', {:d}'.format(card)
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
			return -1
		if self.dealer_sum < 1 or self.dealer_sum > 21:
			return 1
		if self.player_sum < self.dealer_sum:
			return -1
		if self.player_sum > self.dealer_sum:
			return 1
		else:
			return 0

	def _dealer_move(self):
		pass

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
				# self._dealer_move()
				self.dealer_sum += self.draw('dealer')
				if self.isEndGame():
					return self.reward(), (None, None)

		return None, (self.player_sum, self.dealer_show)


class Player(object):
	def __init__(self):
		self.sum = 0
		# self.eps = 1.0
		# self.at = 1.0
		# self.N0 = 100
		self.Nsa = np.ones([21, 10, 2])
		self.Qsa = np.zeros([21, 10, 2])
		self.action = ['hit', 'stick']

	def act(self, state):
		N0 = 100.0

		p, d = state  # player_sum, dealer_show
		p -= 1
		d -= 1

		Nst = np.sum(self.Nsa[p, d])
		eps = N0 / (N0 + Nst)  # eps: random, 1 - eps: best

		roll = np.random.uniform(1)
		if roll > eps:	# act greedily
			a = np.argmax(self.Qsa[p, d])
			self.Nsa[p, d, a] += 1
			return self.action[a]
		else:			# act randomly
			return random.choice(self.action)


	def update(self, reward):
		# if reward is None:
		# 	pass
		# else:
		self.Qsa += 1. / self.Nsa * (reward - self.Qsa)


N_ITER = 10000

def main():
	player = Player()

	for it in range(N_ITER):
		game = Easy21()
		reward = None
		while reward is None:
			state = game.get_state()
			action = player.act(state)
			reward, state = game.step(action)
		player.update(reward)


	# pdb.set_trace()
	
	plt.figure()
	plt.imshow(np.max(player.Qsa, 2))
	plt.colorbar()
	plt.savefig('test.png')






if __name__ == '__main__':
	# game = Easy21()
	main()

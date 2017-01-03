# import tensorflow as tf
import numpy as np


class Easy21(object):
	''' 
	As 2-dim data:
	point: [1, 10]
	color: [0, 1], 0 as black


	def step(state, action):
		
	'''
	def __init__(self, p=1./3):
		'''
		# Player
		# dealer

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
		'''
		self._p_red_ = p

		# player sum v. dealer showing
		self.state = np.zeros([10, 21])

		self.dealer_show = self.draw_1st_card()
		self.dealer_sum = self.dealer_show
		self.player_sum = self.draw_1st_card()
		print('Player: {:d}\nDealer: {:d}'.format(
			self.player_sum,
			self.dealer_show))
		state = [self.dealer_show, self.player_sum]
		state, reward = self.step(state)
		print('End game stats:\n' + 
			'  Player: {:d}\n'.format(state[0]) +
			'  Dealer: {:d}\n'.format(state[1]))


		# self.player.cards = list()
		# self.player.cards.append(self.draw_1st_card())
		# self.dealer.visible = self.draw_1st_card()

		

	def draw(self):
		sign = np.random.binomial(1, self._p_red_)
		point = np.random.randint(low=1, high=11)
		if sign < 0.5:
			return point
		else:
			return - point
			

	def draw_1st_card(self):
		point = np.random.randint(low=1, high=11)
		return point

	def step(self, state, action=None):
		# player's move: stick or draw
		# detection of end-game
		# dealer's move: (criterion)
		# detection of end-game
		self.player_sum += self.draw()
		if self.player_sum < 1 or self.player_sum > 21:
			print('Player busts')
			reward = -1

		elif self.player_sum == 21:
			print('Player wins')
			reward = 1

		if self.dealer_sum < 17:
			self.dealer_sum += self.draw()
			print('Dealer draws a card')

			if self.dealer_sum < 1 or self.dealer_sum > 21:
				print('Dealer busts')
				reward = 1

			elif self.dealer_sum == 21:
				print('Dealer wins')
				reward = -1

		else:
			print('Dealer sticks')


		if self.player_sum < self.dealer_sum:
			print('Dealer wins')
			reward = 1
		elif self.player_sum > self.dealer_sum:
			print('Player wins')
			reward = -1
		else:
			print('Game sets at a draw')
			reward = 0

			# return state, reward
		state = [self.dealer_show, self.player_sum]
		return state, reward


# class Player(object):
# 	def __init__(self):

# 		cards


# 		self.sum()
# 		pass


# class Dealer(object):
# 	def __init__(self):
# 		pass

# 	def act(self):
# 		''' always stick on >= 17 '''
# 		pass


if __name__ == '__main__':
	game = Easy21()

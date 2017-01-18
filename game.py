import numpy as np

class Easy21(object):
    ''' 
    point: between [1, 10]
    color: determined during card drawing procedures
    '''
    def __init__(self, p=1./3):
        self.p = p  # prob of red (minus) card

        # player sum v. dealer showing
        self.dealer_sum = self.dealer_show = self.draw(color='black')
        self.player_sum = self.draw(color='black')
        self._isEnded = False

        # print('Initially:\n' +
        #   '  Player: {:d}\n'.format(self.player_sum) + 
        #   '  Dealer: {:d}\n'.format(self.dealer_show))


    def get_state(self):
        return [self.player_sum, self.dealer_show]

    # def clairvoyance(self):
    #     return [self.player_sum, self.dealer_sum]


    def draw(self, person=None, color=None):
        if color == 'red':
            sign = 1.0
        elif color == 'black':
            sign = 0.0
        else:
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

    def isEnded(self):
        if self._isEnded:
            return True
        if self.player_sum < 1 or self.player_sum > 21:
            return True
        if self.dealer_sum < 1 or self.dealer_sum > 21:
            return True
        else:
            return False

    def _reward(self):
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
        if action == 'stick':
            while self.dealer_sum < 17:
                self.dealer_sum += self.draw('dealer')
                if self.isEnded():
                    break
            self._isEnded = True
            return self._reward(), (self.player_sum, self.dealer_show)
        else:
            self.player_sum += self.draw('player')
            if self.isEnded():
                return self._reward(), (self.player_sum, self.dealer_show)
            else:
                self.dealer_sum += self.draw('dealer')
                if self.isEnded():
                    return self._reward(), (self.player_sum, self.dealer_show)
                else:
                    return 0, (self.player_sum, self.dealer_show)


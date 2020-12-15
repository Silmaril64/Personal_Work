import random
import numpy as np
# zemmari@labri.fr

class Game:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3
    
    ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT]
    ACTIONS_NAMES = ['UP','LEFT','DOWN','RIGHT']
    
    MOVEMENTS = {
        ACTION_UP: (1, 0),
        ACTION_RIGHT: (0, 1),
        ACTION_LEFT: (0, -1),
        ACTION_DOWN: (-1, 0)
    }
    
    num_actions = len(ACTIONS)
    
    def __init__(self, n, m, wrong_action_p=0.1, alea=False):
        self.n = n
        self.m = m
        self.wrong_action_p = wrong_action_p
        self.alea = alea
        self.generate_game()
        
    def _position_to_id(self, x, y):
        """Gives the position id (from 0 to n)"""
        return x + y * self.n
    
    def _id_to_position(self, id):
        """RÃ©ciproque de la fonction prÃ©cÃ©dente"""
        return (id % self.n, id // self.n)
    
    def generate_game(self):
        cases = [(x, y) for x in range(self.n) for y in range(self.m)]
        hole = random.choice(cases)
        cases.remove(hole)
        start = random.choice(cases)
        cases.remove(start)
        end = random.choice(cases)
        cases.remove(end)
        block = random.choice(cases)
        cases.remove(block)
        
        self.position = start
        self.end = end
        self.hole = hole
        self.block = block
        self.counter = 0
        
        if not self.alea:
            self.start = start
        return self._get_state()
    
    def reset(self):
        if not self.alea:
            self.position = self.start
            self.counter = 0
            return self._get_state()
        else:
            return self.generate_game() 
    
    def _get_grille(self, x, y):
        grille = [
            [0] * self.n for i in range(self.m)
        ]
        grille[x][y] = 1
        return grille
    
    def _get_state(self):
        if self.alea:
            return [self._get_grille(x, y) for (x, y) in
                    [self.position, self.end, self.hole, self.block]]
        return self._position_to_id(*self.position)

    def possible_moves(self, x, y):
        res = []
        if x + 1 < self.n:
            res.append(self.ACTION_UP)
        if y - 1 >= 0:
            res.append(self.ACTION_LEFT)
        if x - 1 >= 0:
            res.append(self.ACTION_DOWN)
        if y + 1 < self.m:
            res.append(self.ACTION_RIGHT)
        
        
        return res
        
    def move(self, action):
        """
        takes an action parameter
        :param action : the id of an action
        :return ((state_id, end, hole, block), reward, is_final, actions)
        """
        self.counter += 1
        if action not in self.ACTIONS:
            raise Exception('Invalid action')
        
        choice = random.random()
        if choice < self.wrong_action_p :
            action = (action + 1) % 4
        elif choice < 2 * self.wrong_action_p:
            action = (action - 1) % 4
            
        d_x, d_y = self.MOVEMENTS[action]
        x, y = self.position
        new_x, new_y = x + d_x, y + d_y

        if self.counter > 190:
            self.position = x, y
            return self._get_state(), -10, True, self.possible_moves(x, y)
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_state(), -1, False, self.ACTIONS
        elif self.block == (new_x, new_y):
            return self._get_state(), -1, False, self.possible_moves(new_x, new_y)
        elif self.hole == (new_x, new_y):
            self.position = new_x, new_y
            return self._get_state(), -10, True, None
        elif self.end == (new_x, new_y):
            self.position = new_x, new_y
            return self._get_state(), 10, True, self.possible_moves(new_x, new_y)
        else:
            self.position = new_x, new_y
            return self._get_state(), -1, False, self.possible_moves(new_x, new_y)
        
    def print(self):
        str = ""
        for i in range(self.n - 1, -1, -1):
            for j in range(self.m):
                if (i, j) == self.position:
                    str += "x"
                elif (i, j) == self.block:
                    str += "o"
                elif (i, j) == self.hole:
                    str += "X"
                elif (i, j) == self.end:
                    str += "W"
                else:
                    str += "."
            str += "\n"
        print(str)        

length = 10



game = Game(length, length, wrong_action_p = 0.00001)
alpha = 0.1 # 10% d'apprentissage par tour donc
gamma = 0.5 # Décroissance exponentielle
Q_values = np.zeros((length * length, game.num_actions), dtype = float)

NB_EPOCHS = 1000
game.print()
for epoch in range(NB_EPOCHS):
    game.reset()
    curr_reward = 0
    curr_move = -1
    future_state = -1
    game_over = False
    possible_actions = game.possible_moves(*game.position)
    state = game._get_state()
    total_reward = 0
    moves = []
    if epoch != NB_EPOCHS - 1:
        mini_random = 0.05
    else:
        mini_random = 0.0
    while not(game_over):
        #print("1", end = "")
        #print("Possible moves:",possible_actions)
        # On choisit la meilleure direction grâce aux informations connues
        curr_weights = np.array([Q_values[state][i] for i in possible_actions], dtype = float)
        tempo = np.min(curr_weights)
        if tempo < 0:
            #print(Q_values[state] - tempo + 0.1)
            curr_move = random.choices(possible_actions, curr_weights - tempo + mini_random, k = 1)[0]
        else:
            curr_move = random.choices(possible_actions, curr_weights + mini_random, k = 1)[0]
        #print("2", end = "")
        # On bouge dans la direction voulue
        future_state, curr_reward, game_over, possible_actions = game.move(curr_move)
        #print("3", end = "")
        total_reward += curr_reward
    
        move_maxi = -9999
        for i in range(game.num_actions):
            if move_maxi < Q_values[future_state][i]:
                move_maxi = Q_values[future_state][i]
        #print("4", end = "")
        Q_values[state][curr_move] = Q_values[state][curr_move] + alpha * (curr_reward +
                                                                            gamma * move_maxi -
                                                                            Q_values[state][curr_move])
        state = future_state
        moves.append(curr_move)
        #print("5")
        #print(game.counter)

    
    print("FINISHED ! TOTAL REWARD:", total_reward)
    #print("PATH:", moves)
game.reset()
print("TERRAIN:",game.print())
print("MOVES:", moves)
print("FINAL Q VALUES:", Q_values)


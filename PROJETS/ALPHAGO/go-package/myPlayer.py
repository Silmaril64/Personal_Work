# -*- coding: utf-8 -*-
''' This is the file you have to modify for the tournament. Your default AI player must be called by this module, in the
myPlayer class.

Right now, this class contains the copy of the randomPlayer. But you have to change this!
'''



import time
import Goban 
from random import choice
import numpy as np
import math
from playerInterface import *


C = math.sqrt(2)
NB_ITER = 1000

class NodeData:
    
    def __init__(self, pere = None, move = None):
        self.number = 0
        self.win = 0
        self.pere = pere
        self.children = []

    def UCT(self):
        if pere != None:
            return self.win/(self.number+1) + C*math.sqrt(math.log(self.pere.number)/(self.number+1))
        else:
            return 1



class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and 
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    '''

    def findBiggestUCT(l):
        maxi = 0
        curr_i = 0
        for i in range(len(l)):
            if l[i].UCT() > maxi:
                maxi = l[i].UCT()
                curr_i = i
        return curr_i

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None

    def randomSampling(self, board):
        if self._board.is_game_over():
            return self._board.result()
        
        moves = self._board.legal_moves()
        if moves != []:
             
             
    
    def myChoice(self):
        start = NodeData()
        
        for i in range(NB_ITER):
            current_node = start
            while len(current_node.children) != 0:
                current_node = current_node.children[findBiggestUCT]
            
            moves = self._board.legal_moves()
            if moves != []:
                for i in range(len(moves)):
                    new_node = NodeData(pere = current_node, move = moves[i])
                    current_node.children.append(new_node)
                    result = randomSampling(self._board.push(moves[i]))
                    self._board.pop()
            
    def getPlayerName(self):
        return "Random Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS" 
        moves = self._board.legal_moves() # Dont use weak_legal_moves() here!
        move = myChoice() 
        self._board.push(move)

        # New here: allows to consider internal representations of moves
        print("I am playing ", self._board.move_to_str(move))
        print("My current board :")
        self._board.prettyPrint()
        # move is an internal representation. To communicate with the interface I need to change if to a string
        return Goban.Board.flat_to_name(move) 

    def playOpponentMove(self, move):
        print("Opponent played ", move) # New here
        # the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move)) 

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")




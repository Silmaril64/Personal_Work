import sys, math
import pygame
import pygame.draw
import random
import numpy as np

#Système de carte d'influence

#Plus on est proche d'une ressource, plus on récolte

#Lorsqu'une ressource est corrompue, il faut que les gens
#qui récoltent dessus aient un drawback:
#- direct -> perte sèche de ressource de la banque ?
#- constant -> unité qui pop régulièrement et se précipite
#sur la base la plus proche pour la piller -> PAS FORCEMENT? ON PEUT JUSTE AVOIR UN PERSO A DEPLACER
# ET QUI PERD DE LA VIE EN MARCHANT SUR DE LA CORRUPTION

# Partie terminée quand seul un survit
# (OU toutes les ressources prises ?
# possiblement deux modes de jeu)
# Score -> ressources !!!

board_size = (50,50) 
square_size = 20
screen_size = (board_size[0] * square_size, board_size[1] * square_size)


MAX_RESSOURCE_VALUE = 5.0
MIN_RESSOURCE_VALUE = 0.5

MAX_NB_RESSOURCES = 5
MIN_NB_RESSOURCES = 3

MAX_CORRUPTION_VALUE = 1.0
MIN_CORRUPTION_VALUE = 1.0

DELTA_THRESHOLD = 0.1
NB_PLAYER = 2
NAMES = ["Joueur", "IA"]
POWERS = [None, 4]
NB_CORRUPTION_POINTS = 20
CORRUPT_P_THRESHOLD = 0.75
CORRUPTION_TOTALE = 1.0 # Chance qu'une case complètement corrompue devienne un générateur
NB_MULT = 10 # Nombre de générateurs produits lors de la corruption d'une ressource
#(rend le farming d'une ressource en danger bien plus dangereux)
REZ = 100.0
ORIGIN_DANGER = 2.0 # Pour construire la carte d'influence de l'IA
DESCRIPTION_IA_LEVEL = [1.0, 0.5, 0.25, 0.1, 0.01] # Nombre de secondes à attendre avant de recalculer une nouvelle stratégie 
     

class Ressource:

    def __init__(self, x, y, value = 1.0):
        self._value = value
        self._x = x
        self._y = y
        self._corrupted = False

    @staticmethod
    def createRandomRessources(i):
        res = []
        for j in range(i):
            res.append(Ressource(random.randint(0,board_size[0] - 1),
                                 random.randint(0,board_size[1] - 1),
                                 random.uniform(MIN_RESSOURCE_VALUE, MAX_RESSOURCE_VALUE)))
            # (PLUS D'ACTUALITE) Pour ne pas avoir à s'occuper des conditions limites plus loin (oui je sais c'est mal)
        return res
    


class Player:

    def __init__(self, name, IA_power = None):
        self._x = random.randint(0, board_size[0] - 1)
        self._y = random.randint(0, board_size[1] - 1)
        self._alive = True
        self._life = 100
        self._points = 0.0
        self._name = name
        self._IA_power = IA_power # None pour un joueur humain, pour un bot, pourrait etre toute les cbien de frame
        #il met à jour ses cartes d'influence
        

        

class Map: # Fait tous les calculs en interne

    def __init__(self,
                 ressources = None,
                 nb_player = NB_PLAYER,
                 name_players = NAMES, power_players = POWERS,
                 speed_corruption = 1,
                 corruption_origins = [(random.randint(0,board_size[0] - 1),
                                        random.randint(0,board_size[1] - 1)) for i in range(NB_CORRUPTION_POINTS)]):
        print("debut init")
        if ressources == None:
            ressources = Ressource.createRandomRessources(random.randint(MIN_NB_RESSOURCES,MAX_NB_RESSOURCES))
        self._ressources = ressources
        self._corrupted = 0
        self._corruption_origins = corruption_origins
        self._corruption_map = np.zeros(board_size, dtype = float)
        self.nb_player = nb_player
        self._speed_corruption = speed_corruption
        self._players = [Player(name_players[i], power_players[i]) for i in  range(nb_player)] 
        self._screen = np.ones((board_size[0], board_size[1],3), dtype = int) * 255

        for r in self._ressources: # Les ressources sont vertes
            self._screen[r._x][r._y][1] = 255
            
        for c in self._corruption_origins: # Les origines de corruption sont violettes
            self._screen[c[0]][c[1]][0] = 255
            self._screen[c[0]][c[1]][2] = 255

        pygame.init()
        self._game = pygame.display.set_mode(screen_size)
        self._font = pygame.font.SysFont('Arial',25)
        print("fin init")


    def spreadCorruption(self):
        CONST = 2
        for i in range(len(self._corruption_origins)):
            origin = self._corruption_origins[i]
            for j in range(self._speed_corruption):
                tempo = random.randint(-CONST, CONST)
                plus = (tempo, random.randint(-(CONST - tempo), CONST - tempo))
                #print(plus)
                #if print != (0,0):
                    #while ((origin[0] + plus[0]) >= 0 and (origin[0] + plus[0]) < board_size[0] and
                           #(origin[1] + plus[1]) >= 0 and (origin[1] + plus[1]) < board_size[1] and
                           #self._corruption_map[origin[0] + plus[0]][origin[1] + plus[1]] == 1.0):
                        # On transfère l'augmentation de corruption de plus en plus loin
                        #origin = (origin[0] + plus[0], origin[1] + plus[1])
                origin = (origin[0] + plus[0], origin[1] + plus[1])
                if ((origin[0]) >= 0 and (origin[0]) < board_size[0] and
                    (origin[1]) >= 0 and (origin[1]) < board_size[1]):
                    # Si on est toujours dans le terrain
                    self._corruption_map[origin[0]][origin[1]] += random.uniform(MIN_CORRUPTION_VALUE, MAX_CORRUPTION_VALUE)
                    if self._corruption_map[origin[0]][origin[1]] > 1.0:
                        # On vérifie si le taux n'est pas supérieur à 1
                        self._corruption_map[origin[0]][origin[1]] = 1.0
                        if (random.random() < CORRUPTION_TOTALE) and plus != (0,0):
                            self._screen[self._corruption_origins[i][0]][self._corruption_origins[i][1]][0] = 255
                            self._screen[self._corruption_origins[i][0]][self._corruption_origins[i][1]][1] = 255*(1 - self._corruption_map[self._corruption_origins[i][0]][self._corruption_origins[i][1]])
                            self._screen[self._corruption_origins[i][0]][self._corruption_origins[i][1]][2] = 255               
                            self._corruption_origins.pop(i)                                                                             
                            self._corruption_origins.append(origin)
                    # Il est temps de peindre la case en violet, proportionnellement à la corruption    
                    c = origin
                    self._screen[c[0]][c[1]][0] = 255 
                    self._screen[c[0]][c[1]][1] = 255 *(1 - self._corruption_map[origin[0]][origin[1]])
                    self._screen[c[0]][c[1]][2] = 255 
        return
                
    def convertRessources(self, n):
        toPop = []
        for k in range(len(self._ressources)):
            ress = self._ressources[k]
            tempo = 0.0
            nb = 0
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    
                    if ((ress._x + i) >= 0 and (ress._x + i) < board_size[0] and
                        (ress._y + j) >= 0 and (ress._y + j) < board_size[1]):
                        nb += 1
                        tempo += self._corruption_map[ress._x + i][ress._y + j]
                        
                    if ((ress._x + i) >= 0 and (ress._x + i) < board_size[0] and
                        (ress._y - j) >= 0 and (ress._y - j) < board_size[1]):
                        nb += 1
                        tempo += self._corruption_map[ress._x + i][ress._y - j]

                    if ((ress._x - i) >= 0 and (ress._x - i) < board_size[0] and
                        (ress._y + j) >= 0 and (ress._y + j) < board_size[1]):
                        nb += 1
                        tempo += self._corruption_map[ress._x - i][ress._y + j]

                    if ((ress._x - i) >= 0 and (ress._x - i) < board_size[0] and
                        (ress._y - j) >= 0 and (ress._y - j) < board_size[1]):
                        nb += 1
                        tempo += self._corruption_map[ress._x - i][ress._y - j]
            if ( tempo - nb * CORRUPT_P_THRESHOLD > 10**(-4)):
                ress._corrupted = True
                self._corrupted += 1
                for j in range(NB_MULT):
                    self._corruption_origins.append((ress._x, ress._y))
                toPop.append(k)
        for k in range(len(toPop)):
            self._ressources.pop(toPop[k])
            for i in range(k+1, len(toPop)):
                if toPop[i] > k:
                    toPop[i] -= 1
        return
                
    def calculatePlayers(self, t): # On pourrait faire en sorte qu'une certaine quantité de points soit dépensée
        #pour soigner les joueurs
        for i in range(len(self._players)):
            p = self._players[i]
            if p._alive:
                tempo = 0.0
                for j in range(len(self._ressources)):
                    r = self._ressources[j]
                    dist = np.sqrt( (p._x - r._x)**2 + (p._y - r._y)**2)
                    if dist != 0:
                        tempo += r._value / dist **2
                    else:
                        tempo += r._value
                p._points += tempo * t
                p._life -= self._corruption_map[p._x][p._y] * t * 33 # Donc environ 3 secondes de survie
                #dans la corruption pure
                if p._life <= 0:
                    if p._points >= REZ:
                        p.points -= REZ
                        p.life = 100
                    else:
                        p._life = 0 # Pour ne pas avoir un nombre négatif constamment à l'écran
                        p._alive = False
                        self._screen[p._x][p._y][0] = 255
                        self._screen[p._x][p._y][1] = 255*(1 - self._corruption_map[p._x][p._y])
                        self._screen[p._x][p._y][2] = 255

    def update(self, t):
        #print("debut update")
        self.calculatePlayers(t)
        #print("fin update")
        self.spreadCorruption()
        
        self.convertRessources(2)
        
        
        for r in self._ressources: # Au cas où il aurait été recouvert par spreadCorruption
            if not(r._corrupted): # Vert
                self._screen[r._x][r._y][0] = 0
                self._screen[r._x][r._y][1] = 255
                self._screen[r._x][r._y][2] = 0
            else: # Noir
                self._screen[r._x][r._y][0] = 255
                self._screen[r._x][r._y][1] = 255
                self._screen[r._x][r._y][2] = 255
            
        for c in self._corruption_origins: # Au cas où il aurait été recouvert par spreadCorruption
            self._screen[c[0]][c[1]][0] = 255
            self._screen[c[0]][c[1]][1] = 0
            self._screen[c[0]][c[1]][2] = 0

        for p in self._players:
            if p._alive:
                self._screen[p._x][p._y][0] = 0
                if p._name == "Joueur":
                    self._screen[p._x][p._y][1] = 255
                else:
                    self._screen[p._x][p._y][1] = 0
                self._screen[p._x][p._y][2] = 255
        return

    def drawMe(self):
        self._game.fill((255,255,255))
        for x in range(board_size[0]):
            for y in range(board_size[1]):
                pygame.draw.rect(self._game, 
                    self._screen[x][y],
                    (x * square_size + 1, y * square_size + 1, square_size - 2, square_size - 2)) #self._screen[x][y]
        s = ""
        current = 0
        for i in range(self.nb_player):
            s = "###" + str(self._players[i]._name) + "###"
            self._game.blit(self._font.render(s,1,(0,0,0)),(0,0+current))
            current += 25
            s = str(self._players[i]._life) + "/100"
            self._game.blit(self._font.render(s,1,(0,0,0)),(0,0+current))
            current += 25
            s = str(self._players[i]._points) + " pts"
            self._game.blit(self._font.render(s,1,(0,0,0)),(0,0+current))
            current += 25 * 2 

    def drawText(self, text, position, color = (0,0,0)):
        self._game.blit(self._font.render(text,1,color),position)


def IA(t, m : Map, p : Player):

    ############### PARTIE CREATION CARTE D'INFLUENCE ###############
    influence = np.zeros(board_size, dtype = float)

    for i in range(board_size[0]):
        for j in range(board_size[1]):
            influence[i][j] -= m._corruption_map[i][j]

    for o in m._corruption_origins:
        
        for i in range(3):
            for j in range(3 - i):
                tempo = 1 + i + j # Harmonique
                influence[o[0] + i][o[1] + j] -=  ORIGIN_DANGER / tempo
                influence[o[0] + i][o[1] - j] -=  ORIGIN_DANGER / tempo
                influence[o[0] - i][o[1] + j] -=  ORIGIN_DANGER / tempo
                influence[o[0] - i][o[1] - j] -=  ORIGIN_DANGER / tempo

    for r in m._ressources:

        for i in range(10):
            for j in range(10 - i):
                tempo = (i + j + 1)**2
                influence[o[0] + i][o[1] + j] +=  ORIGIN_DANGER / tempo
                influence[o[0] + i][o[1] - j] +=  ORIGIN_DANGER / tempo
                influence[o[0] - i][o[1] + j] +=  ORIGIN_DANGER / tempo
                influence[o[0] - i][o[1] - j] +=  ORIGIN_DANGER / tempo

    ############### PARTIE CREATION STRATEGIE (déplacement) ###############

    #Trouver la meilleure case (score le plus haut + le plus proche (en ligne droite, ou si des obstacles
    # utiliser prétraitement))
    # TODO: Si le temps, rajouter des obstacles
    current = - ORIGIN_DANGER * 7 - 1 # Théoriquement, inférieur au minimum atteignable
    # (! pas tout à fait si superposition d'origines !) -> aucune importance en pratique,
    #il existe toujours des cases meilleures dans ce cas-là
    curr_dist = -1
    curr_couple = (-1,-1)

    for i in range(board_size[0]):
        for j in range(board_size[1]):
            if influence[i][j] > current:
                current = influence[i][j]
                curr_couple = (i,j)
            elif influence[i][j] == current:
                tempo = np.sqrt( (p._x - i)**2 + (p._y - j)**2)
                if tempo < curr_dist:
                    curr_dist = tempo
                    current = influence[i][j]
                    curr_couple = (i,j)
    # Maintenant, il faut trouver un bon chemin pour y arriver, et le stocker en attendant de recalculer
    # (penser à le stocker en sens inverse pour pop gratuitement) 
                    

def main():
    game_map = Map()
    done = False
    clock = pygame.time.Clock()
    delta = 0.0
    while done == False:
        x = 0
        y = 0
        ms = clock.tick(30) / 1000
        delta += ms
        game_map.update(ms)
        bs = pygame.key.get_pressed()
        if bs[pygame.K_LEFT]:
            x -= 1
        if bs[pygame.K_RIGHT]:
            x += 1
        if bs[pygame.K_UP]:
            y -= 1 
        if bs[pygame.K_DOWN]:
            y += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done=True
            if event.type == pygame.KEYDOWN:
                pass
            if event.type == pygame.MOUSEBUTTONDOWN:
                pass
                #scene.eventClic(event.dict['pos'],event.dict['button'])
            elif event.type == pygame.MOUSEMOTION:
                pass
                #scene.recordMouseMove(event.dict['pos'])
        game_map.drawMe()
        pygame.display.flip()
        if delta >= DELTA_THRESHOLD:
            delta = 0
            for i in range(len(game_map._players)):
                p = game_map._players[i] 
                if p._name == "Joueur":
                    game_map._screen[p._x][p._y][0] = 255
                    game_map._screen[p._x][p._y][1] = 255 * (1 -  game_map._corruption_map[p._x][p._y])
                    game_map._screen[p._x][p._y][2] = 255
                    if p._x + x >= 0 and p._x + x < board_size[0]:
                        p._x += x
                    if p._y + y >= 0 and p._y + y < board_size[1]:
                        p._y += y
                    
    pygame.quit()

if not sys.flags.interactive: main()
    

import sys, math
import pygame
import pygame.draw
import random
import numpy as np
import heapq
import time

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

#ON POURRAIT ETRE CORROMPU EN MOURANT ET ESSAYER DE TUER LES AUTRES EN LES TOUCHANT !!!

board_size = (75,75) 
square_size = 15
screen_size = (board_size[0] * square_size, board_size[1] * square_size)


MAX_RESSOURCE_VALUE = 5.0
MIN_RESSOURCE_VALUE = 5.0

MAX_NB_RESSOURCES = 25
MIN_NB_RESSOURCES = 15

MAX_CORRUPTION_VALUE = 1.0
MIN_CORRUPTION_VALUE = 1.0

DELTA_THRESHOLD = 0.1
NB_PLAYER = 3
NAMES = ["IA", "IA", "IA"] #"Joueur"
POWERS = [0, 1, 2]#None
NB_CORRUPTION_POINTS = 20
CORRUPT_P_THRESHOLD = 0.75
CORRUPTION_TOTALE = 1.0 # Chance qu'une case complètement corrompue devienne un générateur
NB_MULT = 5 # Nombre de générateurs produits lors de la corruption d'une ressource
#(rend le farming d'une ressource en danger bien plus dangereux)
REZ = 100.0
ORIGIN_DANGER = 2.0 # Pour construire la carte d'influence de l'IA
DESCRIPTION_IA_LEVEL = [1.0, 0.5, 0.25, 0.1, 0.01] # Nombre de secondes à attendre avant de recalculer une nouvelle
# stratégie

SPEED_ALIVE = None # La durée entre chaque mouvement (pour ne pas faire un dieu qui se TP)
SPEED_DEAD = None # Meme chose, mais pour le mode zombie (pour ne pas se téléporter sur le joueur pour l'instakill)
     

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
        self._path = [] # Encore une fois, pour les IA, stocker le chemin calculé à la derniere mise a jour
        self._delta = 0
        self._nb_moves = 1 # Pour bouger sans avoir à reset le delta, utilisé pour accumuler le temps pour la MaJ du path
        

        

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
        self._human_players = [ i for i in range(self.nb_player) if self._players[i]._IA_power == None] 
        self._screen = np.ones((board_size[0], board_size[1],3), dtype = int) * 255
        self._influence_map = None # pour eviter, en cas de multiples IA, de recalculer la même map
        self._best_influence_map = None

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
                        p._points -= REZ
                        p._life = 100
                    else:
                        p._life = 0 # Pour ne pas avoir un nombre négatif constamment à l'écran
                        p._alive = False
                        self._screen[p._x][p._y][0] = 255
                        self._screen[p._x][p._y][1] = 255*(1 - self._corruption_map[p._x][p._y])
                        self._screen[p._x][p._y][2] = 255

    def update(self, t):
        self._influence_map = None
        self._best_influence_map = None
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
            else:
                self._screen[p._x][p._y][0] = 212
                self._screen[p._x][p._y][1] = 235
                self._screen[p._x][p._y][2] = 81
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

    
def AStar(m : Map, p : Player, to_ressources = True):
    if to_ressources:
        #print("A* commencé")
        goal_couple = m._best_influence_map[1]
        if m._best_influence_map[0] <= 0:
            base_cost =  - m._best_influence_map[0] + 0.1 # a ajouter a toutes les cases  pour calculer le poids (donc strict. psitif)
        else:
            base_cost = 0 #pas besoin de shift les valeurs en positif, elles le sont déjà toutes (presque impossible)
        come_from = np.zeros(board_size, dtype = int) # On va faire 1=bas / 2=droite / 3=haut / 4=gauche
                                                      #(pour éviter de devoir stocker des tuples)
        current_min_dist_in_heap = np.ones(board_size, dtype = int) * 65000 # Pour savoir si on doit le rempiler ou non
        
        states = np.zeros(board_size, dtype = int) # not seen == 0 / seen (neighbours) == 1 / integrated == 2
        states[p._x][p._y] = 1 # La case du joueur est celle de départ
        neighbours = [(np.sqrt( (p._x - goal_couple[0])**2 + (p._y - goal_couple[1])**2) + 0,(p._x, p._y))] #np.sqrt( (p._x - i)**2 + (p._y - j)**2) + deja_fait
        heapq.heapify(neighbours)

        while neighbours != []:
            current = heapq.heappop(neighbours)[1]
            #print("CURRENT:", current)
            if current == goal_couple:
                #print("A* fini")
                res_list = []
                while current != (p._x, p._y): # La liste se termine donc par le premier mouvement
                    res_list.append(current)
                    if come_from[current[0]][current[1]] == 1:
                        current = (current[0], current[1] - 1)
                    elif come_from[current[0]][current[1]] == 3:
                        current = (current[0], current[1] + 1)
                    elif come_from[current[0]][current[1]] == 2:
                        current = (current[0] + 1, current[1])
                    elif come_from[current[0]][current[1]] == 4:
                        current = (current[0] - 1, current[1])
                #print("Liste resultat A*:",res_list)
                return res_list
            
            elif states[current[0]][current[1]] == 1: # Si le noeud n'a pas déjà été processed
                                                      # (possible si on rempile une seconde fois
                                                      # un noeud avec une valeur inférieure)
                states[current[0]][current[1]] = 2

                # Maintenant, les voisins (penser à définir ici leur come_from, pour ne pas avoir à le chercher plus tard)
                neighbours_to_check = [((current[0],current[1] + 1),1),((current[0],current[1] - 1),3),
                                       ((current[0] + 1,current[1]),4),((current[0] - 1,current[1]),2)]
                for i in range(4):
                    current_neighbour = neighbours_to_check[i][0]
                    current_direction = neighbours_to_check[i][1]
                    if current_neighbour[0] >= 0 and current_neighbour[0] < board_size[0] and current_neighbour[1] >= 0 and current_neighbour[1] < board_size[1]:
                        #print("le voisin est check")
                        if states[current_neighbour[0]][current_neighbour[1]] != 2:
                            if (states[current_neighbour[0]][current_neighbour[1]] != 0):
                                tempo = current_min_dist_in_heap[current_neighbour[0]][current_neighbour[1]] + (base_cost + m._influence_map[current_neighbour[0]][current_neighbour[1]])
                            else:
                                tempo = (base_cost + m._influence_map[current_neighbour[0]][current_neighbour[1]])
                            #print("TEMPO:",tempo, m._influence_map[current_neighbour[0]][current_neighbour[1]], base_cost)
                            if (current_min_dist_in_heap[current_neighbour[0]][current_neighbour[1]] > tempo) :
                                #print("voisin empilé")
                                current_min_dist_in_heap[current_neighbour[0]][current_neighbour[1]] = tempo
                                heapq.heappush(neighbours,(tempo + np.sqrt( (current_neighbour[0] - goal_couple[0])**2 +
                                                                            (current_neighbour[1] - goal_couple[1])**2),
                                                           (current_neighbour[0], current_neighbour[1])))
                                states[current_neighbour[0]][current_neighbour[1]] = 1
                                come_from[current_neighbour[0]][current_neighbour[1]] = current_direction

    else: # Dead zombie mode not implemented, please buy the DLC at GiveMeYourMoneyPls.fr
        return []
        

def IA(m : Map, p : Player):

    if (p._alive):
    ############### PARTIE CREATION CARTE D'INFLUENCE ###############
        if m._influence_map is None: 
            influence = np.zeros(board_size, dtype = float)

            for i in range(board_size[0]):
                for j in range(board_size[1]):
                    influence[i][j] += m._corruption_map[i][j] * MAX_RESSOURCE_VALUE * len(m._ressources)
                    #TODO: vérifier qu'il n'est pas possible que l'IA attende sur du creep si la récompense est trop bonne

            for o in m._corruption_origins:
        
                for i in range(3):
                    for j in range(3 - i):
                        tempo = 1 + i + j # Harmonique
                        if (o[0] + i) < board_size[0] and (o[1] + j) < board_size[1]:
                            influence[o[0] + i][o[1] + j] +=  ORIGIN_DANGER / tempo
                        if (o[0] + i) < board_size[0] and (o[1] + j) >= 0:
                            influence[o[0] + i][o[1] - j] +=  ORIGIN_DANGER / tempo
                        if (o[0] + i) >= 0 and (o[1] + j) < board_size[1]:
                            influence[o[0] - i][o[1] + j] +=  ORIGIN_DANGER / tempo
                        if (o[0] + i) >= 0 and (o[1] + j) >= 0:
                            influence[o[0] - i][o[1] - j] +=  ORIGIN_DANGER / tempo

            for r in m._ressources:

                for i in range(10):
                    for j in range(10 - i):
                        tempo = (i + j + 1)**2 # Real gain
                        if (r._x + i) < board_size[0] and (r._y + j) < board_size[1]:
                            influence[r._x + i][r._y + j] -=  ORIGIN_DANGER / tempo
                        if (r._x + i) < board_size[0] and (r._y + j) >= 0:
                            influence[r._x + i][r._y - j] -=  ORIGIN_DANGER / tempo
                        if (r._x + i) >= 0 and (r._y + j) < board_size[1]:
                            influence[r._x - i][r._y + j] -=  ORIGIN_DANGER / tempo
                        if (r._x + i) >= 0 and (r._y + j) >= 0:
                            influence[r._x - i][r._y - j] -=  ORIGIN_DANGER / tempo
            m._influence_map = influence
    ############### PARTIE CREATION STRATEGIE (déplacement) ###############
    #Trouver la meilleure case (score le plus haut + le plus proche (en ligne droite, ou si des obstacles
    # utiliser prétraitement))
    # TODO: Si le temps, rajouter des obstacles
        current = 10000 #- ORIGIN_DANGER * 7 - 1 # Théoriquement, inférieur au minimum atteignable
    # (! pas tout à fait si superposition d'origines !) -> aucune importance en pratique,
    #il existe toujours des cases meilleures dans ce cas-là
        curr_dist = -1
        curr_couple = (-1,-1)

        for i in range(board_size[0]):
            for j in range(board_size[1]):
                if m._influence_map[i][j] < current:
                    current = m._influence_map[i][j]
                    curr_couple = (i,j)
                elif m._influence_map[i][j] == current:
                    tempo = np.sqrt( (p._x - i)**2 + (p._y - j)**2) # Garder le plus proche de nous
                    if tempo < curr_dist:
                        print("CHANGEMENT !!! current: ", current, "curr_couple: ", curr_couple)
                        curr_dist = tempo
                        current = influence[i][j]
                        curr_couple = (i,j)
        m._best_influence_map = (current, curr_couple)
        print("current: ", current, "curr_couple: ", curr_couple)
    # Maintenant, il faut trouver un bon chemin pour y arriver, et le stocker en attendant de recalculer
    # (penser à le stocker en sens inverse pour pop gratuitement)

        p._path = AStar(m, p, True)
    else: # player mort, donc mode zombie activé)
        p._path = AStar(m, p, False)
        
                    

def main():
    game_map = Map()
    done = False
    clock = pygame.time.Clock()
    alive_players = 0
    while done == False:
        last_alive = alive_players
        alive_players = 0
        x = 0
        y = 0
        ms = clock.tick(20) / 1000
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
        for i in range(len(game_map._players)):
            p = game_map._players[i]
            if p._alive == True:
                alive_players = i + 1
            p._delta += ms
            if p._name == "Joueur":
                #print("Joueur détecté")
                if p._delta >= DELTA_THRESHOLD:
                    #print("temps écoulé")
                    p._delta = 0

                    #Repeindre l'ancienne case en corruption
                    game_map._screen[p._x][p._y][0] = 255
                    game_map._screen[p._x][p._y][1] = 255 * (1 -  game_map._corruption_map[p._x][p._y])
                    game_map._screen[p._x][p._y][2] = 255

                    # Mettre à jour la position
                    if p._x + x >= 0 and p._x + x < board_size[0]:
                        p._x += x
                    if p._y + y >= 0 and p._y + y < board_size[1]:
                        p._y += y
                
                    
            elif p._name == "IA":
                #print("IA détectée")
                if p._delta * 10 / (DESCRIPTION_IA_LEVEL[p._IA_power]) >= p._nb_moves: # On peut bouger jusqu'à 10 fois entre 2 calculs de path
                    p._nb_moves += 1
                    if p._path != []:
                        #Repeindre l'ancienne case en corruption
                        game_map._screen[p._x][p._y][0] = 255
                        game_map._screen[p._x][p._y][1] = 255 * (1 -  game_map._corruption_map[p._x][p._y])
                        game_map._screen[p._x][p._y][2] = 255
                        #print("Chemin à suivre:",p._path)
                        curr_move = p._path.pop()
                        p._x = curr_move[0]
                        p._y = curr_move[1]
                if p._delta >= DESCRIPTION_IA_LEVEL[p._IA_power]:
                    #print("temps écoulé")
                    IA(game_map, p)
                    p._delta = 0
                    p._nb_moves = 1
                    
        if alive_players == 0:
            done = True
    p = game_map._players[0]   
    game_map._game.blit(game_map._font.render(" Game Finished !!!" ,1,(0,0,0)),(0, screen_size[1] / 2.1))
    pygame.display.flip()
    time.sleep(10)
    pygame.quit()

if not sys.flags.interactive: main()
    
#screen_size

import numpy as np
import math
import random as rand
import matplotlib.pyplot as plt
#from mpmath import *

NB_ITER = 200
NB_CITY = 10
NB_FOURMIS = 10
NB_TESTS = 10
NB_LOOP = 10
THRESHOLD = 0.9

def distance(x1, y1, x2, y2):
    #print(x1, y1, x2, y2, math.sqrt((x2-x1)**2 + (y2-y1)**2))
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def mean_matrix2D(m):
    sum = 0.
    nb = 0
    for i in range(len(m)):
        for j in range(len(m[0])):
            sum += m[i][j]
            nb += 1
    return sum/float(nb)

def std_dev_matrix2D(m, mean):
    sum = 0.
    nb = 0
    for i in range(len(m)):
        for j in range(len(m[0])):
            sum += math.pow(abs(m[i][j] - mean),2)
            nb += 1
    return math.sqrt(sum/float(nb))

def mean_diff_with_max_matrix2D(m):
    total_mean = 0.
    for i in range(len(m)):
        tempo_max = 0
        curr_sum = 0.
        for j in range(len(m[0])):
            if tempo_max < m[i][j]:
                curr_sum += tempo_max
                tempo_max = m[i][j]
        total_mean += (tempo_max *(len(m[0]) - 1) - curr_sum) / float(len(m) * (len(m[0]) - 1))
    return total_mean
        

def choice_chance(gamma, alpha, beta, i, j, g, denom_sum):
    return (gamma + math.pow(g.links[i][j], beta) * math.pow(g.phero[i][j], alpha))/denom_sum


class City:

    def __init__(self, x=-1, y=-1, name=""):
        self.x = x
        self.y = y
        self.name = name
    


class Graph:

    def __init__(self, cities, type_of_graph="dense", links=None):
        self.cities = cities
        tempo = len(self.cities)

        if type_of_graph == "dense":
            self.links = np.ones((tempo,tempo), dtype = float)
        else:
            self.links = links
        for i in range(tempo):
            for j in range(tempo):
                if self.links[i][j]:
                    self.links[i][j] = distance(self.cities[i].x, self.cities[i].y, self.cities[j].x, self.cities[j].y)
        self.phero = np.ones((tempo,tempo), dtype = float) # * -1
        #self.phero = np.zeros((tempo, tempo), dtype = float)

#################################################################################################################
        
def fourmi(g, first_city_nb, Q = 0.25, gamma=0.01, alpha=2, beta=2, evap = 0.1, nb_iter = NB_ITER, nb_fourmis = NB_FOURMIS):
    length = len(g.cities)
    current_path = None
    current_length = None
    means = []
    std_devs = []
    mean_diff_with_max = []
    #print(g.links)
    #print(evap)
    for t in range(nb_iter):
        tempo_phero = np.zeros((len(g.phero),len(g.phero[0])),dtype = float)
        for f in range(nb_fourmis):
            current_city = first_city_nb # an int
            current_length = 0.
            current_path = [current_city]
            r = rand.random()
            possible = [i for i in range(len(g.cities)) if i != current_city]

            while possible != []: # TODO: Je suppose ici qu'on ne peux pas rester bloqué dans un cul de sac (donc graphe dense)
                denom_sum = 0.
                for k in possible:
                    if g.links[current_city][k]:
                        if k != current_city: # En theorie toujours le cas ?
                            if k < current_city: #On ne stocke les pheromones que pour un triangle dans la matrice carrée
                                denom_sum += math.pow(g.links[current_city][k], beta) * math.pow(g.phero[k][current_city], alpha) 
                            else: # donc strictement inférieur
                                denom_sum += math.pow(g.links[current_city][k], beta) * math.pow(g.phero[current_city][k], alpha)   
                denom_sum += gamma * len(possible)

                tempo_sum = 0.
                j = 0
                for j in range(len(possible)):
                    tempo_sum += choice_chance(gamma, alpha, beta, current_city, possible[j], g, denom_sum)
                    if tempo_sum >= r:
                        break
                #print(current_length, end="")
                current_length += g.links[current_city][possible[j]]
                current_city = possible[j]
                current_path += [current_city]
                del possible[j]
            #print("\n")
            current_length += g.links[current_path[-1]][first_city_nb]
            phero_unit = Q/current_length
            if current_length < 1:
                print("CURRENT_LENGTH INFERIEURE A UN !!!",t,f, current_path)
            for i in range(length):# Pheromones
                if current_path[i-1] < current_path[i]:
                    tempo_phero[current_path[i-1]][current_path[i]] += phero_unit
                else:
                    g.phero[current_path[i]][current_path[i-1]] += phero_unit
        for x in range(len(tempo_phero)):
            for y in range(x, len(tempo_phero[0])):
                g.phero[x][y] += tempo_phero[x][y]
        g.phero = g.phero * evap
        #means += [mean_matrix2D(g.phero)]
        #std_devs = [std_dev_matrix2D(g.phero, means[-1])]
        #mean_diff_with_max = [mean_diff_with_max_matrix2D(g.phero)]
    return g#, sum(means) / float(nb_iter), sum(std_devs) / float(nb_iter), sum(mean_diff_with_max) / float(nb_iter)

def read_path(g, first, verb = 1):
    if verb:
        print(g.cities[first].name, end=" ")
    lasts = []
    current = first
    length = len(g.cities)
    res = 0.
    for i in range(length):
        curr_max = 0.
        curr_nb = first

        for j in range(length):
            if current < j:
                if curr_max < g.phero[current][j] and j != current and j not in lasts:
                    curr_max = g.phero[current][j]
                    curr_nb = j
            else:
                if curr_max < g.phero[j][current] and j != current and j not in lasts:
                    curr_max = g.phero[j][current]
                    curr_nb = j
        if verb:
            print(g.cities[curr_nb].name, end=" ")
        res += g.links[current][curr_nb]
        lasts += [current]
        current = curr_nb
    if verb:
        print()
    return res

#############################################################################
        


# Values used in tests:
Q_test = [1]
gamma_test = [0.01, 0.1, 1,10]
alpha_test = [1]
beta_test = [1]
evap_test = [0.99, 0.9, 0.5, 0.1]
#nb_iter_fourmis_test = [(10000,1),(1000,10),(100,100),(10,1000), (1,10000)]
nb_iter_fourmis_test = [(1000,1),(100,10),(10,100),(1,1000)]

#Q = 0.25, gamma=0.01, alpha=2, beta=2, evap = 0.1,




def printTest(liste_x, liste_y, title):
    plt.plot(liste_x, liste_y)
    plt.title(title)
    plt.show()

def isCorrectCircle(res_m):
    lasts = []
    current = 0
    length = len(g.cities)
    for i in range(length):
        curr_max = 0.
        curr_nb = 0
        for j in range(length):
            if current < j:
                if curr_max < g.phero[current][j] and j != current and j not in lasts:
                    curr_max = g.phero[current][j]
                    curr_nb = j
            else:
                if curr_max < g.phero[j][current] and j != current and j not in lasts:
                    curr_max = g.phero[current][j]
                    curr_nb = j
        if (i != curr_nb - 1 and curr_nb != length - i - 1): # Dans un sens ou dans l'autre
            return 0
        lasts += [current]
        current = curr_nb
    return 1

def testCorrectCircle(g, Q, gamma, alpha, beta, evap, nb_iter, nb_fourmis, nb_test = NB_TESTS  * 10, ):
    proba = 0
    length = len(g.phero)
    for i in range(nb_test):
        #g, first_city_nb, Q = 0.25, gamma=0.01, alpha=2, beta=2, evap = 0.1, nb_iter = NB_ITER, nb_fourmis = NB_FOURMIS
        tempo = fourmi(g, 0, Q, gamma, alpha, beta, evap, nb_iter, nb_fourmis)
        tempo = isCorrectCircle(tempo)
        #if tempo == 0:
            #print( g.phero)
        proba += tempo
        #read_path(g,0)
        g.phero = np.ones((length, length), dtype = float)
    proba = proba / float(nb_test)
    print("Proba de trouver la bonne route: ", proba)
    return proba

def testCorrectRandom(g, Q, gamma, alpha, beta, evap, nb_iter, nb_fourmis, nb_test = NB_TESTS  / 10):
    length = len(g.phero)
    nb_min = 0
    mini = 100000
    for i in range(nb_test):
        g = fourmi(g, 0, Q, gamma, alpha, beta, evap, nb_iter, nb_fourmis)
        tempo = read_path(g, 0, 0)
        if tempo < mini:
            mini = tempo
            nb_min = 1
        elif tempo == mini:
            nb_min += 1
        g.phero = np.ones((length, length), dtype = float)
    nb_min = nb_min / float(nb_test)
    return nb_min


def printCities(gc):
    length = len(gc)
    print("cities list:", end="")
    for i in range(length):
        print(gc[i].name,"  x:",gc[i].x," y:", gc[i].y,end="\n")
    print()

def generateCircle(nb, rayon):
    c = []
    angle = 2. * math.pi/ float(nb_city)
    for i in range(nb_city): #Genere un cercle de NB_CITY points
        c += [City(rayon * math.cos(angle * i),rayon * math.sin(angle * i), "c{}".format(i))]
    return Graph(c)

def generateRandom(nb, size):
    c = []
    for i in range(nb):
        c+= [City(size * rand.random(), size * rand.random(), "c{}".format(i))]
    return Graph(c)
################TEST testCorrectCircle ###########################



best = []
max_city = 0
for nb_iter, nb_fourmis in nb_iter_fourmis_test:
        for gamma in gamma_test:
            for alpha in alpha_test:
                for beta in beta_test:
                    for evap in evap_test:
                        for Q in Q_test:
                        #fourmi(g, 0, 1., gamma, alpha, beta,  evap, nb_iter =100,nb_fourmis =100)
                        #g, first_city_nb, Q = 0.25, gamma=0.01, alpha=2, beta=2, evap = 0.1, nb_iter = NB_ITER, nb_fourmis = NB_FOURMIS
                            curr_res = 1.
                            nb_city = 3
                            print("CURRENT TEST:")
                            print("NB_ITER =",nb_iter," // NB_FOURMIS =",nb_fourmis," // GAMMA =",gamma," // ALPHA =",alpha," // BETA =",beta," // EVAP =",evap, " // Q =",  Q)
                            while (curr_res >= THRESHOLD):
                                if nb_city > 21:
                                    break 
                                print(nb_city)
                                curr_res = 0.
                                for i in range(NB_LOOP):
                                    g = generateRandom(nb_city,10)
                                    tempo = testCorrectRandom(g, Q , gamma, alpha, beta, evap, nb_iter, nb_fourmis, nb_test = NB_TESTS)
                                    curr_res += tempo
                                nb_city += 1
                                curr_res /= 10.
                                print("Proba de trouver la bonne route: ", curr_res)
                            if nb_city > max_city:
                                max_city = nb_city
                                best = [[nb_iter, nb_fourmis, gamma, alpha, beta, evap, Q]]
                            elif nb_city == max_city:
                                best += [[nb_iter, nb_fourmis, gamma, alpha, beta, evap, Q]]
                            

print("(PROBABLY) BEST COMBINATION FOR RANDOM: ", best)
print("MAX CITY NUMBER:", max_city)
#Les meilleurs combinaisons (allant jusqu'au maximum de 21): [[1000, 1, 1, 1, 1, 0.99, 1],
#[100, 10, 10, 1, 1, 0.99, 1], [10, 100, 1, 1, 1, 0.5, 1], [10, 100, 1, 1, 1, 0.1, 1], [10, 100, 10, 1, 1, 0.99, 1],
#[10, 100, 10, 1, 1, 0.9, 1], [10, 100, 10, 1, 1, 0.5, 1], [1, 1000, 0.1, 1, 1, 0.9, 1], [1, 1000, 1, 1, 1, 0.99, 1],
#[1, 1000, 1, 1, 1, 0.9, 1], [1, 1000, 10, 1, 1, 0.99, 1], [1, 1000, 10, 1, 1, 0.9, 1], [1, 1000, 10, 1, 1, 0.5, 1],
#[1, 1000, 10, 1, 1, 0.1, 1]]

#On se rend compte que les combinaisons ayant les meilleurs résultats ont souvent énormément de fourmis,
#ce qui indique que la phase d'évaporation est une perte de performance. Elle est cependant nécessaire
#pour éviter un overflow. Cette indication est renforcée par le fait que les modèles avec très peu de
#fourmis ont des scores d'évaporation très faibles.

"""best = []
max_city = 0
for nb_iter, nb_fourmis in nb_iter_fourmis_test:
        for gamma in gamma_test:
            for alpha in alpha_test:
                for beta in beta_test:
                    for evap in evap_test:
                        for Q in Q_test:
                        #fourmi(g, 0, 1., gamma, alpha, beta,  evap, nb_iter =100,nb_fourmis =100)
                        #g, first_city_nb, Q = 0.25, gamma=0.01, alpha=2, beta=2, evap = 0.1, nb_iter = NB_ITER, nb_fourmis = NB_FOURMIS
                            curr_res = 1.
                            nb_city = 10
                            print("CURRENT TEST:")
                            print("NB_ITER =",nb_iter," // NB_FOURMIS =",nb_fourmis," // GAMMA =",gamma," // ALPHA =",alpha," // BETA =",beta," // EVAP =",evap, " // Q =",  Q)
                            while (curr_res >= THRESHOLD):
                                if nb_city > 21:
                                    break 
                                print(nb_city)
                                g = generateCircle(nb_city, 1)
                                #printCities(g.cities)
                                #Q, gamma, alpha, beta, evap, nb_iter, nb_fourmis, nb_test = NB_TESTS  * 10, g = g
                                curr_res = testCorrectCircle(g, Q , gamma, alpha, beta, evap, nb_iter, nb_fourmis, nb_test = NB_TESTS  * 10)
                                nb_city += 1
                            if nb_city > max_city:
                                max_city = nb_city
                                best = [[nb_iter, nb_fourmis, gamma, alpha, beta, evap]]
                            elif nb_city == max_city:
                                best += [[nb_iter, nb_fourmis, gamma, alpha, beta, evap]]
                            

print("(PROBABLY) BEST COMBINATION FOR CIRCLE: ", best)
print("MAX CITY NUMBER:", max_city)"""
###################### TEST GAMMA ###################################
"""
def test(Q_test, gamma_test, alpha_test, beta_test, evap_test, nb_test = NB_TESTS, g = g, f1 = 0, nb_iter = NB_ITER):
    full_res = []
    length = len(g.phero)
    for Q in Q_test:
        Q_res = []
        for gamma in gamma_test:
            gamma_res = []
            for alpha in alpha_test:
                alpha_res  = []
                for beta in beta_test:
                    beta_res = []
                    for evap in evap_test:
                        beta_res += [fourmi(g,f1,Q,gamma,alpha,beta,evap, nb_iter)]
                        g.phero = np.ones((length, length), dtype = float)
                        for i in range(nb_test -1 ):
                            tempo = fourmi(g,f1,Q,gamma,alpha,beta,evap, nb_iter)
                            beta_res[-1] = [beta_res[-1][0], beta_res[-1][1] + tempo[1], beta_res[-1][2] + tempo[2],beta_res[-1][3] + tempo[3]]
                            g.phero = np.ones((length, length), dtype = float)
                        beta_res[-1] = [beta_res[-1][0],beta_res[-1][1] / float(NB_TESTS),beta_res[-1][2]/ float(NB_TESTS),beta_res[-1][3]/ float(NB_TESTS)]
                        print(".", end="")
                    alpha_res += [beta_res]
                    print("**", end="")
                gamma_res += [alpha_res]
            #print("gamma_res:",gamma_res)
                print("---", end="")
            Q_res += [gamma_res]
            print("&&&&", end="")
        full_res += [Q_res]
        print("#####")
    print("\n!!! FIN DE LA FONCTION TEST !!!")
    return full_res

print("START TEST GAMMA")
full_res = test(Q_test, gamma_test, alpha_test, beta_test, evap_test)
liste_mean = []
liste_std_dev = []
liste_diff = []
for i in range(len(gamma_test)):
    liste_mean.append(full_res[0][i][0][0][0][1])
    liste_std_dev.append(full_res[0][i][0][0][0][2])
    liste_diff.append(full_res[0][i][0][0][0][3])

printTest(gamma_test, liste_mean, "MEAN")
printTest(gamma_test, liste_std_dev, "STANDARD DEVIATION")
printTest(gamma_test, liste_diff, "MEAN DIFFERENCE WITH MAX")
gamma_test = [0.01]

###################### TEST EVAP ###################################
print("START TEST EVAP")
evap_test = []
for i in range(100):
    evap_test += [1.0/100 * i]
full_res = test(Q_test, gamma_test, alpha_test, beta_test, evap_test)
#print(full_res)
liste_mean = []
liste_std_dev = []
liste_diff = []
for i in range(len(evap_test)):
    #print(full_res[0][0][0][0][i])
    liste_mean.append(full_res[0][0][0][0][i][1])
    liste_std_dev.append(full_res[0][0][0][0][i][2])
    liste_diff.append(full_res[0][0][0][0][i][3])
#print(liste_mean)
printTest(evap_test, liste_mean, "MEAN")
printTest(evap_test, liste_std_dev, "STANDARD DEVIATION")
printTest(evap_test, liste_diff, "MEAN DIFFERENCE WITH MAX")
evap_test = [0.1]
#read_path(fourmi(g, 0)[0],0)



#TODO: FAIRE DES STATS (genre combien de fois ca passe, la moyenne
# et l'ecart-type des pheromones pour differentes variables et NB_CITY, ...

#TODO: Ptet utiliser une valeur threshold ou l'on augmente NB_CITY
# jusqu'a passer en dessous pour chaque genre de test"""

        
        

        
        
            
        
        
    

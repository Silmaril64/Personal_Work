import random

def chaleur1(n,t):
    return random.random() < 1. / pow(t+1, n)

def chaleur2(t):
    return random.random() < 1. / (t + 1.)

def inTabou(couple, tabou):
    for i in range(len(tabou)):
        if couple[0] == tabou[i][0] and couple[1] == tabou[i][1]:
            return 1
    return 0

def calculateConflicts(index, places):
    #print("index dans calculateConflicts:", index)
    length = len(places)
    conflicts = [0 ,[]]
    iter = range(length)
    for i in iter:
        if i != index[0]:
            if index[1] == places[i] or abs(index[0] - i) == abs(index[1]-places[i]): # Il y a aussi les diagonales
                conflicts = [conflicts[0] + 1, conflicts[1] + [i]]
    return conflicts

def local(places, nb_iter, f_chaleur, nb_tabou): # Le terrain est considéré carré
    length = len(places)
    conflicts = [[0, []]] * length
    tabou = []
    curr = -1
    conf = []
    for i in range(length):
        for j in range(i+1, length):
            if places[j] == places[i] or j - i == abs(places[j]-places[i]): # Il y a aussi les diagonales
                conflicts[i] = [conflicts[i][0] + 1, conflicts[i][1] + [j]]
                conflicts[j] = [conflicts[j][0] + 1, conflicts[j][1] + [i]]
    print(conflicts)
                
    for i in range(nb_iter):
        conf = []
        print("DEBUT ITERATION ",i," :\n\n")
        if f_chaleur(i): # On ne check pas tabou, après tout c'est aléatoire
            curr = random.randint(0,length - 1)
            places[curr] = random.randint(0, length - 1)
        else: # Checker tabou, et chercher une bonne place ( ??? ou est-ce que je check tabou ??? )
            print(conflicts)
            max_c = conflicts[0][0]
            curr = 0
            other_curr = []
            for j in range(1,length): #Sélectionner l'une des reines à conflit maximum
                if max_c < conflicts[j][0]:
                    max_c = conflicts[j][0]
                    curr = j
                    other_curr = []
                elif max_c == conflicts[j]: # Pile ou Face (!!! avantage les derniers !!!)
                    if random.random() < 0.5:
                        max_c = conflicts[j][0]
                        other_curr.append(curr)
                        curr = j
                        
            min_conflict = length + 1 # plus de conflit que le nombre maximal de conflit
            curr_place = -1
            if max_c > 0:
                while "toujours pas trouvé un emplacement respectant les tabous":
                    for j in range(length): # On va chercher un endroit où mettre la reine, en respectant les tabous
                        if not inTabou((curr, j), tabou): 
                            conf = calculateConflicts((curr, j), places)
                            if min_conflict > conf[0]:
                                min_conflict = conf[0]
                                curr_place = j

                    if curr_place == -1:
                        if other_curr != []:
                            curr = other_curr.pop()
                        else: #sinon, il faudrait rechercher la valeur en dessous,
                              #mais on va pour l'instant faire aléatoire dans ce cas-là,
                              #qui ne devrait en théorie jamais arriver (a moins d'avoir
                              #une liste de tabous énorme)
                            curr = random.randint(0,length - 1)
                            places[curr] = random.randint(0, length - 1)
                            conf = []
                    else: # On a donc trouvé un couple satisfaisant
                        places[curr] = j
                        break
            else: #Si on a deja une solution complète
                break
                
                        
                    

        print(curr, places[curr])
        for j in conflicts[curr][1]: # Le -1 sur les conflits précédents (je suis parti
                                     #(a moins d'etre tombé sur la meme case, peu probable))
            conflicts[j][0] -= 1
            conflicts[j][1].pop(conflicts[j][1].index(curr))

        if conf != []:  
            conflicts[curr] = conf # On remplace l'ancienne par la nouvelle
        else:
            conflicts[curr] = calculateConflicts((curr,places[curr]), places)
            
        for j in conflicts[curr][1]: # Mise à jour des autres par rapport aux nouveaux conflits
            conflicts[j][0] += 1
            conflicts[j][1].append(curr)

        if len(tabou) < nb_tabou:
            tabou.append((curr, places[curr]))
        else:
            tabou.pop(0)
            tabou.append((curr, places[curr]))

    return places

places = [random.randint(0,8) for i in range(9)]
print("places des reines :", places, "\n")
local(places, 1000, lambda t: chaleur1(2,t), 5)
            



class Rule():
    
    def __init__(self, strRule):
        self.active = False
        self.pre, self.post = [], []
        [strPre,separator,strPost] = strRule.partition("=>")
        if separator == "": # implication not found
          raise Exception("Missing '=>' in rule") 
        self.pre = [s.strip() for s in strPre.split("&")]
        self.post = [s.strip() for s in strPost.split("&")]
        self.active = True
        
    def __str__(self):
        return str(self.pre) + " => " + str(self.post)



class KB():

    def __init__(self, verb = True):
        self.facts=[]
        self.rules=[]
        self.verb = verb

    def addRule(self,r):
        self.rules.append(r)

    def addFact(self,f):
        print("Adding fact ", f)
        self.facts.append(f)

    def rectractFact(self,f):
        self.facts.remove(f)

    def _propagateRule(self,rule):
        empty = True
        ret = False
        for pre in rule.pre:
            if self.verb: print("evaluating ", pre)
            if pre not in self.facts:
                empty = False
                break
        if empty:
            if self.verb: print("Propagating rule " + str(rule.pre) + " => " + str(rule.post))
            for post in rule.post:
                if post not in self.facts:
                    if self.verb: print("Adding " + post + " as a new fact") 
                    self.addFact(post)
                    ret = True # At least one fact has been added
            rule.active = False
        return ret

    def simpleFC(self):
        "Simple Forward Checking algorithm. No smart data structure used"
        loop = True # True if any fact has been added
        while loop:
            loop = False
            for rule in self.rules:
                if rule.active:
                    loop |= self._propagateRule(rule)

    def _getRulesForFact(self, fact):
        "Returns the list of rules that contains this fact as a post"
        return [rule for rule in self.rules if fact in rule.post]

    def _ask(self, f):
        "Asks for the value of a fact"
        print ("Asking for " + str(f) + " to become a new fact")
        answer = input()
        # print("Forcing " + str(f) + " as a new Fact")
        if answer == "Yes" or answer == "Y" or answer == "y" or answer == "yes":
            print("Fact added")
            self.addFact(f) # By default make it true
            return True
        print("Fact refused")
        return False

    def simpleBC(self, fact):
        "Simple Backward chaining for a fact, returns after any new fact is added after any question"
        print("BC for fact " + str(fact))
        for rule in self._getRulesForFact(fact):
            print(rule)
            for pre in rule.pre:
                if pre not in self.facts:
                    rulespre = self._getRulesForFact(pre)
                    if not rulespre: # no rules conclude on it. This is an askable fact
                        res = self._ask(pre)
                        if res:
                            return True
                    else:
                        return self.simpleBC(pre)
        return False




kb = KB()
#kb.addRule(Rule("Toto & Titi => yy & jj"))
#kb.addRule(Rule("yy & Titi => tt"))
#kb.addRule(Rule("jj & tt => youpi"))
#kb.addRule(Rule("jj & ttt => pryoupi"))
#kb.addFact("Toto")
#kb.addFact("Titi")
print("Création du système expert sur la vie")
print("Ajout des règles")

kb.addRule(Rule("mangé => rassasié"))
kb.addRule(Rule("dormi => reposé"))

kb.addRule(Rule("couple => amour"))
kb.addRule(Rule("mariage => banquet & amour"))
kb.addRule(Rule("discussion => interaction"))
kb.addRule(Rule("repas => mangé"))
kb.addRule(Rule("banquet => bien mangé"))

kb.addRule(Rule("interaction => joie sociale"))
kb.addRule(Rule("bien dormi => dormi & joie physique"))
kb.addRule(Rule("bien mangé => mangé & joie physique"))
kb.addRule(Rule("apprentissage & intelligent => joie mentale"))

kb.addRule(Rule("amour => heureux"))
kb.addRule(Rule("joie physique => partiellement heureux"))
kb.addRule(Rule("joie sociale  => partiellement heureux"))
kb.addRule(Rule(" joie mentale => partiellement heureux"))
kb.addRule(Rule("joie sociale & joie physique => heureux"))
kb.addRule(Rule("joie sociale & joie mentale => heureux"))
kb.addRule(Rule(" joie physique & joie mentale => heureux"))
kb.addRule(Rule("joie sociale & joie physique & joie mentale => completement heureux"))
kb.addRule(Rule("completement heureux => heureux")) #useless, tant qu'une autre manière d'être heureux n'existe pas
kb.addRule(Rule("heureux => partiellement heureux")) #useless, tant qu'une autre manière d'être heureux n'existe pas

kb.addRule(Rule("rassasié & reposé => vivant"))
kb.addRule(Rule("vivant & complètement heureux => plénitude"))

print("Ajout des faits de base") 

print("Fin de la création du système expert")
print("Début de la simulation") #ptet faire une boucle en fin qui transforme les "rassasié 100%" en "rassasié 75%"
# faits à rechercher dans l'ordre de priorité: vivant -> plénitude
# faits constants (à définir dès le début): couple
kb.simpleFC()
print(1)

res = True
while res:
    res = kb.simpleBC("youpi")
print(2)
kb.simpleFC()
print(3)

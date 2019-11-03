#ifndef GRILLE_H_INCLUDED
#define GRILLE_H_INCLUDED

#ifndef TAILLE
#define TAILLE 20
#endif

struct terrain{
    int visible[TAILLE*TAILLE];
    int bombs[TAILLE*TAILLE];
};

int afficher_grille(struct terrain t, int *win);

int jouer_coup(struct terrain *t,int i, int j);

struct terrain random_gen(int seed, int bombs);

int coup_possible(struct terrain t, int i, int j);

#endif // GRILLE_H_INCLUDED

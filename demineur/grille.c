#ifndef GRILLE_C_INCLUDED
#define GRILLE_C_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "grille.h"

int compter(struct terrain t, int i, int j){ //returns the number of bombs around the selected cell

    int tempo = 0;
    if (j<TAILLE-1){
        tempo += t.bombs[i*TAILLE+j+1];
        if (i<TAILLE-1){
            tempo += t.bombs[(i+1)*TAILLE+j+1];
        }
        if (i>0) {
            tempo += t.bombs[(i-1)*TAILLE+j+1];
        }
    }
    if (j>0){
        tempo += t.bombs[i*TAILLE+j-1];
        if (i<TAILLE-1){
            tempo += t.bombs[(i+1)*TAILLE+j-1];
        }
        if (i>0) {
            tempo += t.bombs[(i-1)*TAILLE+j-1];
        }
    }
    if (i<TAILLE-1){
        tempo += t.bombs[(i+1)*TAILLE+j];
    }
    if (i>0){
        tempo += t.bombs[(i-1)*TAILLE+j];
    }
    return tempo;
}

int afficher_grille(struct terrain t, int *win){
    int res = 0;
    *win = 0;
    for (int i = 0;i<TAILLE ;i++){
            if (!(i%5)){
                printf("\n");
            }
        for (int j = 0 ;j<TAILLE ;j++){
            if (!(j%5)){
                printf(" ");
            }
            if (t.visible[i*TAILLE+j] == 1){
                if (t.bombs[i*TAILLE+j]){
                    printf("B ");
                    res = 1;
                } else {
                    printf("%d ",compter(t,i,j));
                }
            } else if (t.visible[i*TAILLE+j]){ //donc == 2, car soit 0, soit 1, soit 2
                printf("# ");
            } else {
                printf("X ");
                if (!t.bombs[i*TAILLE+j]){
                    (*win) ++;
                }
            }
        }
        printf("\n");
    }
    return res;
}

int spreading(struct terrain *t, int i, int j){ //NE PAS REVELER EN DIAGONALE !!!
    if (!compter(*t,i,j)){ //dans le cas où il n'y a pas de bombes voisines, on spread la révélation sur toute la surface libre.
        t->visible[i*TAILLE+j] = 1;
        if (j<TAILLE-1 && !t->visible[i*TAILLE+j+1]){
                spreading(t,i,j+1);
        }
        if (j>0 && !t->visible[i*TAILLE+j-1]){
            spreading(t,i,j-1);
        }
        if (i<TAILLE-1 && !t->visible[(i+1)*TAILLE+j]){
            spreading(t,i+1,j);
        }
        if (i>0 && !t->visible[(i-1)*TAILLE+j]){
            spreading(t,i-1,j);
        }
        return 0;
    }
    t->visible[i*TAILLE+j] = 1; //dans le cas ou il y a une bombe voisine, on ne devoile que la case cliquée
    return 0;
}

int jouer_coup(struct terrain *t ,int i ,int j){
    if (!coup_possible(*t,i,j)){
        return 1;
    }
    spreading(t,i,j);
    return 0;
}

struct terrain random_gen(int seed, int bombs){
    struct terrain t = {{},{}};
    if (bombs <= TAILLE*TAILLE){
            printf("HEHO\n");
        int x,y;
        if (seed == -1){
            srand(time(0)); //randint()
        } else {
            srand(seed);
        }
        for (int i = 0 ;i<bombs ;i++){
            x = rand()%TAILLE;
            y = rand()%TAILLE;
            while (t.bombs[x*TAILLE+y]){
                x = rand()%TAILLE;
                y = rand()%TAILLE;
            }
            t.bombs[x*TAILLE+y] = 1;
        }
    }
    return t;
}

int coup_possible(struct terrain t, int i, int j){
    return (!(i<0 || i>=TAILLE || j<0 || j>=TAILLE || t.visible[i*TAILLE+j] == 1));
}

#endif // GRILLE_C_INCLUDED

#ifndef TEST_C_DEFINED
#define TEST_C_DEFINED

#include <stdio.h>
#include <stdlib.h>
#include "grille.h"

int test_afficher_grille(){
    int res = 1;
    int win;
    struct terrain t = {{0,0,1,0,0,1,0,1,0}, {0,0,1,1,0,0,1,0,0}};
    res = (res && afficher_grille(t,&win) == 1);
    struct terrain t1 = {{0,0,0,0,0,1,0,1,0}, {0,0,1,1,0,0,1,0,0}};
    res = (res && afficher_grille(t1,&win) == 0);
    return (!res);
}

int tests(){
    return (test_afficher_grille());
}

#endif

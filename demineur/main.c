#include "grille.h"
#include "test.h"



int main()
{
    struct terrain t = random_gen(-1,20);
    int game = 1;
    int i = 0;
    int j = 0;
    int type = 0;
    int win = 0;
    //return (tests());
    while (game){

        printf("Saisissez X:");
        scanf("%d",&i);
        printf("Saisissez Y:");
        scanf("%d",&j);
        printf("\n");
        while (!coup_possible(t,i,j)){
            printf("Saisissez X:");
            scanf("%d",&i);
            printf("Saisissez Y:");
            scanf("%d",&j);
            printf("\n");
        }
        printf("Quel type ? (0:vide, 1:gerer le marqueur mine):");
        scanf("%d",&type);
        if (type){
            t.visible[i*TAILLE+j] = (t.visible[i*TAILLE+j] + 2 )%4;
        } else if (t.visible[i*TAILLE+j] == 0){
            jouer_coup(&t,i,j);
        }
        game = !afficher_grille(t,&win);
        if (!win){
            printf("BRAVOOOOOOOOOOO\n");
            return 0;
        }
    }
    printf("T'ES NUUUUUUUL\n");
    return 0;
}

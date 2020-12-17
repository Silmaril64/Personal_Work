import gzip, os.path
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Reshape
import numpy as np

epochs = 10
batch_size = 1024
model = tensorflow.keras.models.load_model('.')

def reconstruct_lists_from_string(s):
    """Je ne prend pas en compte les nombres à plus de 1 chiffre, car je n'en ai ici pas besoin;
    Je dois par contre prendre en compte les nombres négatifs"""
    res = []
    signe = 1
    final = []
    for i in range(len(s)):
        if s[i] == '[':
            if len(res) == 0:
                print(".", end="")
            res.append([])
        elif s[i] == ']':
            tempo = res.pop()
            if len(res) != 0:
                res[-1].append(tempo)
            else: 
                final.append(tempo)
        elif s[i] == '1' or s[i] == '0':
            #print(".")
            res[-1].append(signe*int(s[i]))
            signe = 1
        elif s[i] == '-':
            signe = -1
        else: # Si espace ou virgule
            pass
        
    return final

        
    

with open('./data/100_iter.json', 'r') as jsonfile:
    #data = json.load(jsonfile)
    data = jsonfile.read()
    data = reconstruct_lists_from_string(data)
    tempo = []
    for x in data:
        tempo += x
    data = tempo

    
X_data = [data[i][0] for i in range(len(data)) ]
Y_data = [data[i][1] for i in range(len(data)) ]
#print(data)
#print(X_data)
#print(Y_data)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1)

X_train = np.asarray(X_train).astype(np.float32)
Y_train = np.asarray(Y_train).astype(np.float32)





history = model.fit(X_train,Y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, verbose=1)

model.save('.')

import numpy as np # importerar numPY som np
import matplotlib.pyplot as plt # importerar matplotlib som plt
import os # importerar så att man kan använda os röst

# point 1 = red flower  point 0 = blue flower

# length, width, type/color
data = [[3,    1.5,  1],
        [2,    1,    0],
        [4,    1.5,  1],
        [3,    1,    0],
        [3.5,  0.5,  1],
        [2,    0.5,  0],
        [5.5,  1,    1],
        [1,    1,    0]]

mystery_flower = [4.5, 1] # påhittad blomma som man inte var värde på vilken färg det är



def sigmoid(x):
    return(1/(1 + np.exp(-x))) # en sigmoid funktion som delas på exponetialfunktionen av x

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x)) # här multiplicerar man sigmoid x med sigmoid x -1

T = np.linspace(-6, 6, 100)

plt.plot(T, sigmoid(T), c='r') # "plottar" hur funktionen ser ut 
plt.plot(T, sigmoid_p(T), c='b') # -----------||----------------


# Training loop

for i in range(1, 1000): # tränings loop som inte inte behövs
    ri = np.random.randint(len(data))

#scatter data # värderna på blommorna vi har alla värden på och lagt upp dem på en graf

plt.axis([0, 6, 0, 6])
plt.grid()

for i in range(len(data)):
    point = data[i]
    color = "r"
    if point[2] == 0:
        color = "b"
    plt.scatter(point[0], point[1], c=color)

# training loop # här börjar då den riktiga tränings loopen

learning_rate = 0.5 # learning rate är hur snabbt AIn lär sig 
costs = [] # används för att visa alla värden vi får ut

w1 = np.random.rand() # "vikt 1"
w2 = np.random.rand() # "vikt 2"
b = np.random.rand() # bias används för att balansera båda "vikterna"
 
for i in range(50000):
    ri = np.random.randint(len(data))
    point = data[ri] #antal "tal" som AIn ska gå igenom
    
    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)
    
    target = point[2]
    cost = np.square(pred - target)
    
    
    dcost_pred = 2 * (pred -target)
    dpred_dz = sigmoid_p(z)
    
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1
    
    dcost_dz = dcost_pred * dpred_dz
    
    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db 
    
    w1 = w1 - learning_rate * dcost_dw1   #
    w2 = w2 - learning_rate * dcost_dw2   #
    b = b - learning_rate * dcost_db      #
                                          # Här tar man learning rate multiplicerat med "kostnaden" minus "vikten" eller "bias" 
    if i % 100 == 0:
        cost_sum = 0
        for j in range(len(data)):
            point = data[ri]
            
            z = point[0] * w1 + point[1] * w2 +b
            pred = sigmoid(z)
            
            target = point[2]
            cost_sum += np.square(pred - target)
            
        costs.append(cost_sum/len(data)) # Tillhör "costs = []" som används för att visa värden vi får ut efter AIn är klar

plt.plot(costs) # "plottar" upp hur grafen ser ut    

#seeing model prediction
#seeing model prediction
# Visar vilke färg blomman har, längd, bredd och exakta värde. allt över 0.5 är röd och allt under 0.5 är blå

for i in range(len(data)):
    point = data[i]
    print(point)
    z = point[0] * w1 + point[1] * w2 +b
    pred = sigmoid(z)
    print("pred:{}".format(pred))

for i in range(len(data)):
    point = data[i]
    print(point)
    z = point[0] * w1 + point[1] * w2 +b
    pred = sigmoid(z)
    print("pred:{}".format(pred))

z = mystery_flower[0] * w1 + mystery_flower[1] * w2 +b # här tar vi reda på vilken färg den "mystiga blomman" har
pred = sigmoid(z)
pred

# Här använder vi en metod som berättar vilken färg blomman har (ganska onödig)
def which_flower(length, width):
    z = lenght * w1 + width * w2 + b
    pred = sigmoid(z)
    if pred < 0.5:
        os.system("say blue")
    else:
        os.system("say red")

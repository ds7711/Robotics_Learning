
1. Q function:
Q(s, a) = expected return of taking action a under state s

    An object that initialize with a list:
        # of items in the list: number of layers in the neural network
        The integers in the list specifies the number of units in each layer

    The object should have a batch training function and save function to save intermediate weights

     The object should have a method to load previous trained weights


2. Naive Policy function: action selection using softmax or epsilon greedy
    May be relatively slow because it has to be performed many many times

3. More advanced policy function (???)
  How to select an action base on the Q function.


4. Convert the trajectory into the training data that can be used for Q function


5. Function that generate the trajectories
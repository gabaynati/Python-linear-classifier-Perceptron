import numpy as np
import copy as copy
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore');

#threshold is used to classify to class 0 or class 1
threshold = 0.5;
learning_rate = 0.1;
w = np.array([0, 0, 0]);
#traning set is used to learn the algorithm the desired output.
training_set = np.array([(np.array([1, 0, 0]), 1), (np.array([1, 0, 1]), 1), (np.array([1, 1, 0]), 1), (np.array([1, 1, 1]), 0)]);




while (True):
    previous_w=copy.deepcopy(w);#saving the last weights vector to check if it updated during the last 2 iterations.


    #iterating thruogh all the examples in the training set
    for x in training_set:
        s=np.dot(x[0],w);#dot product : WX

        #n is the network
        if(s>threshold):
            n=1
        else:
            n=0;

        #e is the error between the calculation and the actual desired result
        e=x[1]-n;

        #d is the correction needed
        d=learning_rate*e;

        #updating the weights vector
        w=np.add(w,x[0]*d);



    #printing the figure

    #ploting the dots
    plt.plot(0, 0,'or');
    plt.plot(0, 1,'or');
    plt.plot(1, 0,'or');
    plt.plot(1, 1,'or');



    #ploting the x2 as function of x1
    x1 = np.arange(-1, 2.5, 1);
    try:
        x2=((w[0]+x1*w[1]-threshold)/(w[2]))*(-1);
    except RuntimeError:
        pass
    plt.plot(x1, x2);
    plt.xlabel('x1', fontsize=20);
    plt.ylabel('x2', fontsize=20);
    plt.title(w, fontsize=20);
    plt.ylim(-1, 2);
    plt.show();


    #the algotithm is done when the weights did not changed during the last two iteration on the training set
    if(np.array_equal(previous_w,w)):
        break;



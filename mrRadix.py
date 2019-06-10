# radix_conversion returns a list of Radix-X weights by converting a pre-trained weights vector W
# into a set of normalized and discretized values

import math
import numpy as np

#x = input("Enter X for Radix-X:")
#input_string = input("Enter list of weights separated by a space:")
#W = list(map(int, input_string.split()))



#######################################Jason code########################################
x = 5
W = [5.5, 10.2, 2.3, 5, 13.5]

# X is radix-X value; W = vector containing pre-trained weights


def normalized_matrix(x, W):

    # normalize weights to a min of 0 and max of 1
    y = np.true_divide((np.add(W, -min(W))), (max(W)-min(W)))

    # normalize weights to a min of -X/2 and max of X/2
    normalized = np.add((np.multiply(y,x)), -x/2)

    return normalized


def quantized_matrix(normalized_matrix):
    # discretize floating weights: round negative weights up and positive weights down

    # initialize quantized matrix
    Y = []

    # getting length of weight matrix
    length = len(normalized_matrix)

    for n in range(length):
        for normalized_weight in normalized_matrix:
            if normalized_weight < 0:
                Y.append(math.ceil(normalized_weight))
            else:
                Y.append(math.floor(normalized_weight))
        return Y

print("Radix wieght(Jason) : ",quantized_matrix(normalized_matrix(x,W)))

############################################################################






##############################Jaeheum's code#################################
#nw = 1 #(range : -1,0,1) equal to (x = 3) on your code
nw = 2 #(range : -2,-1,0,1,2) equal to (x = 5)
#nw = 3 #(range : -3,-2,-1,0,1,2,3) equal to (x = 7)

W = [5.5, 10.2, 2.3, 5, 13.5]


def map(W, a, b, min, max):
    # a------W--b => min------------W----max // 선형 변환(linear mapping)
    #W = -(min*(W-b) + max*(a-W))/(a-b)
    W = (W-a)*(max-min)/(b-a) + min
    return W


def radix(W, nw):
    W_max = np.max(W)  # find Max value
    W_min = np.min(W)  # find min value

    W = map(W, W_min, W_max, -nw, nw)

    return np.round(W)

print("Radix wieght(Jaheum) : ",radix(W, nw))




def radix_ReLu(W, nw): #Here, W doesn't mean weight. W means Ofcourse output of convolution. Don't confuse.
    W = np.maximum(W, 0) #ReLu activation. I've couldn't find ReLu() function in numpy library. But tensorflow offers that.

    W_max = np.max(W)
    W_min = np.min(W)

    W = map(W, W_min, W_max, 0, nw*2)

    return np.round(W)

print("Radix activation : ",radix_ReLu(W, nw))


# In real code, tensorflow function is used instead of numpy finction. Because numpy can't drives tensor.
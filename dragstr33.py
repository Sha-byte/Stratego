import pickle
import numpy as np
import random



a_file = open('strategi33.pkl', "rb")
output = pickle.load(a_file)
#print(output)

def getmove(info):
    v = []
    moves=[]
    print(output[info])
    for key in output[info]:
        v.append(output[info][key])
        moves.append(key)
    print(np.random.choice(moves, 1, p = v)[0])

#getmove((3, '', 'x', 'x', 'x', '', '12', '1F', '11', ''))


print(output[(9, '11', '', 'x', '', '', '', 'x', '12', '1F')])

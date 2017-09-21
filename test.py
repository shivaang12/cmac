#Continuous cmac
#Author - Shivang Patel

import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt

#Taking step of 6.28/100 = 0.0628
x = 0.0209

#Generating 100 points
x_x=np.arange(0,300)
y = np.arange(0)

#print x
#Taking sine function here to train and test to data
for i in range (0,300):
    s = math.sin((x_x[np.array(i)]) * (x))
    #s = s * ((x_x[np.array(i)]) * (x))
    y = np.append(y,s)

xy = np.dstack((x_x * x,y))
#xy = xy.tolist()
#xy_rand = random.shuffle(xy)
test_num = np.arange(0)
inp_num = np.arange(0)

##Changing the Algorithm
##This is the old Algorithm
'''while (inp_num.size < 70):
    ran_num = random.randint(0,99)
    if ((ran_num in inp_num) == False ):
        inp_num = np.append(inp_num,ran_num)

while (test_num.size < 30):
    ran_num = random.randint(0,99)
    if((ran_num in inp_num) == False):
        test_num = np.append(test_num,ran_num)
'''
##This will be the new Algorithm
for i in range(300):
    if (i%3 == 1 and test_num.size < 90 ):
        test_num = np.append(test_num,i)
    else:
        inp_num = np.append(inp_num,i)

#Rearranging the order from assending to descending order
inp_num = np.sort(inp_num)
test_num = np.sort(test_num)

#print inp_num.size
#Defining Weights
#w_save = np.arange(0)

#Initializing some usefull variables
w_val = 0.0
w_num = 1
inc_val = 0
w_zero = np.arange(0.0)
w_save = w_zero
err_val = 0.0
#flip = 1
q_val = 0
err_arr = np.arange(0)
rms_arr = np.arange(0)
time_arr = np.arange(0)
time_gen_arr = np.arange(0)
rms = 1
for gen in range(3,35,2):
    start = time.time()
    w = np.random.rand(300)
    times = 1000
    pad_val = (gen-1)/2
    w_zero = np.array([0])
    for i in range(pad_val):
        w = np.append(w_zero,w)
        w = np.append(w,w_zero)
        w = np.append(w,w_zero)

    while(rms > 0.01):
        w_z = np.arange(0)
        for j in range(0,210):
            #q_val = j/2
            #print q_val
            for k in range(gen):
                if(k == 0):
                    w_val = (w_val + w[np.array(k + j)] * 0.25)
                if(k == gen-1):
                    w_val = (w_val + w[np.array(k + j)] * 0.75)
                w_val = w_val + w[np.array(k + j)]
            w_y_val = w_val/gen
            #w_trained = np.append(w_trained,w_val)
            y_val = (math.sin(inp_num[np.array(j)] * x))
            #y_val = y_val * ((x_x[np.array(i)]))
            if gen==3:
                w_z = np.append(w_z,w_y_val)
                #print j
            err_val = y_val - w_y_val
            #print err_val
            err_arr = np.append( err_arr, err_val)
            corrected_val = err_val/gen
            for k in range(gen):
                if(k == 0):
                    w[np.array(k + j)] = (w[np.array(k + j)]) + (corrected_val * 0.25)
                if(k == gen-1):
                    w[np.array(k + j)] = w[np.array(k + j)] + (corrected_val * 0.75)
                w[np.array(k + j)] = w[np.array(k + j)] + corrected_val
            w_val = 0.0
            #print w, q_val
            #err_val = 0.0
            #w_val = 0.0
        #print w.size
        rms = np.mean(err_arr**2)
        #times = times -1

    print gen
    if gen==3:
        w_save = w_z
        w_weight_save = w
        #print w_save
    end = time.time()
    time_arr = np.append(time_arr, (end-start))
    time_gen_arr = np.append(time_gen_arr, gen)
    rms_arr = np.append(rms_arr,rms)
    rms = 1
w_35 = w_save
w_new = w_weight_save[1::2]
#print w_save,w_new
new_gen = 3
test_val = 0.0
w_new_arr = np.arange(0)
for j in range(0,90):
    #q_val = j/2
    #test_num[np.array(q_val)]
    w_avg = w_new[np.array(j)] + (w_new[np.array(j - 1)] * 0.25) + (w_new[np.array(j + 1)] * 0.75)
    w_avg = w_avg / new_gen
    w_new_arr = np.append(w_new_arr,w_avg)
new_test_data = test_num * x
x_final = x_x * x
plt.plot(new_test_data,w_new_arr,'ro',x_final,y)
plt.xlabel('Radians')
plt.ylabel('Sin(x)')
plt.show()

#plt.plot(x_final,y,)
#plt.show()





plt.plot(time_gen_arr ,time_arr)
plt.xlabel('Generalization')
plt.ylabel('Time')
plt.show()
plt.plot(time_gen_arr,rms_arr)
plt.xlabel('Generalization')
plt.ylabel('Error')
plt.show()
#plt.plot()


#SOME UNUSEFULL CODE

        #break
    #break



    #for i in range(0,35):
    #    for j in range(2):
    #        for k in range(gen):
    #            w_val = w_val + w[np.array(k)]
    #        w_val = w_val / gen
    #        print w_val

import multiprocessing
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
from threading import Thread, Lock
import math
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
import random
import shutil
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import datetime
import threading

v_rates=list()
best_correctness=0.0
best_params_=0
version=0
Gtimestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
GAMax=80
GAParams=7
mutex = Lock()

class rates(object):
    def __init__(self, date, time, open_, high, low, close):
        self.date=date
        self.time=time
        self.o=open_
        self.h=high
        self.l=low
        self.c=close
        self.ohlc=(open_+high+low+close)/4.0
        self.hlc=(high+low+close)/3.0
        self.cStd=0.0
        self.dcStd=0.0
        self.dhcStd=0.0
        self.dclStd=0.0
        self.t1Std=0.0
        self.KalmanNextPredict=0.0


class shared_data(object):
    def __init__(self):
        self.rates=list()
        self.version=0
        self.best_fitness=0.0
        self.Gtime=0        
    
    def set_version(self, value):
        self.version = value

    def get_version(self):
        return self.version

    def set_rates(self, value):
        self.rates = value

    def get_rates(self):
        return self.rates   
    
    def set_best_fitness(self, value):
        self.best_fitness=value
    
    def get_best_fitness():
        return self.best_fitness 
    
    def set_Gtime(self, value):
        self.Gtime=value
    
    def get_Gtime():
        return self.Gtime     


def check_version():
    read_version=open('version.txt', 'r')
    ver = read_version.read().splitlines()
    global version
    version=int(ver[0])
    read_version.close()

    set_version = open('version.txt','w')    
    set_version.write(str(version+1))
    set_version.close()  

def resize(value2resize, OldMin, OldMax, NewMin, NewMax):
    NewValue = (((value2resize - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return NewValue 

def GetDayBeginIndex(data, i):
    idx=i
    if i<=0:
        return 0
    while data[idx].date==data[i].date and idx>=0:
        idx=idx-1
    
    return (idx+1)

def calc_std4Kalman(data, std_length):
    idx=len(data)-1
    d=max(len(data)-std_length,0)
    dm=len(data)
    s=np.array([])
    for i1 in range(d,dm):
        s=np.append(s,[data[i1].c])
    data[idx].cStd=s.std()  
    
    d1=max(len(data)-std_length,1)
    dm1=len(data)
    s1=np.array([])
    for i1 in range(d1,dm1):
        cc=abs(data[i1].c-data[i1-1].c)
        co=abs(data[i1].o-data[i1-1].c)
        if (data[i1].c>data[i1-1].c and data[i1].o>data[i1-1].c and data[i1].c>data[i1].o):
            cc=cc-co
        if (data[i1].c<data[i1-1].c and data[i1].o<data[i1-1].c and data[i1].c<data[i1].o):
            cc=cc-co                    
        s1=np.append(s1,[cc-co])
    if (dm1>d1):
        data[idx].dcStd=s1.std() 

    s2=np.array([])
    s3=np.array([])
    s4=np.array([])    
    for i1 in range(d1,dm1):
        hc=abs(data[i1].h-data[i1].c)
        cl=abs(data[i1].c-data[i1].l)
        s2=np.append(s2,[hc])
        s3=np.append(s3,[cl])
        v4=max(0,i1-4)
        dc2=abs(data[i1].c-data[v4].ohlc)
        s4=np.append(s4,[dc2])
    if (dm1>d1):
        data[idx].dhcStd=s2.std()
        data[idx].dclStd=s3.std() 
        data[idx].t1Std=s4.std()

def read_rates():
    #read_rates=open('SBER_170316_180828-5m.txt', 'r')  
    read_rates=open('SBER_150302_200425-5m.txt', 'r') 
    print("Start reading rates from file...")
    CloseStdLen=5
    v_rates=list()
    for line in read_rates:
        l1=line.split(',')
        if (l1[2].isdigit() and int(l1[3])<190000):   
            #rts=np.array([float(l1[5]), float(l1[6]), float(l1[7]), 0.0, (float(l1[5])+float(l1[6])+float(l1[7]))/3.0])
            rts=rates(int(l1[2]),int(l1[3]), float(l1[4]), float(l1[5]), float(l1[6]), float(l1[7]))
            v_rates.append(rts)
            calc_std4Kalman(v_rates, CloseStdLen)          
                    
    read_rates.close()  
    print("Reading rates from file is completed!")
    return v_rates
    

# specifying the size of the state vector with dim_x 
# and the size of the measurement vector that you will be using with dim_z

def generate_vector(data, step, var):
    v=np.array([])
    v1=max(0,step-1)
    v2=max(0,step-2)
    v3=max(0,step-3)
    v4=max(0,step-4)
    if (var==0):
        v=[data[step].c, data[step].c-data[v1].c, data[step].c-data[v2].c, data[v1].c-data[v2].hlc, data[step].c-data[v3].hlc]
    if (var==1):
        v=[data[step].c, data[step].c-data[v1].c, data[step].c-data[v2].c, data[step].c-data[v3].c, data[step].c-data[v2].hlc]      
    if (var==2):
        v=[data[step].c, data[step].c-data[v1].c, data[step].c-data[v2].c, data[step].c-data[v2].hlc, data[step].c-data[v3].ohlc]   
    if (var==3):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v2].hlc]   
    if (var==4):
        if (v1>0 and data[step].date!=data[v1].date):
            v1=GetDayBeginIndex(data, step)
        if (v2>0 and data[step].date!=data[v2].date):
            v2=GetDayBeginIndex(data, step)   
        if (v3>0 and data[step].date!=data[v3].date):
            v3=GetDayBeginIndex(data, step)    
        if (v4>0 and data[step].date!=data[v4].date):
            v4=GetDayBeginIndex(data, step)     
        v=[data[step].c, data[step].c-data[v1].c, data[step].c-data[v2].c, data[step].c-data[v2].hlc, data[step].c-data[v3].ohlc] 
    if (var==5):
        if (v1>0 and data[step].date!=data[v1].date):
            v1=GetDayBeginIndex(data, step)
        if (v2>0 and data[step].date!=data[v2].date):
            v2=GetDayBeginIndex(data, step)   
        if (v3>0 and data[step].date!=data[v3].date):
            v3=GetDayBeginIndex(data, step)    
        if (v4>0 and data[step].date!=data[v4].date):
            v4=GetDayBeginIndex(data, step)     
        v=[data[step].c, data[step].c-data[v1].c, data[step].c-data[v2].c, data[step].c-data[v2].c, data[step].c-data[v3].c]         
    if (var==6):
        v=[data[step].c, data[step].c-data[v1].c, data[step].c-data[v2].c, data[step].c-data[v3].ohlc, data[step].c-data[v2].ohlc]   
    if (var==7):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v2].h]       
    if (var==8):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v2].l]    
    if (var==9):
        v=[data[step].c, data[step].c-data[v2].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v2].h]       
    if (var==10):
        v=[data[step].c, data[step].c-data[v2].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v2].l]          
                
       
    return v

def generate_vector_O(data, step):
    v=np.array([])
    v1=max(0,step-1)
    v2=max(0,step-2)
    v3=max(0,step-3)
    v4=max(0,step-4)

    if (step==0):
        v=[data[step].o, 0.0, 0.0, 0.0, 0.0]   
    if (step>0):
        v=[data[step].o, data[step].o-data[v1].c, data[step].o-data[v1].h, data[step].o-data[v1].l, data[step].o-data[v2].c]
                               
    return v

def generate_vector_C(data, step, var):
    v=np.array([])
    v1=max(0,step-1)
    v2=max(0,step-2)
    v3=max(0,step-3)
    v4=max(0,step-4)

    v=[]
    if (var==0):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v2].l]
    if (var==1):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v2].l]        
    if (var==2):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v2].hlc]        
    if (var==3):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v3].hlc]                
    if (var==4):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v4].hlc]
    if (var==5):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v3].ohlc] 
    if (var==6):
        v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v4].ohlc]               
                               
    return v

def generate_vector2_C(data, step):
    v=np.array([])
    v1=max(0,step-1)
    v2=max(0,step-2)
    v3=max(0,step-3)
    v4=max(0,step-4)

    v=[data[step].c, data[step].c-data[v1].c, data[step].h-data[step].c, data[step].c-data[step].l, data[step].c-data[v4].ohlc]               
    v2=[data[step].cStd, data[step].dcStd, data[step].dhcStd, data[step].dclStd, data[step].t1Std] 
                               
    return v,v2



def ukf_init(x1, x2, x3, x4):
    def fx(x, dt):
         # state transition function - predict next state based
         # on constant velocity model x = vt + x_0
        F = np.array([[1.0, x1, x2, x3, x4],
                     [0., 1.0, 0., 0., 0.],
                     [0., 0., 1.0, 0., 0.],
                     [0., 0., 0., 1.0, 0.],
                     [0., 0., 0., 0., 1.0],
                     ], dtype=float)
        return np.dot(F, x)
    
    def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]
        return np.array([x[0], x[1], x[2], x[3], x[4]])    
    dt = 30
    # create sigma points to use in the filter. This is standard for Gaussian processes
    points = MerweScaledSigmaPoints(5, alpha=0.10550, beta=1.6, kappa=0.)

    kf = UnscentedKalmanFilter(dim_x=5, dim_z=5, dt=dt, fx=fx, hx=hx, points=points)
    kf.P *= 100. # initial uncertainty
    z_std = 0.25
    kf.R = np.diag([z_std**1, z_std**1, z_std**1, z_std**1, z_std**1]) # 1 standard
    kf.Q =np.eye(5)*.001   
    return kf
    

def check_kalman(individual_):
    assert GAParams==len(individual_)  

    x1=resize(individual_[0], 0, GAMax, -0.15, 0.15)
    x2=resize(individual_[1], 0, GAMax, -0.15, 0.15)
    x3=resize(individual_[2], 0, GAMax, -0.15, 0.15)
    x4=resize(individual_[3], 0, GAMax, -0.15, 0.15)

    x5=resize(individual_[4], 0, GAMax, 0.15, 0.35)
    x6=resize(individual_[5], 0, GAMax, 0.15, 0.35)
    feature_type=int(resize(individual_[6], 0, GAMax, 0., 6.4))

    feature_vector=list()
    feature_vector.append(x1) 
    feature_vector.append(x2) 
    feature_vector.append(x3) 
    feature_vector.append(x4) 
    feature_vector.append(x5) 
    feature_vector.append(x6)
    feature_vector.append(feature_type)
    
    kf=ukf_init(x1,x2,x3,x4)
    predicted_next_close=0.0
    true_close=0.0
    previous_close=0.0    
    true_predictions=0
    false_predictions=0
    
    z=generate_vector_O(v_rates, 0)
    kf.x=z
    kf.predict() 
    z,z2=generate_vector2_C(v_rates, 0)
    kf.update(z)
    kf.predict()
    v_rates[0].KalmanNextPredict=kf.x[0]
    
    for i in range(1,len(v_rates)-1):       
        if (v_rates[i].date!=v_rates[i-1].date):
            z=generate_vector_O(v_rates, i)
            kf.update(z)
            kf.predict()
        z,z2=generate_vector2_C(v_rates, i)  
        kf.Q[0,0]=z2[0]*x5
        kf.Q[1,1]=z2[1]*x6           
        kf.Q[2,2]=z2[2]*x6
        kf.Q[3,3]=z2[3]*x6
        kf.Q[4,4]=z2[4]*x6
        kf.update(z)        
        kf.predict()
        v_rates[i].KalmanNextPredict=kf.x[0]
        if(np.sign((v_rates[i+1].c-v_rates[i].c)*(v_rates[i].KalmanNextPredict-v_rates[i].c))>0):
            true_predictions=1+true_predictions
            #print("+1 Tp")
        else:
            false_predictions=1+false_predictions           
            #print("+1 Fp")
    
    #print("F: ", kf.F)
    #print("R: ", f2.R)
    #print("Q: ", f2.Q)
    fitness=1.0*true_predictions/(true_predictions+false_predictions)
    
    mutex.acquire()    
    print("TP: %s, FP: %s, Fitness = %2.6f" %(true_predictions, false_predictions, fitness))

    global best_correctness
    if (fitness>best_correctness):
        best_correctness=fitness
        
        file_1 = open('results\\ukf2_opt_'+str(version)+'_'+Gtimestr+'.txt','a')  
        file_1.write("DS: %2.6f; " % (best_correctness))
        file_1.write("Features: ")
        for f in range(len(feature_vector)):
            file_1.write("%5.5f, " % feature_vector[f])        
        file_1.write("\n")
        file_1.close()    
            
    print("DS best: %2.6f" % best_correctness) 
    mutex.release()
    
    return fitness,

# END check_kalman

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)    
toolbox = base.Toolbox()    

toolbox.register("attr_init", random.randint, 0, GAMax)
# Structure initializers        
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_init, GAParams)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)    
toolbox.register("evaluate", check_kalman)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def init_pool(version_, v_rates_, Gtimestr_):
    # This will run in each child process.
    global version
    global v_rates
    global best_correctness
    global Gtimestr
    version = version_
    v_rates = v_rates_
    best_correctness=0.0
    Gtimestr=Gtimestr_

if __name__ == "__main__":  
    
    random.seed(7) 
    check_version()   
    shutil.copyfile('ukf2.py', '.\\arch\\ukf2_'+str(version)+'_'+Gtimestr+'.py')      
    v_rates=list()    
    v_rates=read_rates()

    pool = multiprocessing.Pool(processes=7, initializer=init_pool, initargs=(version, v_rates, Gtimestr,))  
    toolbox.register("map", pool.map) 
    
    pop = toolbox.population(n=140)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
           
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.15, ngen=20, stats=stats, halloffame=hof, verbose=True)
    print(hof)            
    print("Best_correctness: %f" % best_correctness)          
    print("Best_params: ", best_params_)     
    pool.close()



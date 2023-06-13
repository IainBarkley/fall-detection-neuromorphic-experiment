# lmu_classifier.py
# Terrence C. Stewart's LMU implementation in Nengo

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os
import nengo

sys.path.insert(0,'../networks')
from ldn import LDN

sys.path.insert(0,'../')
from utilities import generate_train_test_split

data_dir = '../fall_detection_data/merged_processed/'
subject_file = 'SA06-merged.csv'

# LDN parameters
size_in = 9         # 9 features of the fall detection data
theta = 0.5
q = 20

# generate "training" data for the output layer of the network
# dt = 0.001
# ts = np.arange(10000)*dt
# stim1 = np.sin(ts*2*np.pi).reshape(-1,1)
# stim2 = np.sin(ts*2*np.pi*2).reshape(-1,1)
# x1 = LDN(theta=theta, q=q).apply(stim1)
# x2 = LDN(theta=theta, q=q).apply(stim2)
# eval_points = np.vstack([x1, x2])
# targets = np.hstack([np.ones(len(x1)), -np.ones(len(x2))]).reshape(-1,1)

# load data and generate the train/test split
# ddf = pd.read_csv(os.path.join(data_dir,subject_file),index_col=0)

train_test_split = 0.5

domain = list(range(500))

result = []
while len(domain) > train_test_split:
    print(domain)
    n = np.random.choice(domain)
    result.append(n)
    domain = [x for x in domain if x <= n - 10 or x >= x + 10]
    
np.random.shuffle(result)
print(result)

# out = generate_train_test_split(ddf)

# samples = [df.iloc[x:x+block_size] for x in np.random.randint(len(df), size=num_samples)]
# train_xs = 
# train_ys = 

# test_xs = 
# test_ys = 

# model = nengo.Network()
# with model:
    # # stim = nengo.Node(lambda t: np.sin(2*np.pi*t) if t<4 else np.sin(2*np.pi*t*2))
    
    # ldn = nengo.Node(LDN(theta=theta, q=q))
    # nengo.Connection(stim, ldn, synapse=None)
    
    # neurons = nengo.Ensemble(n_neurons=200, dimensions=q, neuron_type=nengo.LIF())
    # nengo.Connection(ldn, neurons)
    
    # # initialize the network with the training data
    # category = nengo.Node(None, size_in=1)
    # nengo.Connection(neurons, category, eval_points=eval_points, function=targets)
    
    # p_stim = nengo.Probe(stim)
    # p_ldn = nengo.Probe(ldn)
    # p_category = nengo.Probe(category, synapse=0.01)

# sim = nengo.Simulator(model)
# with sim:
    # sim.run(8)
    
# fig,ax = plt.subplots(1,1)
# ax.plot(sim.trange(), sim.data[p_stim], label='stimulus')
# ax.plot(sim.trange(), sim.data[p_category], label='output')
# ax.legend()
# plt.show()    
# lmu_classifier.py
# Terrence C. Stewart's LMU implementation in Nengo

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os
import nengo

sys.path.insert(0,'../networks')
from ldn import LDN

sys.path.insert(0,'../')
from utilities import generate_train_test_split

training_data_dir = '../fall_detection_data/not_merged_processed/'
subject_to_train_on = 'SA06'
training_strategy = 'within-subject'
included_features = 'All'

# LDN parameters
size_in = 9         # 9 features of the fall detection data
theta = 1.0
q = 42

# simulation parameters
dt = 0.01

# training dataset parameters
train_test_split = 0.5
scaling_factor = 1.
test_on_train = True

# prediction parameters
decision_threshold = 0.5

### prepare training and test data ###
# load data and generate the train/test split

all_trial_files = list(set( [f for f in os.listdir(training_data_dir) if ".csv" in f] ))
if training_strategy == 'within-subject':
    all_trials = list(set( [f for f in all_trial_files if subject_to_train_on in f] ))
elif training_strategy == 'across-subject':
    all_trials = all_trial_files

train_trials = np.random.choice(all_trials,size = int( len(all_trials)*train_test_split), replace=False ).flatten().tolist()
test_trials = [ e for e in all_trials if e not in train_trials ]

if included_features == 'All':
    n_features = 9
else:
    n_features = len(included_features)
    
### prepare training data ###
# these 'empty' matrix have an undesired row of zeros that will need to be removed later
train_xs = np.zeros( n_features ).reshape(1,-1)
lmu_train_xs = np.zeros( n_features * q ).reshape(1,-1)

train_ys = np.zeros( 1 ).reshape(1,-1)
for train_trial in train_trials:

    # load the data from the trial
    temp_df = pd.read_csv(os.path.join(training_data_dir,train_trial),index_col=0)

    # select the desired input features
    if included_features == 'All':
        temp_train_xs = temp_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].values
    else:
        temp_train_xs = temp_df[included_features].values
        
    # TO DO apply scaler HERE on temp_train_xs)
    
    #scaler = StandardScaler().fit(temp_train_xs)
    #temp_train_xs = scaler.transform(temp_train_xs)
    temp_lmu_train_xs = LDN(theta = theta, q = q, size_in = n_features).apply(temp_train_xs,dt=dt)
    
    # append it to the full training data set
    train_xs = np.vstack([train_xs,temp_train_xs[1:,:]])
    lmu_train_xs = np.vstack([lmu_train_xs,temp_lmu_train_xs[1:,:]])
    
    temp_train_ys = temp_df['Fall/No Fall'].values.reshape(-1,1)
    train_ys = np.vstack([train_ys,temp_train_ys[1:,:]])
    
# remove the first row of zeros created at initialization
train_xs = train_xs[1:,:]        
lmu_train_xs = lmu_train_xs[1:,:]
train_ys = train_ys[1:,:]

if test_on_train == False:

    ### prepare testing data ###
    # this 'empty' matrix creates an undesired row of zeros that will need to be removed later
    test_xs = np.zeros( n_features ).reshape(1,-1)
    test_ys = np.zeros( 1 ).reshape(1,-1)
    for test_trial in test_trials:
    
        # load the data from the trial
        temp_df = pd.read_csv(os.path.join(training_data_dir,test_trial),index_col=0)
        
        # select the desired input features
        if included_features == 'All':
            temp_test_xs = temp_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].values
        else:
            temp_test_xs = temp_df[included_features].values
        
        # scaler = StandardScaler().fit(temp_test_xs)
        # temp_test_xs = scaler.transform(temp_test_xs)
        
        # append it to the full training data set
        test_xs = np.vstack([test_xs,temp_test_xs])
        temp_test_ys = temp_df['Fall/No Fall'].values.reshape(-1,1)
        test_ys = np.vstack([test_ys,temp_test_ys])
    
    test_xs = test_xs[1:,:]
    test_ys = test_ys[1:,:]

else:
    test_xs = train_xs
    test_ys = train_ys

#######################################


model = nengo.Network()
with model:

    def stim_func(t):
        index = int(t/dt)-1
        return test_xs[index,:]

    stim = nengo.Node( size_out = size_in, output = stim_func )

    ldn = nengo.Node( LDN( theta = theta, q = q, size_in = size_in))
    nengo.Connection(stim, ldn, synapse=None)

    neurons = nengo.Ensemble(n_neurons=1000, dimensions=q*size_in, neuron_type=nengo.LIFRate())
    nengo.Connection(ldn, neurons, synapse = None)

    # initialize the network with the training data
    category = nengo.Ensemble(n_neurons=1000,dimensions=1,radius=scaling_factor, neuron_type=nengo.Direct())
    nengo.Connection(neurons, category, eval_points=lmu_train_xs, function=train_ys, synapse = None)

    p_stim = nengo.Probe(stim,synapse=None)
    p_ldn = nengo.Probe(ldn,synapse=None)
    p_category = nengo.Probe(category, synapse=None)

sim = nengo.Simulator(model,dt=dt)
with sim:
    sim.run(test_xs.shape[0]*dt)

predictions = np.where(sim.data[p_category].flatten() > np.max(sim.data[p_category].flatten())*decision_threshold, 1, 0)

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)

ax1.plot(sim.trange(), test_ys.flatten(), label='True')
ax2.plot(sim.trange(), sim.data[p_category]/scaling_factor, label='Output')
ax2.plot(sim.trange(), predictions, label='Predicted Label')
ax1.legend()
ax2.legend()
plt.show()

print(predictions.dtype)
print(train_ys.flatten().dtype)

tn, fp, fn, tp = confusion_matrix( train_ys, predictions.flatten() ).ravel()

for performance_metric, number in zip(('True Negatives','False Positives','False Negatives','True Positives'),(tn, fp, fn, tp)):
    print('{}: {}'.format(performance_metric,number))
import nengo
import nengo_spa as spa

D = 512

model = spa.Network()
with model:
    
    color = spa.State(D, label='color')
    shape = spa.State(D, label='shape')
    
    memory = spa.State(D, label='memory', feedback=1)
    
    color * shape >> memory
    
    query = spa.State(D, label='query')
    
    answer = spa.State(D, label='answer')
    
    memory*~query >> answer
    
    print(model.n_neurons)
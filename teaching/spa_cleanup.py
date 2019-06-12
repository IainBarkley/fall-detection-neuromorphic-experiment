import nengo
import nengo_spa as spa

D = 64

model = spa.Network()
with model:
    
    color = spa.State(D, label='color')
    shape = spa.State(D, label='shape')
    
    memory = spa.State(D, label='memory', feedback=1)
    
    color * shape >> memory
    
    query = spa.State(D, label='query')
    
    answer = spa.WTAAssocMem(input_vocab=D, threshold=0.7, 
                             mapping=['RED', 'BLUE', 'CIRCLE', 'TRIANGLE'], 
                             label='answer')
    
    memory*~query >> answer
    


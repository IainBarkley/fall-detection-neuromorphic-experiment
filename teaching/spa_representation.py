import nengo
import nengo_spa as spa

D = 64

model = spa.Network()
with model:
    
    def stim_func(t):
        if t<0.1:
            return spa.sym.DOG+0.2*spa.sym.BLUE
        elif 0.2<t<0.3:
            return spa.sym.CAT
        else:
            return 0
    
    stim = spa.Transcode(stim_func, output_vocab=D)
    
    vision = spa.State(D, label='vision', neurons_per_dimension=50,
                       subdimensions=64,
                       represent_cc_identity=False)
    
    memory = spa.State(D, feedback=1, feedback_synapse=0.1, label='memory',
                       subdimensions=64, represent_cc_identity=False)
    
    vision*0.1 >> memory

    stim >> vision
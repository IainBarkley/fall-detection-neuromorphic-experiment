import nengo
import nengo_spa as spa

D = 64

model = spa.Network()
with model:
    
    stim = spa.Transcode(lambda t: spa.sym.A if t<0.1 else 0, output_vocab=D)

    memory = spa.State(D, feedback=1)
    stim >> memory
    
    with spa.ActionSelection():
        spa.ifmax(spa.dot(memory, spa.sym.A), spa.sym.B >> memory)
        spa.ifmax(spa.dot(memory, spa.sym.B), spa.sym.C >> memory)
        spa.ifmax(spa.dot(memory, spa.sym.C), spa.sym.D >> memory)
        spa.ifmax(spa.dot(memory, spa.sym.D), spa.sym.E >> memory)
        spa.ifmax(spa.dot(memory, spa.sym.E), spa.sym.A >> memory)
    
        
            

import nengo
import nengo_spa as spa

D = 64

model = spa.Network()
with model:
    vision = spa.State(D, label='vision')
    
    speech = spa.State(D, label='speech')
    
    with spa.ActionSelection():
        spa.ifmax(spa.dot(vision, spa.sym.DOG), spa.sym.BARK >> speech)
        spa.ifmax(spa.dot(vision, spa.sym.CAT), spa.sym.MEOW >> speech)
        spa.ifmax(spa.dot(vision, spa.sym.RAT), spa.sym.SQUEAK >> speech)
        spa.ifmax(0.5, spa.sym.NOTHING >> speech)
        
        
    
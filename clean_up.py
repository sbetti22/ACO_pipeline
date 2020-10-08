# ACO pipeline part 4
# ACO pipeline clean up

# written by Sarah Betti 2020

import numpy as np
import os
import glob

def clean_up(fil, targs):
    '''
    delete all b_, db_, and fdb_ frames 
    '''
    if isinstance(targs, str):
        targs=[targs]
    D = glob.glob(fil+'/darks/*/*.fit')
    Dfold = glob.glob(fil+'/darks/*/')
    
    F = glob.glob(fil+'/flats/*/*.fit')
    Fb = glob.glob(fil+'/flats/b_*.fit')
    Ffold = glob.glob(fil+'/flats/*/')
    
    for i in [D, F, Fb]:
        for j in i:
            os.remove(j)
    
    for i in [Dfold, Ffold]:
        for j in i:
            os.rmdir(j)
    
    for name in targs:
        T = glob.glob(fil+'/' + name + '/*/*.fit')
        Tfold = glob.glob(fil+'/' + name + '/*/')
        
        for i in T:
            os.remove(i)
        for i in Tfold:
            os.rmdir(i)
    
    
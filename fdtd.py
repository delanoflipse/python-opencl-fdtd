import numpy as np
import time
from lib.simulation import simulation_step

count = 1000
start = time.time()

for i in range(count):
    simulation_step()
    
end = time.time()
print(end-start)
print((end-start)/ count)
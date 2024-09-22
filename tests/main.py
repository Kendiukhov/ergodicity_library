from ergodicity.process.basic import BrownianMotion
from ergodicity.configurations import *
from ergodicity.process.default_values import *

bm = BrownianMotion()

simulation = bm.simulate(t=1, num_instances=3, timestep=0.1, save=True, plot=True)



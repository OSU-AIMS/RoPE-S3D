import klampt
from klampt import WorldModel
from klampt import vis
import math
import time

import numpy as np
from robotpose.simulation.kinematics import ForwardKinematics


fwd = ForwardKinematics()

world = WorldModel()

world.loadElement(r"urdf\motoman_mh5_support\urdf\mh5l.urdf")
robot = world.robot(0)

robot.setConfig([0]*15)

for n in range(1,15):
    link = robot.link(n)

    print(f"{n}:{link.getWorldPosition([0,0,0])} {link.getWorldDirection([0,0,0])} {link.getName()}")


print(fwd.calc([np.pi,0,0,0,0,0]))
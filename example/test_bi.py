import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem





def solve_walking_problem(robot, problem, init_state):
    # create a crocoddyl solver
    display = crocoddyl.MeshcatDisplay(robot, 4, 4, False)
    ddp = crocoddyl.SolverFDDP(problem)
    ddp.setCallbacks(
        [
            crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackDisplay(display),
        ]
    )
    
    # solve problem
    ddp.th_stop = 1e-9
    init_xs = [init_state] * (problem.T + 1)
    init_us = []
    maxiter = 1000
    regInit = 0.1
    ddp.solve(init_xs, init_us, maxiter, False, regInit)

    # display the solution
    display.rate = -1
    display.freq = 1
    display.displayFromSolver(ddp)

    return ddp



# Creating the lower-body part of Talos
talos_legs = example_robot_data.load("talos")
for joint in talos_legs.model.joints:
        print(joint.shortname())

# Setting up the 3d walking problem
rightFoot = "right_sole_link"
leftFoot = "left_sole_link"
gait = SimpleBipedGaitProblem(talos_legs.model, rightFoot, leftFoot)


# Create the initial state
q0 = talos_legs.q0.copy()
v0 = pinocchio.utils.zero(talos_legs.model.nv)
x0 = np.concatenate([q0, v0])


# Creating the walking problem
stepLength = 0.6  # meters
stepHeight = 0.1  # meters
timeStep = 0.0375  # seconds
stepKnots = 20
supportKnots = 20
problem = gait.createWalkingProblem(
    x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots
)


sol = solve_walking_problem(talos_legs, problem, x0)
import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio as pin

import crocoddyl
from crocoddyl.utils.biped import plotSolution
from pinocchio.robot_wrapper import RobotWrapper
from assets import ASSET_DIR
import time
import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem
from lower_body_walk import LowerBodyWalk
from utils import load_robot, visualize_robot, get_half_sitting_pose, create_init_state



'''def create_init_state(robot,q0=None):
    # Create the initial state
    if q0 is None:
        q0 = get_half_sitting_pose(robot)
    v0 = pin.utils.zero(robot.model.nv)
    x0 = np.concatenate([q0, v0])
    return x0'''


def create_walking_problem(robot, right_foot, left_foot, init_state):
    q0 = get_half_sitting_pose(robot)
    robot.model.referenceConfigurations['half_sitting'] = q0
    gait = LowerBodyWalk(robot.model, right_foot, left_foot)
    # create initial state
    x0 = init_state

    # Creating the walking problem
    stepLength = 0.6  # meters
    stepHeight = 0.1  # meters
    timeStep = 0.0375  # seconds
    stepKnots = 20
    supportKnots = 10
    problem = gait.createWalkingProblem(
        x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots
    )
    return problem


def solve_walking_problem(problem, init_state):
    # create a crocoddyl solver
    ddp = crocoddyl.SolverFDDP(problem)
    ddp.setCallbacks(
        [
            crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose(),
        ]
    )
    
    # solve problem
    ddp.th_stop = 1e-9
    init_xs = [init_state] * (problem.T + 1)
    init_us = []
    maxiter = 1000
    regInit = 0.1
    ddp.solve(init_xs, init_us, maxiter, False, regInit)

    return ddp


def display_solution(robot, ddp):
    display = crocoddyl.MeshcatDisplay(robot, 4, 4, False)
    # display the solution
    display.rate = -1
    display.freq = 1
    display.displayFromSolver(ddp)

if __name__ == "__main__":
    # load robot
    robot = load_robot()
    right_foot = 'right_ankle_roll_link'
    left_foot = 'left_ankle_roll_link'


    # create initial state
    init_state = create_init_state(robot)

    # create a simple biped gait problem
    problem = create_walking_problem(robot, right_foot, left_foot, init_state)

    # solve the problem
    ddp = solve_walking_problem(problem, init_state)
    

    # display the solution
    display_solution(robot, ddp)

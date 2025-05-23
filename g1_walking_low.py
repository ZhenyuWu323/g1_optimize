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




def create_walking_problem(robot, right_foot, left_foot, init_state):
    gait = LowerBodyWalk(robot.model, right_foot, left_foot)
    # create initial state
    x0 = init_state

    # Creating the walking problem
    stepLength = 0.4  # meters
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
    while True:
        display.displayFromSolver(ddp)
        time.sleep(1.0)

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

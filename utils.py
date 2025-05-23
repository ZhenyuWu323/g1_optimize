import os
import re
import time
import crocoddyl
import pinocchio as pin
import numpy as np

from assets import ASSET_DIR
from pinocchio.robot_wrapper import RobotWrapper


def visualize_robot(robot):
    # visualize
    viz = pin.visualize.MeshcatVisualizer(robot.model,robot.collision_model,robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel(collision_color=[0.5, 0.5, 0.5, 1.0],visual_color=[0.5, 0.5, 0.5, 1.0])
    q0 = pin.neutral(robot.model)
    time.sleep(0.5)
    viz.display(robot.q0)
    viz.displayVisuals(True)



def get_half_sitting_pose(robot):
    q0 = pin.neutral(robot.model)
    
    # Set the root joint (floating base) position
    q0[2] = 0.793  # Set the base height to match URDF
    
    joint_pos = {
        "left_hip_pitch_joint": -0.10,
        "left_knee_joint": 0.30,
        "left_ankle_pitch_joint": -0.2,
        "right_hip_pitch_joint": -0.10,
        "right_knee_joint": 0.30,
        "right_ankle_pitch_joint": -0.2,
    }
    for jid, jname in enumerate(robot.model.names):
        for name, value in joint_pos.items():
            if name in jname:
                jmodel = robot.model.joints[jid]
                if jmodel.nq > 0:
                    q0[jmodel.idx_q] = value
                    break
    return q0


def load_robot(urdf_name="g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"):
    model_path = os.path.join(ASSET_DIR, urdf_name)
    mesh_dir = ASSET_DIR
    robot = RobotWrapper.BuildFromURDF(model_path, mesh_dir, pin.JointModelFreeFlyer())
    
    # set half sitting pose
    q0 = get_half_sitting_pose(robot)
    robot.q0 = q0
    robot.model.referenceConfigurations['half_sitting'] = q0

    # set velocities to zero
    v0 = np.zeros(robot.model.nv)
    robot.model.defaultState = np.concatenate([q0, v0])
    return robot


def create_init_state(robot):
    # Create the initial state
    q0 = robot.q0.copy()
    v0 = pin.utils.zero(robot.model.nv)
    x0 = np.concatenate([q0, v0])
    return x0


def display_solution(robot, ddp):
    display = crocoddyl.MeshcatDisplay(robot, 4, 4, False)
    # display the solution
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(ddp)
        time.sleep(1.0)


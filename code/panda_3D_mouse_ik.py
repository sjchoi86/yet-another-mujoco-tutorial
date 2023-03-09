# import all the dependencies
import os, glfw
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
from mujoco_utils import *

import pyspacemouse
import time

pyspacemouse.close()
success = pyspacemouse.open()
import copy

from flexIk import *
from flexIk.velIkSolver import SolverTypes as SolverTypes

# set numpy print options
np.set_printoptions(precision=2, suppress=True, linewidth=100)

# print MuJoCo version
print("MuJoCo version:[%s]" % (mujoco.__version__))

# parse panda URDF model
xml_path = "../model/panda/franka_panda.xml"
env = MuJoCoParserClass(name="Panda", rel_xml_path=xml_path, VERBOSE=True)
model = mujoco.MjModel.from_xml_path(xml_path)
print("[Panda] parsed.")

# initialize viewer
env.init_viewer(
    viewer_title="IK using Panda",
    viewer_width=1200,
    viewer_height=800,
    viewer_hide_menus=True,
)
env.update_viewer(azimuth=180, distance=3.84, elevation=-8.32, lookat=[0.02, 0.05, 0.7])
env.update_viewer(VIS_TRANSPARENT=True)
env.reset()  # reset

# Set IK targets
ik_body_name = "panda_eef"

# set joint angles
qInit = np.array((0, 25, 0, -120, 0, 145, 0)) * np.pi / 180.0
qCurrent = copy.deepcopy(qInit)

# set initial configuration for arm and gripper
env.forward(q=qCurrent, joint_idxs=env.rev_joint_idxs)  # Set joint angles
env.forward(
    q=[0.04, -0.04], joint_idxs=env.pri_joint_idxs
)  # Open gripper (prismatic joints)
gripperState = False  # gripper state (True: closed, False: open)

# initialize the input for the flex Ik solver
numDof = 7
C = np.eye(numDof)
I = np.eye(numDof)
qMin = env.rev_joint_min
qMax = env.rev_joint_max
qdMin = np.array([-2] * numDof)
qdMax = np.array([2] * numDof)
qddMax = np.array([1] * numDof)
ts = 5e-1
s = 1
sPrev = s
tol = 1e-4
errNormMax = 2e-1
kNs = 1e-3

# initialize 3D mouse input
state = pyspacemouse.read()
mouse_pos_init = copy.deepcopy(env.data.body("panda_eef").xpos)
mouse_pos = copy.deepcopy(mouse_pos_init)
mouse_ori_init = copy.deepcopy(env.get_R_body(body_name="panda_eef"))
mouse_ori = copy.deepcopy(mouse_ori_init)
scale_pos_init = 2e-4
scale_ori_init = 1e-2

# display settings
display_target = True

# IK loop
imgs, img_ticks, max_tick = [], [], 10000
while (env.tick < max_tick) and env.is_viewer_alive():
    # Update the spacemouse input
    state = pyspacemouse.read()
    scale_pos = scale_pos_init
    scale_ori = scale_ori_init

    # update mouse pose only if the scale factor is increasing above the threshold
    mouse_pos += scale_pos * np.rad2deg(np.array([-state.y, state.x, state.z]))
    mouse_ori_temp = rpy2r(
        scale_ori * np.array((-state.roll, -state.pitch, -state.yaw))
    )
    mouse_ori = np.matmul(mouse_ori_temp, mouse_ori)

    # Reset if button 1 is pressed
    if state.buttons[0] and state.buttons[1]:  # reset to initial configuration
        qCurrent = copy.deepcopy(qInit)
        mouse_pos = copy.deepcopy(mouse_pos_init)
        mouse_ori = copy.deepcopy(mouse_ori_init)
        time.sleep(0.2)

    elif state.buttons[0] and not state.buttons[1]:
        mouse_pos = copy.deepcopy(env.get_p_body(ik_body_name))
        mouse_ori = copy.deepcopy(env.get_R_body(ik_body_name))
        time.sleep(0.2)

    # Flip gripper state
    elif state.buttons[1] and not state.buttons[0]:
        if not gripperState:
            env.forward(q=[-0.04, 0.04], joint_idxs=env.pri_joint_idxs)
            gripperState = True
        else:
            env.forward(q=[0.04, -0.04], joint_idxs=env.pri_joint_idxs)
            gripperState = False
        time.sleep(0.2)

    # update box constraints
    dqLow = np.maximum(
        (qMin - qCurrent) / ts,
        -qdMax,
        -np.sqrt(np.maximum(2 * qddMax * (qCurrent - qMin), 0)),
    )
    dqUpp = np.minimum(
        (qMax - qCurrent) / ts,
        qdMax,
        np.sqrt(np.maximum(2 * qddMax * (qMax - qCurrent), 0)),
    )

    # Numerical IK
    J, err = env.get_ik_ingredients(
        body_name=ik_body_name, p_trgt=mouse_pos, R_trgt=mouse_ori, IK_P=True, IK_R=True
    )

    # clamp error
    errNorm = np.linalg.norm(err)
    if errNorm > errNormMax:
        errClamped = (errNormMax / errNorm) * err
    else:
        errClamped = err

    # nullspace bias
    qdNs = kNs * (qInit - qCurrent)
    qdNs = qdNs.reshape(-1, 1)

    dxGoalData = [errClamped, qdNs]
    JData = [J[:, env.rev_joint_idxs], I]

    dq, sData, exitCode = velIk(
        C, dqLow, dqUpp, dxGoalData, JData, SolverTypes.ESNS_MT, invTypes.SRINV
    )
    s = sData[0]

    # Update q and FK
    qCurrent = qCurrent + dq * ts

    # normalized joint position [0, 1]
    qNormal = np.maximum((qMax - qCurrent), (qCurrent - qMin)) / (qMax - qMin)

    if env.tick % 50 == 0 and s < 0.4:
        if np.any(qNormal >= 0.99):
            print(qNormal)
        print(s)

    # update model
    env.forward(q=qCurrent, joint_idxs=env.rev_joint_idxs)

    # Render
    env.plot_T(
        p=env.get_p_body(body_name=ik_body_name),
        R=env.get_R_body(body_name=ik_body_name),
        PLOT_AXIS=True,
        axis_len=0.1,
        axis_width=0.003,
        PLOT_SPHERE=False,
        sphere_r=0.05,
        sphere_rgba=[1, 0, 0, 0.9],
    )
    if display_target:
        env.plot_T(
            p=mouse_pos,
            R=mouse_ori,
            PLOT_AXIS=True,
            axis_len=0.1,
            axis_width=0.003,
            PLOT_SPHERE=False,
            sphere_r=0.05,
            sphere_rgba=[0, 0, 1, 0.9],
        )
    env.render()


# Close viewer
env.close_viewer()
print("Done.")

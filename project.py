import numpy as np

from astrobee import Astrobee
from mpc import MPC
from simulation import EmbeddedSimEnvironment

# Set the path to the trajectory file:
trajectory_quat = './Dataset/trajectory_quat.txt'

# Path of your tuning.yaml
tuning_file_path = 'tuning.yaml'

# Set astrobee dynamics
abee = Astrobee(trajectory_file=trajectory_quat)

# If successful, test-dynamics should not complain
abee.test_dynamics()

# Instantiate controller
u_lim, x_lim = abee.get_limits()

MPC_HORIZON = 10

# Create MPC Solver
# Select the parameter type with the argument param='P1'  - or 'P2', 'P3'
ctl = MPC(model=abee,
          dynamics=abee.model,
          param='P2',
          N=MPC_HORIZON,
          ulb=-u_lim, uub=u_lim,
          xlb=-x_lim, xub=x_lim,
          tuning_file=tuning_file_path)

# Reference tracking --------------------------------------------------------------------------------

# Track a static reference
x_d = abee.get_static_setpoint()
ctl.set_reference(x_d)
# Set initial state
x0 = abee.get_initial_pose()
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.model,
                                 controller=ctl.mpc_controller,
                                 time=80)
'''
t, y, u = sim_env.run(x0)
sim_env.visualize()  # Visualize state propagation
'''

# Activate Tracking - Track a time varying reference
tracking_ctl = MPC(model=abee,
                   dynamics=abee.model,
                   param='P2',
                   N=MPC_HORIZON,
                   trajectory_tracking=True,
                   ulb=-u_lim, uub=u_lim,
                   xlb=-x_lim, xub=x_lim)
sim_env_tracking = EmbeddedSimEnvironment(model=abee,
                                          dynamics=abee.model,
                                          controller=tracking_ctl.mpc_controller,
                                          time=80)  # 80
'''
t, y, u = sim_env_tracking.run(x0)
sim_env_tracking.visualize()  # Visualize state propagation
sim_env_tracking.visualize_error()
'''

# Activate forward propagation - Estimate Honey trajectory
abee.test_forward_propagation()
tracking_ctl.set_forward_propagation()
t, y, u = sim_env_tracking.run(x0)
sim_env_tracking.visualize()  # Visualize state propagation
sim_env_tracking.visualize_error()
sim_env_tracking.metrics_and_score()


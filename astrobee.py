from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca
import numpy as np
import numpy.matlib as nmp
from util import *


class Astrobee(object):
    def __init__(self,
                 trajectory_file,
                 mass=9.6,
                 inertia=np.diag([0.1534, 0.1427, 0.1623]),
                 h=0.1,
                 **kwargs):
        """
        Astrobee Robot, N5 tester class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param inertia: inertia tensor of the Astrobee
        :type inertia: np.diag
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        :param model: select between 'euler' or 'quat'
        :type model: str
        """

        # Model
        self.nonlinear_model = self.astrobee_dynamics_quat
        self.n = 13
        self.m = 6
        self.dt = h

        # Model prperties
        self.mass = mass
        self.inertia = inertia

        # Set CasADi functions
        self.set_casadi_options()

        # Set nonlinear model with a RK4 integrator
        self.model = self.rk4_integrator(self.nonlinear_model)

        # Set path for trajectory file
        self.trajectory_file = trajectory_file

    def set_casadi_options(self):
        """
        Helper function to set casadi options.
        """
        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

    def astrobee_dynamics_quat(self, x, u):
        """
        Astrobee nonlinear dynamics with Quaternions.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:]

        # 3D Force
        f = u[0:3]

        # 3D Torque
        tau = u[3:]

        # Model
        pdot = v
        vdot = ca.mtimes(r_mat_q(q), f)/self.mass
        qdot = ca.mtimes(xi_mat(q),w)/2
        wdot = ca.mtimes(ca.inv(self.inertia), tau-ca.cross(w, ca.mtimes(self.inertia, w)))  # ω = J^(−1)(t −ω×Jω)

        dxdt = [pdot, vdot, qdot, wdot]

        return ca.vertcat(*dxdt)

    def create_linearized_dynamics(self, x_bar=None, u_bar=None):
        """
        Helper function to populate Ac and Bc with continuous-time
        dynamics of the system.
        """

        # Set CasADi variables
        x = ca.MX.sym('x', self.n)
        u = ca.MX.sym('u', self.m)

        # Jacobian of exact discretization
        Ac = ca.Function('Ac', [x, u], [ca.jacobian(
                         self.astrobee_dynamics_quat(x, u), x)])
        Bc = ca.Function('Bc', [x, u], [ca.jacobian(
                         self.astrobee_dynamics_quat(x, u), u)])

        # Linearization points
        if x_bar is None:
            x_bar = np.zeros((13, 1))

        if u_bar is None:
            u_bar = np.zeros((6, 1))

        self.Ac = np.asarray(Ac(x_bar, u_bar))
        self.Bc = np.asarray(Bc(x_bar, u_bar))

        return self.Ac, self.Bc

    def casadi_c2d(self, A, B, x_bar=None, u_bar=None):
        """
        Continuous to Discrete-time dynamics
        """
        # Set CasADi variables
        x = ca.MX.sym('x', A.shape[1])
        u = ca.MX.sym('u', B.shape[1])

        # Create an ordinary differential equation dictionary. Notice that:
        # - the 'x' argument is the state
        # - the 'ode' contains the equation/function we wish to discretize
        # - the 'p' argument contains the parameters that our function/equation
        #   receives. For now, we will only need the control input u
        ode = {'x': x, 'ode': ca.DM(A) @ x + ca.DM(B) @ u, 'p': ca.vertcat(u)}

        # Here we define the options for our CasADi integrator - it will take care of the
        # numerical integration for us: fear integrals no more!
        options = {"abstol": 1e-5, "reltol": 1e-9, "max_num_steps": 100, "tf": self.dt}

        # Create the integrator
        self.Integrator = ca.integrator('integrator', 'cvodes', ode, options)

        # Now we have an integrator CasADi function. We wish now to take the partial
        # derivaties w.r.t. 'x', and 'u', to obtain Ad and Bd, respectively. That's wher
        # we use ca.jacobian passing the integrator we created before - and extracting its
        # value after the integration interval 'xf' (our dt) - and our variable of interest
        Ad = ca.Function('jac_x_Ad', [x, u], [ca.jacobian(
                         self.Integrator(x0=x, p=u)['xf'], x)])
        Bd = ca.Function('jac_u_Bd', [x, u], [ca.jacobian(
                         self.Integrator(x0=x, p=u)['xf'], u)])

        # If you print Ad and Bd, they will be functions that can be evaluated at any point.
        # Now we must extract their value at the linearization point of our chosing!
        if x_bar is None:
            x_bar = np.zeros((13, 1))

        if u_bar is None:
            u_bar = np.zeros((6, 1))

        return np.asarray(Ad(x_bar, u_bar)), np.asarray(Bd(x_bar, u_bar))

    def linearized_discrete_dynamics(self, x, u):
        """
        Method to propagate discrete-time dynamics for Astrobee

        :param x: state
        :type x: np.ndarray, ca.DM
        :param u: control input
        :type u: np.ndarray, ca.DM
        :return: state after dt seconds
        :rtype: np.ndarray, ca.DM
        """
        Ac, Bc = self.create_linearized_dynamics(x, u)
        Ad, Bd = self.casadi_c2d(Ac, Bc, x, u)

        # x_next = np.dot(Ad, x) + np.dot(Bd, u) 

        return Ad, Bd

    def rk4_integrator(self, dynamics):
        """
        Runge-Kutta 4th Order discretization.
        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state at next step
        :rtype: ca.MX
        """
        x0 = ca.MX.sym('x0', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)

        x = x0

        k1 = dynamics(x, u)
        k2 = dynamics(x + self.dt / 2 * k1, u)
        k3 = dynamics(x + self.dt / 2 * k2, u)
        k4 = dynamics(x + self.dt * k3, u)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion: TODO(Pedro-Roque): check best way to propagate
        rk4 = ca.Function('RK4', [x0, u], [xdot], self.fun_options)

        return rk4

    def forward_propagate(self, x_s, npoints, radius=0.5):
        """
        Forward propagate the observed state given a constant velocity.

        The output should be a self.n x npoints matrix, with the
        desired offset track.

        :param x_s: starting state
        :type x_s: np.ndarray
        :param npoints: number of points to propagate
        :type npoints: int
        :return: forward propagated trajectory
        :rtype: np.ndarray
        """

        # x_s = x_s.reshape((self.n, 1))

        x_r = np.zeros((self.n, npoints))
        r_mat = r_mat_q_np(x_s[6:10])[:,0]
        x_r[0:3, 0:1] = x_s[0:3].reshape(3, 1) + np.multiply(r_mat, np.array([radius, 0, 0])).reshape(3, 1)
        x_r[3:6, 0:1] = x_s[3:6].reshape(3, 1)
        x_r[6:10, 0:1] = x_s[6:10].reshape(4, 1)
        x_r[10:13, 0:1] = x_s[10:13].reshape(3, 1)
        for i in range(npoints-1):
            p_next = x_s[0:3] + x_s[3:6] * self.dt
            q_next = x_s[6:10] + np.dot(xi_mat_np(x_s[6:10]), x_s[10:13]) / 2 * self.dt
            x_s[0:3] = p_next
            x_s[6:10] = q_next
            r_mat = r_mat_q_np(x_s[6:10])[:,0]
            x_r[0:3, i+1:i+2] = x_s[0:3].reshape(3, 1) + np.multiply(r_mat, np.array([radius, 0, 0])).reshape(3, 1)  # Pb
            x_r[3:6, i+1:i+2] = x_s[3:6].reshape(3, 1)
            x_r[6:10, i+1:i+2] = (q_next / np.linalg.norm(q_next)).reshape(4, 1) # q_next
            x_r[10:13, i+1:i+2] = x_s[10:13].reshape(3, 1) 
        
        return x_r

    def get_trajectory(self, t, npoints, forward_propagation=False):
        """
        Provide trajectory to be followed.
        :param t0: starting time
        :type t0: float
        :param npoints: number of trajectory points
        :type npoints: int
        :return: trajectory with shape (Nx, npoints)
        :rtype: np.array
        """

        if t == 0.0:
            tmp = np.loadtxt(self.trajectory_file, ndmin=2)
            self.trajectory = tmp.reshape((self.n, int(tmp.shape[0] / self.n)), order="F")

        if forward_propagation is False:
            id_s = int(round(t / self.dt))
            id_e = int(round(t / self.dt)) + npoints
            x_r = self.trajectory[:, id_s:id_e]
        else:
            # Take a point and propagate the kinematics
            id_s = int(round(t / self.dt))
            x_start = self.trajectory[:, id_s]
            x_r = self.forward_propagate(x_start, npoints)

        return x_r

    def get_initial_pose(self):
        """
        Helper function to get a starting state, depending on the dynamics type.

        :return: starting state
        :rtype: np.ndarray
        """
        # project:
        # x0 = np.zeros((self.n, 1))
        # x0[0] = 11.0
        # x0[1] = -7.5
        # x0[2] = 5.2
        # x0[8] = 1.0

        # custom:
        x0 = np.zeros((self.n, 1))
        x0[0] = 8
        x0[1] = -3.5
        x0[2] = 1
        x0[8] = 1.0

        return x0

    def get_static_setpoint(self):
        """
        Helper function to get the initial state of Honey for setpoint stabilization.
        """
        xd = np.array([[11, -7, 4.8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).T

        return xd

    def get_limits(self):
        """
        Get Astrobee control and state limits for ISS

        :return: state and control limits
        :rtype: np.ndarray, np.ndarray
        """
        u_lim = np.array([[0.85, 0.41, 0.41, 0.085, 0.041, 0.041]]).T

        x_lim = np.array([[13, 13, 13,
                           0.5, 0.5, 0.5,
                           1, 1, 1, 1,
                           0.1, 0.1, 0.1]]).T

        return u_lim, x_lim

    # ----------------------------------------
    #               Unit Tests
    # ----------------------------------------

    def test_forward_propagation(self):
        """
        Unit test to check if forward propagation is correctly implemented.
        """

        x0 = np.array([[11, 0.3, 0.4, 0, 0.1, 0, 0, 0, 0, 1, 0.1, 0, 0]]).T
        x_r = self.forward_propagate(x0, 30)
        xd = np.array([11.5, 0.54, 0.4, 0.0, 0.1, 0.0, 0.11971121, 0.0, 0.0, 0.99280876, 0.1, 0.0, 0.0])
        eps = np.linalg.norm(x_r[:, 24] - xd)
        if eps > 1e-2:
            print("Forward propagation has a large error. Double check your forward propagation.")
            exit()

    def test_dynamics(self):
        """
        Unit test to check if the Astrobee dynamics are correctly set.
        """
        x0 = np.array([[11, 0.3, 0.4, 0, 0.1, 0, 0, 0, 0, 1, 0.1, 0, 0]]).T
        u0 = np.array([[.1, .1, .1, .01, .01, .01]])
        xd = np.array([[11.0001, 0.310052, 0.400052,
                        0.00104168, 0.101036, 0.00104685,
                        0.00516295, 0.000174902, 0.000154287, 0.999987,
                        0.106519, 0.0070057, 0.00615902]]).T
        xt = self.model(x0, u0)
        eps = np.linalg.norm(np.array(xt) - xd)
        if eps > 1e-4:
            print("Dynamic has large error. Double check your dynamics.")
            exit()

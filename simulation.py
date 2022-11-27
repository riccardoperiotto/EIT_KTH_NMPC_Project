import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time


class EmbeddedSimEnvironment(object):

    def __init__(self, model, dynamics, controller, time=100.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.total_sim_time = time  # seconds
        self.dt = self.model.dt
        self.estimation_in_the_loop = False
        self.total_time = 0.0
        self.max_ct = 0.0
        self.avg_ct = 0.0
        self.cvg_t = 10000.0
        self.ss_error = 0.0
        self.convergence_pose_error = 0.0
        self.convergence_attitude_error = 0.0

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt) + 1  # account for 0th
        t = np.array([0])
        x_vec = np.array([x0]).reshape(self.model.n, 1)
        u_vec = np.empty((6, 0))
        e_vec = np.empty((12, 0))

        cvg_error = np.empty((12,0))

        # For each stage i: get control input and obtain next state
        for i in range(sim_loop_length):

            # Take current state
            x = x_vec[:, -1].reshape(self.model.n, 1)
            
            # Get control input and error
            u, error, step_time = self.controller(x, i * self.dt)

            # Update the state
            x_next = self.dynamics(x, u)
            x_next[6:10] = x_next[6:10] / ca.norm_2(x_next[6:10])

            # Compute metrics
            self.total_time += step_time
            pose_error = np.linalg.norm(error[0:3])
            attitude_error = np.rad2deg(np.linalg.norm(error[6:9]))
            if self.cvg_t == 10000.0 and pose_error <= 0.05 and attitude_error <= 10.0:
                self.cvg_t = self.total_time
                self.convergence_pose_error = pose_error
                self.convergence_attitude_error = attitude_error
                cvg_error = error
            elif pose_error > 0.05 or attitude_error > 10.0:
                self.cvg_t = 10000.0
            if step_time > self.max_ct:
                self.max_ct = step_time

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(self.model.n, 1), axis=1)
            u_vec = np.append(u_vec, np.array(u).reshape(self.model.m, 1), axis=1)
            e_vec = np.append(e_vec, error.reshape(12, 1), axis=1)

        # Get error and metrics for the terminal stage
        _, error, step_time = self.controller(x_next, i * self.dt)
        self.total_time += step_time
        self.avg_ct = self.total_time/sim_loop_length
        e_vec = np.append(e_vec, error.reshape(12, 1), axis=1)

        # Save data
        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.e_vec = e_vec
        self.sim_loop_length = sim_loop_length
        self.ss_error = error

        print("congergence error: ", cvg_error)

        return t, x_vec, u_vec

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.x_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        '''
        print("------- SIMULATION STATUS -------")
        print("Energy used: ", np.sum(np.abs(u_vec * 10)))
        print("Position integral error: ", np.sum(np.abs(self.e_vec[0:3, :])))
        print("Attitude integral error: ", np.sum(np.abs(self.e_vec[6:9, :])))
        print("Computational total time: ", self.total_time)
        '''

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig2, (ax5, ax6) = plt.subplots(2)
        ax1.clear()
        ax1.set_title("Astrobee States")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position [m]")
        ax1.set_xlabel("Time [s]")
        ax1.grid()

        ax2.clear()
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x3", "x4", "x5"])
        ax2.set_ylabel("Velocity [m/s]")
        ax2.set_xlabel("Time [s]")
        ax2.grid()

        ax3.clear()
        ax3.plot(t, x_vec[6, :], 'r--',
                 t, x_vec[7, :], 'g--',
                 t, x_vec[8, :], 'b--')
        ax3.legend(["x6", "x7", "x8"])
        ax3.set_ylabel("Attitude [rad]")
        ax3.set_xlabel("Time [s]")
        ax3.grid()

        ax4.clear()
        ax4.plot(t, x_vec[10, :], 'r--',
                 t, x_vec[11, :], 'g--',
                 t, x_vec[12, :], 'b--')
        ax4.legend(["x9", "x10", "x11"])
        ax4.set_ylabel("Ang. velocity [rad/s]")
        ax4.set_xlabel("Time [s]")
        ax4.grid()

        # Plot control input
        ax5.clear()
        ax5.set_title("Astrobee Control inputs")
        ax5.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'g--',
                 t[:-1], u_vec[2, :], 'b--')
        ax5.legend(["u0", "u1", "u2"])
        ax5.set_ylabel("Force input [N]")
        ax5.set_xlabel("Time [s]")
        ax5.grid()

        ax6.clear()
        ax6.plot(t[:-1], u_vec[3, :], 'r--',
                 t[:-1], u_vec[4, :], 'g--',
                 t[:-1], u_vec[5, :], 'b--')
        ax6.legend(["u3", "u4", "u5"])
        ax6.set_ylabel("Torque input [Nm]")
        ax6.set_xlabel("Time [s]")
        ax6.grid()

        plt.show()

    def visualize_error(self):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.e_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.e_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig2, (ax5, ax6) = plt.subplots(2)
        ax1.clear()
        ax1.set_title("Trajectory Error")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position Error [m]")
        ax1.set_xlabel("Time [s]")
        ax1.grid()

        ax2.clear()
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x3", "x4", "x5"])
        ax2.set_ylabel("Velocity Error [m/s]")
        ax2.set_xlabel("Time [s]")
        ax2.grid()

        ax3.clear()
        ax3.plot(t, x_vec[6, :], 'r--',
                 t, x_vec[7, :], 'g--',
                 t, x_vec[8, :], 'b--')
        ax3.legend(["ex", "ey", "ez"])
        ax3.set_ylabel("Attitude Error [rad]")
        ax3.set_xlabel("Time [s]")
        ax3.grid()

        ax4.clear()
        ax4.plot(t, x_vec[9, :], 'r--',
                 t, x_vec[10, :], 'g--',
                 t, x_vec[11, :], 'b--')
        ax4.legend(["x9", "x10", "x11"])
        ax4.set_ylabel("Ang. velocity Error [rad/s]")
        ax4.set_xlabel("Time [s]")
        ax4.grid()

        # Plot control input
        ax5.clear()
        ax5.set_title("Astrobee Control inputs")
        ax5.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'g--',
                 t[:-1], u_vec[2, :], 'b--')
        ax5.legend(["u0", "u1", "u2"])
        ax5.set_ylabel("Force input [N]")
        ax5.set_xlabel("Time [s]")
        ax5.grid()

        ax6.clear()
        ax6.plot(t[:-1], u_vec[3, :], 'r--',
                 t[:-1], u_vec[4, :], 'g--',
                 t[:-1], u_vec[5, :], 'b--')
        ax6.legend(["u3", "u4", "u5"])
        ax6.set_ylabel("Torque input [Nm]")
        ax5.set_xlabel("Time [s]")
        ax6.grid()

        plt.show()

    def metrics_and_score(self):
        """
        Computation of performance metrics and aggregated score
        """
        score = 0.0
        score -= max(round((self.max_ct - 0.1) * 100, 3), 0.0) * 0.1
        # Penalize average above
        if self.avg_ct > 0.1:
            score += (0.1 - self.avg_ct) * 30
        else:
            score += max((0.1 - self.avg_ct), 0.0) * 5
        # Factor in convergence time
        score += max((35.0 - self.cvg_t), 0.0) * 0.1

        ss_p = np.sum(np.abs(self.ss_error[0:3]))
        ss_a = np.sum(np.abs(self.ss_error[6:9]))

        # Factor in steady-state errors
        score += (self.convergence_pose_error - ss_p) * 100
        # score += np.rad2deg(self.convergence_attitude_error - ss_a) * 1
        score += self.convergence_attitude_error - ss_a

        print("------- SIMULATION STATUS -------")
        print("Energy used: ", np.sum(np.abs(self.u_vec * 10)))
        print("Position integral error: ", np.sum(np.abs(self.e_vec[0:3, :])))
        print("Attitude integral error: ", np.sum(np.abs(self.e_vec[6:9, :])))
        print("Position steady state error: ", ss_p)
        print("Attitude steady state error: ", ss_a)
        print("Computational total time: ", self.total_time)
        print("Computational maximum time: ", self.max_ct)
        print("Computational average time: ", self.avg_ct)
        print("Convergence time: ", self.cvg_t)
        print("Aggregated score: ", score)
        print("Errors at convergence [pose,attitude]: [", self.convergence_pose_error, ",", self.convergence_attitude_error, "]")

from util import *
import time

dt = 0.1
n = 13
radius = 0.5

X_S = np.array([[11, 0.3, 0.4, 0, 0.1, 0, 0, 0, 0, 1, 0.1, 0, 0]]).T
npoints = 30

## FIRST CASE ##

ct = time.time()

x_s = X_S.reshape((n, 1))
x_r = np.zeros((n,npoints))  

pB = x_s[0:3] + np.multiply(r_mat_q_np(x_s[6:10])[:,0],np.array([radius,0,0])).reshape((3,1))

x_temp = np.concatenate([pB, x_s[3:6], x_s[6:10], x_s[10:]])
x_r[:,0] = x_temp[:,0]

for iter in range(npoints-1):
    
    pdot = x_s[3:6]
    qdot = np.dot(xi_mat_np(x_s[6:10]), x_s[10:])/2

    pnext = x_s[:3] + pdot*dt
    qnext = x_s[6:10] + qdot*dt
    
    qnext = qnext/np.linalg.norm(qnext)

    x_s = np.concatenate([pnext, x_s[3:6], qnext, x_s[10:]])

    pB = pnext + np.multiply(r_mat_q_np(x_s[6:10])[:,0],np.array([radius,0,0])).reshape((3,1))
    x_temp = np.concatenate([pB, x_s[3:6], qnext, x_s[10:]])

    x_r[:,iter+1] = x_temp[:,0]

# print('x_r', x_r)

ct = time.time() - ct
print('Time 1', ct)


## SECOND CASE ##

ct = time.time()

x_s = X_S.flatten()
x_r = np.zeros((n,npoints))  

pB = x_s[0:3] + np.multiply(r_mat_q_np(x_s[6:10])[:,0],np.array([radius,0,0]))

x_temp = np.concatenate([pB, x_s[3:6], x_s[6:10], x_s[10:]])
x_r[:,0] = x_temp

for iter in range(npoints-1):
    
    pdot = x_s[3:6]
    qdot = np.dot(xi_mat_np(x_s[6:10]), x_s[10:])/2

    pnext = x_s[:3] + pdot*dt
    qnext = x_s[6:10] + qdot*dt
    
    qnext = qnext/np.linalg.norm(qnext)

    x_s = np.concatenate([pnext, x_s[3:6], qnext, x_s[10:]])

    pB = pnext + np.multiply(r_mat_q_np(x_s[6:10])[:,0],np.array([radius,0,0]))
    x_temp = np.concatenate([pB, x_s[3:6], qnext, x_s[10:]])

    x_r[:,iter+1] = x_temp

# print('x_r', x_r)

ct = time.time() - ct
print('Time 2', ct)


## THIRD CASE ##

ct = time.time()

x_s = np.reshape(X_S,(13,1))
x_r=np.zeros((n,npoints))
pB = x_s[0:3,0] + np.dot(r_mat_q_np(x_s[6:10,0]),np.array([radius,0,0]))
pB = np.reshape(pB,(3,1))
#x_temp = [pB, x_s[3:6], x_s[6:10], x_s[10:]]
x_temp = np.zeros((n,1))
x_temp[0:3] = pB
x_temp[3:6] = x_s[3:6]
x_temp[6:10] = x_s[6:10]
x_temp[10:] = x_s[10:]
x_r[:,0:1] = x_temp

for iter in range(npoints-1):
    pdot = x_s[3:6]
    #vdot = ca.mtimes(r_mat_q(q), f)/self.mass
    qdot = np.dot(xi_mat_np(x_s[6:10]), x_s[10:])/2
    #wdot = ca.mtimes(ca.inv(self.inertia), tau-ca.cross(w, ca.mtimes(self.inertia, w))) # ω = J^(−1)(t −ω×Jω)

    pnext = x_s[:3] + pdot*dt
    qnext = x_s[6:10] + qdot*dt

    qnext = qnext/np.linalg.norm(qnext)

    x_s[0:3] = pnext
    x_s[3:6] = x_s[3:6]
    x_s[6:10] = qnext
    x_s[10:] = x_s[10:]

    x_s = np.reshape(x_s,(13,1))

    pB = x_s[0:3].transpose() + np.dot(r_mat_q_np(x_s[6:10]),np.array([radius,0,0]))
    pB = np.reshape(pB,(3,1))
    x_temp[0:3] = pB
    x_temp[3:6] = x_s[3:6]
    x_temp[6:10] = qnext
    x_temp[10:] = x_s[10:]

    x_r[:,iter+1:iter+2] = x_temp

# print('x_r', x_r)

ct = time.time() - ct
print('Time 3', ct)


########################################################

# test code for error computation

'''
? faster  ?

xr = xr.reshape((xr.shape[0], 1))
e = x - xr
eq = ca.DM.ones(3, 1) * (1 - ca.mtimes(qr.T, q)**2)
return np.concatenate((e[0:6],eq,e[10:])).reshape((12,1))
'''


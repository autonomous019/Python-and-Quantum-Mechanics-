#%matplotlib inline

#from matplotlib import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from qutip import *

chi = 0.2
N1 = 5
N2 = 5

a = tensor(destroy(N1), qeye(N2))
na = tensor(num(N1),     qeye(N2))
b = tensor(qeye(N1),    destroy(N2))
nb = tensor(qeye(N1),    num(N2))
#qeye is identity operator in qutip 
H = - chi * (a * b + a.dag() * b.dag())
#QObj.dag Returns adjoint (dagger) of object.

#print(a)

#print(b)

print(H)


# start in the ground (vacuum) state
psi0 = tensor(basis(N1,0), basis(N2,0))
tlist = np.linspace(0, 10, 100)
c_ops = []
e_ops = []
output = mesolve(H, psi0, tlist, c_ops, e_ops)
output

na_e = np.zeros(shape(tlist))
na_s = np.zeros(shape(tlist))
nb_e = np.zeros(shape(tlist))
nb_s = np.zeros(shape(tlist))

for idx, psi in enumerate(output.states):
    na_e[idx] = expect(na, psi)
    na_s[idx] = expect(na*na, psi)
    nb_e[idx] = expect(nb, psi)
    nb_s[idx] = expect(nb*nb, psi)

# substract the average squared to obtain variances
na_s = na_s - na_e ** 2
nb_s = nb_s - nb_e ** 2
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8,5))

line1 = axes[0,0].plot(tlist, na_e, 'r', linewidth=2)
axes[0,0].set_ylabel(r'$\langle a^\dagger a \rangle$', fontsize=18)

line2 = axes[0,1].plot(tlist, nb_e, 'b', linewidth=2)

line3 = axes[1,0].plot(tlist, na_s, 'r', linewidth=2)
axes[1,0].set_xlabel('$t$', fontsize=18)
axes[1,0].set_ylabel(r'$Std[a^\dagger a]$, $Std[b^\dagger b]$', fontsize=18)

line4 = axes[1,1].plot(tlist, nb_s, 'b', linewidth=2)
axes[1,1].set_xlabel('$t$', fontsize=18)

fig.tight_layout()

# pick an arbitrary time and calculate the wigner functions for each mode
xvec = np.linspace(-5, 5, 200)
t_idx_vec = [0, 10, 20, 30]

fig, axes = plt.subplots(len(t_idx_vec), 2, sharex=True, sharey=True, figsize=(8, 4 * len(t_idx_vec)))

for idx, t_idx in enumerate(t_idx_vec):
    psi_a = ptrace(output.states[t_idx], 0)
    psi_b = ptrace(output.states[t_idx], 1)
    W_a = wigner(psi_a, xvec, xvec)
    W_b = wigner(psi_b, xvec, xvec)

    cont1 = axes[idx, 0].contourf(xvec, xvec, W_a, 100)
    cont2 = axes[idx, 1].contourf(xvec, xvec, W_b, 100)





#entanglement logarithmic negativity section

R_op = correlation_matrix_quadrature(a, b)


def plot_covariance_matrix(V, ax):
    """
    Plot a matrix-histogram representation of the supplied Wigner covariance matrix.
    """
    num_elem = 16
    xpos, ypos = np.meshgrid(range(4), range(4))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(num_elem)
    dx = 0.75 * np.ones(num_elem)
    dy = dx.copy()
    dz = V.flatten()

    #nrm = colors.Normalize(-0.5, 0.5)
    #colors = cm.jet(nrm((np.sign(dz) * abs(dz) ** 0.75)))

    ax.view_init(azim=-40, elev=60)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    ax.axes.w_xaxis.set_ticklabels(("$q_-$", "$p_-$", "$q_+$", "$p_+$"), fontsize=20)
    ax.axes.w_yaxis.set_ticklabels(("$q_-$", "$p_-$", "$q_+$", "$p_+$"), fontsize=20)


# pick arbitrary times and plot the photon distributions at those times
t_idx_vec = [0, 20, 40]

fig, axes = plt.subplots(len(t_idx_vec), 1, subplot_kw={'projection': '3d'}, figsize=(6, 3 * len(t_idx_vec)))

for idx, t_idx in enumerate(t_idx_vec):
    # calculate the wigner covariance matrix
    V = wigner_covariance_matrix(R=R_op, rho=output.states[idx])

    plot_covariance_matrix(V, axes[idx])

fig.tight_layout()
plt.show()



























# %%
# import sys
import numpy as np
import matplotlib.pyplot as plt
import fipy
# %%
Nx = 40
Ny = 30

x_max = 0.7
y_max = 0.7

dx = x_max/Nx
dy = y_max/Ny

mesh = fipy.Grid2D(dx=dx, dy=dy, nx=Nx, ny=Ny)
x, y = mesh.cellCenters

p = fipy.CellVariable(mesh=mesh, name=r"$P$", value = 1/(Nx*Ny))


# psi = f(x) - div (D(x))
# f(x) = (-2(x1-0.5), -6(x2-0.4))
# D(x) = 0.5*[[x1**2,0],[0,x2**2]]
# div(D(x)) = d/dx1 (0.5*x1**2,0) + d/dx2 (0,0.5*x2**2) = (x1, x2)
psi = fipy.CellVariable(mesh=mesh, value = [-3*x+1, -7*y+2.4])
D = fipy.CellVariable(mesh=mesh, value = [0.5*x**2, 0.5*y**2])

# %%
eq = fipy.ConvectionTerm(coeff=psi,var=p) == fipy.DiffusionTerm(coeff=D,var=p)
eq.sweep(var=p)

p_sol = p.value.reshape(Ny,Nx)
# integrate over x
C = np.trapz(p_sol, dx=dx, axis=1)
# integrate over y
C = np.trapz(C, dx=dy)
p_sol = p_sol/C

fig, ax = plt.subplots()
ax.pcolormesh(p_sol)
# add colorbar
cbar = plt.colorbar(ax.pcolormesh(p_sol));

# reset x labels to reflect range from 0 to x_max
ax.set_xticks(np.linspace(0,Nx,10))
ax.set_xticklabels(np.round(np.linspace(0,x_max,10),2)) # round to 2 decimal places
# reset y labels to reflect range from 0 to y_max
ax.set_yticks(np.linspace(0,Ny,10))
ax.set_yticklabels(np.round(np.linspace(0,y_max,10),2))


# %%
Nx = 40
Ny = 30

x_min = 0.0
y_min = 0.0
x_max = 0.7
y_max = 0.7

dx = (x_max-x_min)/Nx
dy = (y_max-y_min)/Ny

mesh = fipy.Grid2D(dx=dx, dy=dy, nx=Nx, ny=Ny)
x, y = mesh.cellCenters

p = fipy.CellVariable(mesh=mesh, name=r"$P$", value = 1/(Nx*Ny))

def f(x):
    return np.array([-2*(x[0]-0.5), -6*(x[1]-0.4)])

def D(x):
    return np.array([0.5*x[0]**2,0.5*x[1]**2])

f_vals = f([x,y]).reshape(2,Ny,Nx)
D_vals = D([x,y]).reshape(2,Ny,Nx)

# get Div(D)
dX= [dx,dy]
divD = np.zeros_like(f_vals)
for i in range(D_vals.shape[0]):
    divD[i] = np.gradient(D_vals[i], dX[i], axis=(i+1)%2, edge_order=2)
# %%
# check f_vals, D_vals, divD
x_vec = np.linspace(x_min,x_max,Nx)
y_vec = np.linspace(y_min,y_max,Ny)
fig, ax = plt.subplots()
ax.quiver(x_vec, y_vec, f_vals[0], f_vals[1])

fig, ax = plt.subplots()
ax.streamplot(x_vec, y_vec, f_vals[0], f_vals[1])

fig, ax = plt.subplots()
ax.streamplot(x_vec, y_vec, f_vals[0]-divD[0], f_vals[1]-divD[1])

fig, ax = plt.subplots()
ax.pcolormesh(D_vals[0])

fig, ax = plt.subplots()
ax.pcolormesh(D_vals[1])

fig, ax = plt.subplots()
ax.pcolormesh(divD[0])

fig, ax = plt.subplots()
ax.pcolormesh(divD[1])
# %%
# psi = f(x) - div (D(x))
# f(x) = (-2(x1-0.5), -6(x2-0.4))
# D(x) = 0.5*[[x1**2,0],[0,x2**2]]
# div(D(x)) = d/dx1 (0.5*x1**2,0) + d/dx2 (0,0.5*x2**2) = (x1, x2)
f_vals = f_vals.reshape(2,-1)
D_vals = D_vals.reshape(2,-1)
divD = divD.reshape(2,-1)
psi = fipy.CellVariable(mesh=mesh, value = [f_vals[0]-divD[0], f_vals[1]-divD[1]])
D = fipy.CellVariable(mesh=mesh, value = [D_vals[0],D_vals[1]])

# %%
eq = fipy.ConvectionTerm(coeff=psi,var=p) == fipy.DiffusionTerm(coeff=D,var=p)
eq.sweep(var=p)

p_sol = p.value.reshape(Ny,Nx)
# integrate over x
C = np.trapz(p_sol, dx=dx, axis=1)
# integrate over y
C = np.trapz(C, dx=dy)
p_sol = p_sol/C

fig, ax = plt.subplots()
ax.pcolormesh(p_sol)
# add colorbar
cbar = plt.colorbar(ax.pcolormesh(p_sol));

# reset x labels to reflect range from 0 to x_max
ax.set_xticks(np.linspace(0,Nx,10))
ax.set_xticklabels(np.round(np.linspace(x_min,x_max,10),2)) # round to 2 decimal places
# reset y labels to reflect range from 0 to y_max
ax.set_yticks(np.linspace(0,Ny,10))
ax.set_yticklabels(np.round(np.linspace(x_min,y_max,10),2))
# %%

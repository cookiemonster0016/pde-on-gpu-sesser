# PorousConvection

[![Build Status](https:/github.com/cookiemonster0016/pde-on-gpu-sesser/actions/workflows/CI.yml/badge.svg)](https://github.com/cookiemonster0016/pde-on-gpu-sesser/actions/workflows/CI.yml)

**Motivation**
The goal of this work is to simulate three-dimensional porous convection driven by a vertical temperature gradient. Particular emphasis is placed on an efficient and scalable numerical implementation using domain decomposition and GPU acceleration. The correctness and scalability of the method are assessed by comparing single-GPU and multi-GPU results.

**Governing Equations**

The system is described by Darcay law (and mass conservation):

Describing how the temperature changes depending on buoyancy and transport in a porous medium.

$$
\mathbf{q}
=
- k \left( \nabla P - \rho(T)\,\mathbf{g} \right)
$$

Mass conservation:
$$
\nabla \cdot \mathbf{q} = 0
$$

and the advection diffusion equation:

$$
\phi \frac{\partial T}{\partial t}
+ \mathbf{q} \cdot \nabla T
=
\lambda \nabla^{2} T
$$

Describing how the temperature 
changes depending on the fluid flow.

**Numerical Methods**

The equations stated before are discretized and and solved on a staggered grid, with the temperature and pressure fluxes defined on the cell faces and the temperature and pressure in the cell centers.

They are solved implicitly using the pseudo transient method for the temperature and the pressure.

$$
\mathcal{F}(u) = 0
$$

The idea is to interduce an artificial time $\tau$

$$
\frac{\partial u}{\partial \tau} + \mathcal{F}(u) = 0
$$

And relax the systhem until it reaches a steady state:
$$
u^{k+1} = u^{k} - \Delta \tau \, \mathcal{F}(u^{k})
$$

If the steady state is reached for the temperature if:
$$
- \phi \frac{\partial T}{\partial t}
- \mathbf{q} \cdot \nabla T
+ \lambda \nabla^{2} T = 0 $$
and for the pressure:
$$\nabla \cdot \mathbf{q} = 0$$

This derivatives are computed numerically by central differences.

Then the whole systhem goes forward a real timestep.

In one pseudotimestep the following steps need to be done to relax the system:

1. Compute the Darcy fluxes:
$$ \mathbf{q}_D \leftarrow \mathbf{q}_D - \theta_D \left( \mathbf{q}_D +
k \left( \nabla P - \rho(T)\,\mathbf{g} \right)
\right)
$$

2. Update the pressures

$$P \leftarrow P - \beta_D \, \nabla \cdot \mathbf{q}_D $$

3. Compute the termal diffusion fluxes:
$$
\mathbf{q}_T \leftarrow
\mathbf{q}_T
-
\theta_T
\left(
\mathbf{q}_T
+
\lambda \nabla T
\right)
$$

4. Compute the material derivative of the temperature with the upwinding scheme
$$
\frac{DT}{Dt} = \frac{T - T_{\text{old}}}{\Delta t} + \frac{1}{\phi}\, \mathbf{q}_D\cdot \nabla T $$

5. Update the temperature:
$$
T \leftarrow
T - \frac{\displaystyle \frac{\mathrm{D}T}{\mathrm{D}t} + \nabla \cdot \mathbf{q}_T }
{\displaystyle 1 /\Delta t + \beta_T}
$$

6. Apply the boundary conditions.

7. Then the temperature and the pressure residual are computed,

$$ r_T = - \phi \frac{\partial T}{\partial t}
- \mathbf{q} \cdot \nabla T
+ \lambda \nabla^{2} T $$
and for the pressure:
$$r_p = \nabla \cdot \mathbf{q}$$

and only if both of them are below a defined threshhold the next timestep begins.


**Conclusion and Results**

This is the simulation ran on 32 GPUs with a resolution of 1012 * 500 * 250 for 2000 timesteps.

<video width="320" height="240" controls>
  <source src="./plots/out_T.mp4" type="video/mp4">
</video>

A three-dimensional porous convection problem was successfully implemented and solved using a finite-difference method with GPU acceleration and MPI-based domain decomposition. Additionally it can also be run with CPU and multiple CPUs

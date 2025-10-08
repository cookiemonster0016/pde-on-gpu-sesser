All plots and animations which not explicitly asked to put in the read me are still saved in this folder.
I plotted the error of the iterative loop only for the last timestep, but there is nearly no difference to the errorplots of the other timesteps and to not obtain 10 similar plots of the error I only included one here.

Exercise 1:

![./anim_diff.gif]

Here the diffusion equation is solved implicit instead of explicit.
For the implicit solution we do not simpy invert the tridiagonal matrix discribing our Problem, insted an iterative scheme is used.
A pseudo time $\tau$ is interduced and we discretize the pseudo time and solve for the next pseudotimestep ($c^{k+1}$) until we reach a steady state.
$$\frac{dc}{d\tau} \nearlyequal \frac{c^{k+1}-c^k}{d\tau}= - \frac{dc}{dt} + \frac{d^2 c}{dx^2}$$
If we achived a steady state $\frac{dc}{d\tau} = 0$ and we solve exactly our original problem.
If we always reach a steady state by solving for $c^{k+1}$ is not very trivial and can be shown if we look at our original equation as a diffusion equation with damping or if we use fourier analysis. Now it would be very interesting to analize if this solving method will converge for different pdes (not only diffusion)

Exercise2:
![./adv_diff.png]
Compared to the exercise of last week we can now choose our time step exactly to be the correct size to avoid numerical diffusion in the transport equation, because due to the implicit solve we have no restrictions anymore to the maximum step size of the diffusion part. This is very nice.

Exercise 3:
![./conv_vs_fact.png]
In this exercise we investigate different pseudotimestep sizes. We can observe that there are idfferent amounts of iterations needed depending on the size of tau and rho. We can see that we can go down to around 6 iterations per cell, which is very effective. If we would not use the implicit solve but solve the systhem with inverting the n by n matrix, where n is the number of cells in x direction, we would have a cost of $n^3$ (probably a bit less, because the matrix is sparse and tridiagonal) where we now only have a cost of $6n$, which is fantastic (I checked this for different n and it is always around 6 so it does not grow or decrease significantly for different n). 
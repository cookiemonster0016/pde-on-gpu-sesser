Task 1
 ![Poreus Convection Explicit](./poreus_convection_2D.gif)

Task2
 ![Poreus Convection Implicit ](./poreus_convection_implicit_2D.gif)

The fully implicit method only needs around half the amount of iterations compared to the method where the temperature step is done explicitly. A very interesting observation is that this is already true for the first timestep (both methods have the same initial conditions). I would assume, that because the temperature and the pressure are relaxed together in a coupled system (high Ra = 1000 in this case) this might be an advantage to also relax them together as they are coupled. They somehow must "support" each other to faster reach a converged solution. In the fully implicit method the temperature does not change rapidly and the pressure needs to adjust in many steps, instead the pressure and the temperature slowly converge simoltaniously. It is also important to note, that the fully implicit solver is not necessarily faster only because it needs less iterations. One itertation does more calculations in the fully implicit method.


Task3
 ![Poreus Convection Implicit Ra = 1000](./poreus_convection_implicit_2D.gif)

![Poreus Convection Implicit Ra = 100](./poreus_convection_implicit_2D_Ra=100.gif)

 ![Poreus Convection Implicit Ra = 40](./poreus_convection_implicit_2D_Ra=40.gif)

 ![Poreus Convection Implicit Ra = 10](./poreus_convection_implicit_2D_Ra=10.gif)

The results match the described behaviour in the task.
As Rayleigh number increases, the solution transitions from diffusion-dominated to more convection-dominated patterns. Low Ra results show nearly symmetric, slowly decaying temperature profiles. Ra controls the strength of convective part, and higher values lead to more motion due to temperature differences.
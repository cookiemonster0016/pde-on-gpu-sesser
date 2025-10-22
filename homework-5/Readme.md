Task 1
 ![Poreus Convection Explicit](./poreus_convection_2D.gif)

Task2
 ![Poreus Convection Implicit ](./poreus_convection_implicit_2D.gif)




Did the number of iterations required for convergence change compared to the version with the explicit temperature update? Try to come up with the explanation for why the number of iterations changed the way it changed and write a sentence about your thoughts on the topic.

The fully implicit method only needs around half the amount of iterations compared to the method where the temperature step is done explicitly. A very interesting observation is that this is already true for the first timestep. I would assume, that because the temperature and the pressure are relaxed together in a coupled system (high Ra = 1000 in this case) this might be an advantage to also relax them together as they are coupled. They somehow must "support" each other to faster reach a converged solution. But honestly i do not know exactly why this is. Todo find out why this is the case. 


Task3
 ![Poreus Convection Implicit Ra = 1000](./poreus_convection_implicit_2D.gif)

  ![Poreus Convection Implicit Ra = 100](./poreus_convection_implicit_2D_Ra=100.gif)

 ![Poreus Convection Implicit Ra = 40](./poreus_convection_implicit_2D_Ra=40.gif)

 ![Poreus Convection Implicit Ra = 10](./poreus_convection_implicit_2D_Ra=10.gif)

 As Rayleigh number increases, the solution transitions from diffusion-dominated to more convection-influenced patterns. Low Ra results show nearly symmetric, slowly decaying temperature profiles. Ra controls the strength of convective part, and higher values lead to more motion due to temperature differences.
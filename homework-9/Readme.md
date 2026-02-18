 
 Task 1.
 
 ![1D diffusion with 2 processors](./diffusion_1D_2procs.png)
  ![1D diffusion with n processors](./diffusion_1D_nprocs.png)

 One can see how the processors communicate correctly with each other.

  Task 2.
![2D diffusion with 4 processors on MPI](./diffusion_2D_MPI.gif)
This is the 2D diffusion with 4 processors communicating with MPI.
This command was used to run the code, becasue we need to launch MPI inside of the mpiexecjl 
~/.julia/bin/mpiexecjl -n 4 julia --project l9_diffusion_2D_mpi.jl
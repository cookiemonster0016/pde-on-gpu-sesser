# # 2D linear diffusion Julia MPI solver
# run: ~/.julia/bin/mpiexecjl -n 4 julia --project diffusion_2D_mpi.jl
using Plots, Printf, MAT
import MPI, CUDA

# enable plotting by default
if !@isdefined do_save; do_save = true end

# MPI functions
@views function update_halo!(A, neighbors_x, neighbors_y, comm)
    # Send to / receive from neighbor 1 in dimension x ("left neighbor")
    if neighbors_x[1] != MPI.PROC_NULL
        #1. start defining a send buffer
        sendbuf = CUDA.zeros(size(A[2,:]))
        copyto!(sendbuf, A[2,:])
        #2. initialize a recive bugger
        recvbuf = CUDA.zeros(size(A[1,:]))
        #3. use mpiSend and MPI Recv
        MPI.Send(sendbuf,  neighbors_x[1], 0, comm)
        MPI.Recv!(recvbuf, neighbors_x[1], 1, comm)
        #4. Assign values from recv buffer to column of A
        copyto!(A[1,:], recvbuf)
    end
    # Send to / receive from neighbor 2 in dimension x ("right neighbor")
    if neighbors_x[2] != MPI.PROC_NULL
        sendbuf = CUDA.zeros(size(A[end-1,:]))
        copyto!(sendbuf, A[end-1,:])

        recvbuf = CUDA.zeros(size(A[end,:]))#must this be a CUDA Array
        MPI.Recv!(recvbuf, neighbors_x[2], 0, comm)
        MPI.Send(sendbuf,  neighbors_x[2], 1, comm)
        copyto!(A[end,:], recvbuf)
    end
    # Send to / receive from neighbor 1 in dimension y ("bottom neighbor")
    if neighbors_y[1] != MPI.PROC_NULL
        sendbuf = CUDA.zeros(size(A[:,2]))
        copyto!(sendbuf, A[:,2])
        recvbuf = CUDA.zeros(size(A[:,1]))
        MPI.Send(sendbuf,  neighbors_y[1], 2, comm)
        MPI.Recv!(recvbuf, neighbors_y[1], 3, comm)
        copyto!(A[:,1], recvbuf)
    end
    # Send to / receive from neighbor 2 in dimension y ("top neighbor")
    if neighbors_y[2] != MPI.PROC_NULL
        sendbuf = CUDA.zeros(size(A[:,end-1]))
        copyto!(sendbuf, A[:,end-1])
        recvbuf = CUDA.zeros(size(A[:,end]))
        MPI.Recv!(recvbuf, neighbors_y[2], 2, comm)
        MPI.Send(sendbuf,  neighbors_y[2], 3, comm)
        copyto!(A[:,end], recvbuf)
    end
    return
end

@views function diffusion_2D_mpi(; do_save=true)
    # MPI
    MPI.Init()

    comm = MPI.COMM_WORLD
    me = MPI.Comm_rank(comm)
    # select device
    # comm_l = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, me)
    # me_l = MPI.Comm_rank(comm_l)
    # GPU_ID = CUDA.device!(me_l)
    GPU_ID = CUDA.device!(0) # on daint.alps each MPI proc sees a single GPU with ID=0
    sleep(0.1me)
    println("Hello world, I am $(me) of $(MPI.Comm_size(comm)) using $(GPU_ID)")
    MPI.Barrier(comm)
    # Physics
    lx, ly     = 10.0, 10.0
    D          = 1.0
    nt         = 100
    nvis       = 5
    itvis      = 1
    # Numerics
    nx, ny     = 32, 32                             # local number of grid points
    nx_g, ny_g = dims[1]*(nx-2)+2, dims[2]*(ny-2)+2 # global number of grid points
    # Derived numerics
    dx, dy     = lx/nx_g, ly/ny_g                   # global
    dt         = min(dx,dy)^2/D/4.1
    # Array allocation
    qx         = CUDA.zeros(nx-1,ny-2)
    qy         = CUDA.zeros(nx-2,ny-1)
    # Initial condition
    x0, y0     = coords[1]*(nx-2)*dx, coords[2]*(ny-2)*dy
    xc         = [x0 + ix*dx - dx/2 - 0.5*lx  for ix=1:nx]
    yc         = [y0 + iy*dy - dy/2 - 0.5*ly  for iy=1:ny]
    C          = CuArray(exp.(.-xc.^2 .-yc'.^2))
    t_tic = 0.0
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time() end
        qx  .= .-D*diff(C[:,2:end-1], dims=1)/dx
        qy  .= .-D*diff(C[2:end-1,:], dims=2)/dy
        C[2:end-1,2:end-1] .= C[2:end-1,2:end-1] .- dt*(diff(qx, dims=1)/dx .+ diff(qy, dims=2)/dy)
        update_halo!(C, neighbors_x, neighbors_y, comm_cart)

        if do_save && it%nvis == 0
            file = matopen("$(@__DIR__)/output/mpi2D_out_C_$(me)_$(itvis).mat", "w"); write(file, "C", Array(C)); close(file) 
            itvis += 1
        end

    end
    t_toc = (Base.time()-t_tic)
    if (me==0) @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2/1e9*nx*ny*sizeof(lx))/(t_toc/(nt-10)), sigdigits=2)) end
    # Save to visualise
    if do_save file = matopen("$(@__DIR__)/mpi2D_out_C_$(me).mat", "w"); write(file, "C", Array(C)); close(file) end
    MPI.Finalize()
    return
end

diffusion_2D_mpi(; do_save=do_save)
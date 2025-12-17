#!/bin/bash -l
#SBATCH --account class04
#SBATCH --job-name="convect2D"
#SBATCH --output=convect2D.o
#SBATCH --error=convect2D.e
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

srun --uenv julia/25.5:v1 --view=juliaup julia --project porousConvection_2D_xpu.jl
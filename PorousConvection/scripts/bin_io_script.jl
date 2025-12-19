include("../src/utils.jl")

using Random 
using Plots

"""
main()

Generate a random 3×3 array, save it to disk, load it back,
and return the loaded array.
"""
function main()
    # generate random 3x3 array
    A = rand(3, 3)

    # initialize array for reading
    B = zeros(3, 3)
    # save array to file
    save_array("array.bin", A)

    # load array from file
    B = load_array("array.bin", B)

    return B
end

# run main and plot result
B = main()
heatmap(B)
using Random 
using Plots

"""
save_array(name, A)

saves array A to location name.bin
"""
function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end

"""
load_array(name, A)

loads an array A from location name.bin
"""
function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

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
    load_array("array.bin", B)

    return B
end

# run main and plot result
B = main()
heatmap(B)
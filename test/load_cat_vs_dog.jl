using HDF5
using Plots: plot, heatmap
function data_load()
    permute(a) = permutedims(read(a), collect(reverse(1:ndims(a))))
    @show pwd()
    X_train = permute(h5open("test/data/trainset.hdf5", "r")["X_train"])
    y_train = permute(h5open("test/data/trainset.hdf5", "r")["Y_train"])
    X_test = permute(h5open("test/data/testset.hdf5", "r")["X_test"])
    y_test = permute(h5open("test/data/testset.hdf5", "r")["Y_test"])
    return X_train, y_train, X_test, y_test
end

function normalize(x::Array)
    return x ./ 254
end
function flatten_image(x::Array)
    # flatten_array = reshape(x, (size(x)[begin], :))
    flatten_array = Array{Any, 2}(undef, (size(x)[begin], size(x)[end]^2))
    for i in 1:size(x)[begin]
        # println(collect(Iterators.flatten(x[i, :, :])))
        flatten_array[i, :] = collect(Iterators.flatten(x[i, :, :]))
    end
    return flatten_array
end




function preprocess_data()
    X_train, y_train, X_test, y_test = data_load()
    X_train, X_test = flatten_image(X_train), flatten_image(X_test)
    X_train, X_test = normalize.((X_train, X_test))
    
    return X_train, y_train, X_test, y_test
end

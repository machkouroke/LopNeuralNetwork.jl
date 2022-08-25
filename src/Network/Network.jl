include("function.jl")

const SEED = 3
mutable struct NeuralNetwork
    W::Dict{Int64, Matrix{Float64}}
    b::Dict{Int64, Matrix{Float64}}
    α::Float64
    number_per_layer::Array{Int64}
    seed::TaskLocalRNG
    """
        init(x::Neuron, X::AbstractMatrix)
    Initialisation of a neural network with random weights and bias
    # Arguments:
    - `number_of_feature::Int64`: number of features of the input data
    - `number_per_layer::Array`: the number of neurons per layer 
    """
    function NeuralNetwork(number_of_feature::Int64, number_per_layer::Array; seed_value::TaskLocalRNG=seed!(SEED))
        layer = [number_of_feature, number_per_layer..., 1]
        W = Dict(i => randn(seed_value, layer[i+1], layer[i]) for i in 1:length(layer)-1)
        b = Dict(i => randn(seed_value, layer[i+1], 1) for i in 1:length(layer)-1)
        new(W, b, 0.01, layer, seed_value)
    end
end


"""
    forward_propagation(x::NeuralNetwork, X::AbstractMatrix)
Forward propagation of the neural network. For a given input X, the function returns the output of the neural network
# Arguments:
- `network::NeuralNetwork`: the neural network
- `X::AbstractMatrix`: the input data of size (number_of_feature, number_of_sample)
"""
function forward_propagation(network::NeuralNetwork, X::Array)::Dict{Int64, Matrix{Float64}}
    A = Dict{Int64, Matrix{Float64}}(0 => X)
    for i in 1:size(network.number_per_layer)[1] - 1
        answer = A[i-1]
        Z = z(network.W[i], answer, network.b[i])
        A[i] = a(Z)
    end
    return A
end

function back_propagation(network::NeuralNetwork, Y::Array, A::Dict{Int64, Matrix{Float64}})::Tuple{Dict{Int64, Matrix{Float64}}, Dict{Int64, Matrix{Float64}}}
    number_of_layer = size(network.number_per_layer)[1] - 1
    dW, db = [Dict() for i in 1:3]
    dZ = A[number_of_layer] .- Y
    m = size(Y)[2]
    for i in number_of_layer:-1:1
        dW[i] = (1 / m) .* (dZ * A[i-1]')
        db[i] = (1 / m) .* sum(dZ, dims=2)
        if i > 1
            dZ = (network.W[i]' * dZ) .* (A[i-1] .* (1 .- A[i-1]))
        end
    end
    return dW, db
end

function predict(network::NeuralNetwork, data::Array)
    final_result = forward_propagation(network, data)[length(network.number_per_layer) - 1]
    # @show final_result
    return final_result .>= 0.5
end
function fit!(network::NeuralNetwork, data::Array, output::Array, data_test, y_test)
    loss, accuracy, accuracy_test = gradient!(network, data, output, data_test, y_test)
    return loss, accuracy, accuracy_test
end
function score(network::NeuralNetwork, data::AbstractMatrix, output::AbstractMatrix)
    return  sum(predict(network, data) .== output) / size(output)[2]
end


function gradient!(network::NeuralNetwork, data::AbstractMatrix, output::AbstractMatrix, data_test, output_test; iter::Int64=1000)
    loss = []
    accuracy = []
    accuracy_test = []
    println("Start of gradient")
    for _ in ProgressBar(1:iter)
        # Neurone update
        A = forward_propagation(network, data)
        dW, db = back_propagation(network, output, A)
        network.W, network.b = update(dW, db, network.W, network.b, network.α)

        # Accuracy update
        push!(loss, log_loss(A[length(network.number_per_layer) - 1], output))
        y_pred = predict(network, data)
        push!(accuracy, binary_accuracy(y_pred', output'))
        y_pred_test = predict(network, data_test)
        push!(accuracy_test, binary_accuracy(y_pred_test', output_test'))

    end
    return loss, accuracy, accuracy_test
end

# # Feature extraction
# n_feature = 2
# n_element = 120
# stop_train = 100
# n_neural_per_layer = [2]
# X, y = make_blobs(n_element, n_feature; centers=2, as_table=false)
# X = permutedims(X, (2, 1))
# y = [i == 1 ? 0 : 1 for i in reshape(y, (1, n_element))]
# X_train, y_train, X_test, y_test = X[:, 1:stop_train], y[:, 1:stop_train], X[:, stop_train + 1:n_element], y[:, stop_train + 1:n_element]

# # Initialisation of the neural network and fit
# neuron = NeuralNetwork(n_feature, n_neural_per_layer)
# # loss, accuracy, accuracy_test = fit!(neuron, X_train, y_train, X_test, y_test)

# # # Plot the loss and accuracy
# # p1 = plot(loss, title="Loss")
# # p2 = plot(accuracy, title="Accuracy")
# # p3 = plot(accuracy_test, title="Accuracy Test")
# # plot(p1, p2, p3, layout=(1,3))
# # score(neuron, X_train, y_train)
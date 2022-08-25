using LopNeuralNetwork
using Test


@testset "Random Data" begin
    # Feature extraction
    const n_feature = 2
    const n_element = 120
    const stop_train = 100
    const n_neural_per_layer = [2]
    const number_of_layer = length(n_neural_per_layer) + 1
    X, y = make_blobs(n_element, n_feature; centers=2, as_table=false)
    X = permutedims(X, (2, 1))
    X = permutedims(X, (2, 1))
    y = [i == 1 ? 0 : 1 for i in reshape(y, (1, n_element))]
    X_train, y_train, X_test, y_test = X[:, 1:stop_train], y[:, 1:stop_train],
    X[:, stop_train+1:n_element], y[:, stop_train+1:n_element]
    # Initialisation of the neural network and fit
    random_network = NeuralNetwork(n_feature, n_neural_per_layer)
    @test_set "Neurone Initialisation test" begin
        @test size(random_network.W) == number_of_layer
        @test size(random_network.b) == number_of_layer
    end
    @testset "Dimensionnality test" begin
        @test "Dimensionnality test" begin
            for (i, w) in enumerate(random_network.W)
                @test size(w) == (n_neural_per_layer[i + 1] , n_neural_per_layer[i])
            end
            for (i, b) in enumerate(random_network.b)
                @test size(b) == (n_neural_per_layer[i + 1], 1)
            end
        end
    end
    
end

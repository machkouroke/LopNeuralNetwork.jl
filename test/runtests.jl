using LopNeuralNetwork
using Test
using MLJ: make_blobs

include("random_data.jl")
include("diemensionnality_test.jl")
include("plot_test.jl")
@testset "Random Data" begin
    # Data generation
    n_feature = 2
    n_element = 1200
    stop_train = 1000
    n_neural_per_layer = [2]
    X_train, y_train, X_test, y_test, number_of_layer = generate_random_data(n_feature, n_element, stop_train, n_neural_per_layer)
    
    # Initialisation of the neural 
    random_network = NeuralNetwork(n_feature, n_neural_per_layer)
    @testset "Neurone Initialisation test" begin
        @test length(random_network.W) == number_of_layer
        @test length(random_network.b) == number_of_layer
    end
    dimensionnality(random_network)

    # Fitting the neural network
    loss, accuracy, accuracy_test = fit!(random_network, X_train, y_train, X_test, y_test)
    dimensionnality(random_network)

    # Plotting the results
    plot_test(loss, accuracy, accuracy_test)
end

@testset "Cat vs Dog" begin
    include("load_cat_vs_dog.jl")
    
    # Data generation
    trans(X) = permutedims(X, (2, 1))
    X_train, y_train, X_test, y_test = trans.(preprocess_data())
    n_feature = size(X_train)[1]
    n_element = size(X_train)[2]
    n_neural_per_layer = [10, 10, 10, 10, 10]  
    number_of_layer = length(n_neural_per_layer) + 1  
    # Initialisation of the neural 
    cat_dog_network = NeuralNetwork(n_feature, n_neural_per_layer)
    @testset "Neurone Initialisation test" begin
        @test length(cat_dog_network.W) == number_of_layer
        @test length(cat_dog_network.b) == number_of_layer
    end
    dimensionnality(cat_dog_network)
    # Fitting the neural network
    loss, accuracy, accuracy_test = fit!(cat_dog_network, X_train, y_train, X_test, y_test)
    dimensionnality(cat_dog_network)

    # Plotting the results
    plot_test(loss, accuracy, accuracy_test)
end

"Done"

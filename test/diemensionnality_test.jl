dimensionnality(random_network) = @testset "Dimensionnality test" begin
    @testset "Dimensionnality test" begin
        for i in eachindex(random_network.W)
            @test size(random_network.W[i]) == (random_network.number_per_layer[i + 1] , random_network.number_per_layer[i])
        end
        for i in eachindex(random_network.b)
            @test size(random_network.b[i]) == (random_network.number_per_layer[i + 1], 1)
        end
    end
end
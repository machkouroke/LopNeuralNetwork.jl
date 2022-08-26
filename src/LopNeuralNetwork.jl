module LopNeuralNetwork
using Metrics
using Plots: plot
using ProgressBars
using Random: seed!, TaskLocalRNG
export NeuralNetwork, forward_propagation, back_propagation, predict, score, fit!
include("Network/Network.jl")


end

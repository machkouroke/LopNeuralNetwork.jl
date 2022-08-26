using CairoMakie
function plot_test(loss, accuracy, accuracy_test)
    fig = Figure()
    lines(fig[1, 1], loss, color = :red, label="loss")
    axislegend("Legend", position = :ct)
    lines(fig[2, 1], accuracy, color = :green, label="accuracy")
    axislegend("Legend", position = :ct)
    lines(fig[1, 2], accuracy_test, color = :blue, label="accuracy_test")
    axislegend("Legend", position = :ct)
    save("plot_test.png", fig)
end
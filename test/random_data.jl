function generate_random_data(n_feature, n_element, stop_train, n_neural_per_layer)
    number_of_layer = length(n_neural_per_layer) + 1
    X, y = make_blobs(n_element, n_feature; centers=2, as_table=false)
    X = permutedims(X, (2, 1))
    y = [i == 1 ? 0 : 1 for i in reshape(y, (1, n_element))]
    X_train, y_train, X_test, y_test = X[:, 1:stop_train], y[:, 1:stop_train], X[:, stop_train+1:n_element], y[:, stop_train+1:n_element]
    return X_train, y_train, X_test, y_test, number_of_layer
end

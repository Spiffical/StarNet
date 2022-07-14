from torch.nn import Linear, Sequential, LeakyReLU


# Creating NN
def DNN(sizes, num_targets, output_dim):
    # Linear layers
    layers = list()
    layers.append(Linear(num_targets, sizes[0]))
    layers.append(LeakyReLU(0.07))
    for i in range(len(sizes)-1):
        layers.append(Linear(sizes[i], sizes[i+1]))
        layers.append(LeakyReLU(0.07))
    layers.append(Linear(sizes[-1], output_dim))

    model = Sequential(*layers)
    return model
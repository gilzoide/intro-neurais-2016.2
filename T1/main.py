import sys
sys.path.insert(0, '../')

from Models import (SeedsClassificationDataset, Test1RegressionDataset,
                    ClassificationFullyConnectedNeuralNetwork,
                    RegressionFullyConnectedNeuralNetwork)

seedsDataset = SeedsClassificationDataset.load_data('./seeds.txt')
text1Dataset = Test1RegressionDataset.load_data('./teste1.txt')

clasNet = ClassificationFullyConnectedNeuralNetwork(use_2_hidden_layers=False)
clasNet2H = ClassificationFullyConnectedNeuralNetwork(use_2_hidden_layers=True)
regNet = RegressionFullyConnectedNeuralNetwork(use_2_hidden_layers=False)
regNet2H = RegressionFullyConnectedNeuralNetwork(use_2_hidden_layers=True)

for learning_rate in [0.00001, 0.0000001, 0.000000001]:
    for momentum in [0.1, 0.9]:
        for cycles in [1000, 10000]:
            for validation_split in [0.1, 0.3]:
                results = clasNet.train(
                    data=seedsDataset['data'],
                    labels=seedsDataset['labels'],
                    learning_rate=learning_rate,
                    momentum=momentum,
                    validation_split=validation_split,
                    epochs=cycles
                )

                # print as (type, n_hidden_layers, epochs, validation_split,
                # learning_rate, momentum, loss, validation_loss)
                print(', '.join(map(str, ("clas", 1,) + results)))

                results = clasNet2H.train(
                    data=seedsDataset['data'],
                    labels=seedsDataset['labels'],
                    learning_rate=learning_rate,
                    momentum=momentum,
                    validation_split=validation_split,
                    epochs=cycles
                )
                print(', '.join(map(str, ("clas", 2,) + results)))

                results = regNet.train(
                    data=text1Dataset['data'],
                    labels=text1Dataset['labels'],
                    learning_rate=learning_rate,
                    momentum=momentum,
                    validation_split=validation_split,
                    epochs=cycles
                )
                print(', '.join(map(str, ("reg", 1,) + results)))

                results = regNet2H.train(
                    data=text1Dataset['data'],
                    labels=text1Dataset['labels'],
                    learning_rate=learning_rate,
                    momentum=momentum,
                    validation_split=validation_split,
                    epochs=cycles
                )
                print(', '.join(map(str, ("reg", 2,) + results)))

                # reset experiments
                clasNet = ClassificationFullyConnectedNeuralNetwork(
                    use_2_hidden_layers=False)
                clasNet2H = ClassificationFullyConnectedNeuralNetwork(
                    use_2_hidden_layers=True)
                regNet = RegressionFullyConnectedNeuralNetwork(
                    use_2_hidden_layers=False)
                regNet2H = RegressionFullyConnectedNeuralNetwork(
                    use_2_hidden_layers=True)

                # force print
                import sys
                sys.stdout.flush()

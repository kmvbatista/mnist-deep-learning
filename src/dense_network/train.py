from network import Network
import mnist_loader

loader = mnist_loader.Mnist_loader()
training_data, validation_data, test_data = loader.load_data()
model = Network(qtt_neurons_each_layer=[784, 30, 10], training_data=training_data)
model.train_and_test(epochs=2, mini_batch_size=2, learning_rate=3.0, test_data=test_data)
    
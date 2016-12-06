from models import *

iris_data = IrisClassificationDataset.load_data('iris.data')

model = RBF(4, 3, 30)

model.train(iris_data['data'], iris_data['labels'], epochs=1000, lr=0.1)
print "Final Accuracy: {:.6f}".format(model.evaluate(iris_data['data'], iris_data['labels']))
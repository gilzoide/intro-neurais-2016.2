import sys
from models import *

iris_data = IrisClassificationDataset.load_data('iris.data')

model = MLP(4, 4, 3)

results = model.train(
    data=iris_data['data'],
    labels=iris_data['labels'],
    learning_rate=0.001,
    momentum=0.9,
    validation_split=0.1,
    batch_size=100,
    epochs=10000
)

# save losses on a plot
plot(results['loss'], results['val_loss'])
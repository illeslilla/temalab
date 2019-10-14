import tensorflow
import numpy


def forward(x, w1, b1, w2, b2, train=True):
    z = tensorflow.nn.sigmoid(tensorflow.matmul(x, w1) + b1)
    z2 = tensorflow.matmul(z, w2) + b2
    if train:
        return z2
    return tensorflow.nn.sigmoid(z2)


def init_weights(shape):
    return tensorflow.Variable(tensorflow.random_normal(shape, stddev=0.1))


x = numpy.array([[1, 0], [0, 1], [1, 1], [0, 0]])
y = numpy.array([[1], [1], [0], [0]])

# placeholders ????
phX = tensorflow.placeholder(tensorflow.float32, [None, 2])
phY = tensorflow.placeholder(tensorflow.float32, [None, 1])

# weights, 5 hidden nodes
w1 = init_weights([2, 5])
b1 = init_weights([5])
w2 = init_weights([5, 1])
b2 = init_weights([1])

y_hat = forward(phX, w1, b1, w2, b2)
pred = forward(phX, w1, b1, w2, b2, False)

# learning rate
lr = 0.1
# epochs ?????
epochs = 500

# cost???
cost = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(logits = y_hat, labels = phY))

# train
train = tensorflow.train.AdamOptimizer(lr).minimize(cost)

# session, init variables
init = tensorflow.global_variables_initializer()
session = tensorflow.Session()
session.run(init)

# start training
for i in range(epochs):
    session.run(train, feed_dict = {phX: x, phY: y})
    c = session.run(cost, feed_dict = {phX: x, phY: y})
    cost.append(c)
    if i % 100 == 0:
        print(f"Iteration {i}. Cost: {c}.")

print("Training complete.")

# prediction
prediction = session.run(pred, feed_dict = {phX: x})
print("Percentage: ")
print(prediction)
print("Prediction: ")
print(numpy.round(prediction))
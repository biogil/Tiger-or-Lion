from datasets.mnist import load_mnist
from Networks import *
from nntools import *

(x_train, t_train), (x_test, t_test) = load_data()

batch_size = 50
train_size = x_train.shape[0]
epochs = 15
iters_per_epoch = max(int(train_size / batch_size), 1)
iters_num = iters_per_epoch*epochs

optimizer = Adam()

net = SimpleCNN(input_dim=(3,56,56), conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                hidden_size=200, output_size=2, weight_init_std=0.01)

train_loss_list = []
train_acc_list = []
test_acc_list = []

epoch = 0

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = net.gradient(x_batch, t_batch)
    optimizer.update_dict(net.params, grads)

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iters_per_epoch == 0:
        epoch += 1

        x_train_sample, t_train_sample = x_train, t_train
        x_test_sample, t_test_sample = x_test, t_test

        train_acc = net.accuracy(x_train_sample, t_train_sample, batch_size=batch_size)
        test_acc = net.accuracy(x_test_sample, t_test_sample, batch_size=batch_size)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("=== epoch:" + str(epoch) + ", train acc:"+str(train_acc)+", test acc:"+str(test_acc)+" ===")

net.save_params()

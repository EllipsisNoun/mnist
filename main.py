import numpy as np
import torch
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import matplotlib.pyplot as plt
from plot import make_loss_plot, view_classify
from utils import set_seeds, compute_accuracy_from_predictions, predict_by_max_logit


set_seeds(seed=1)  # make learning repeatable by setting seeds

batch_size = 5  # number of examples to process at one time, the training set (50,000) is too big to do them all at once

transform = transforms.Compose(
    [

        transforms.Resize((128,128)),  # resises the image to 64/64 as it has to be a square
        transforms.ToTensor(),  # this maps pixels values from 0 to 255 to the 0 to 1 range and makes a PyTorch tensor
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),  # this then maps the pixel tensor values to the -1 to 1 range
    ]
)


train_set = datasets.FGVCAircraft('./', split='train', transform=transform, download=True)  # load the training set
test_set = datasets.FGVCAircraft('./', split='test', transform=transform, download=True)  # load the test set

# create loaders that will iterate through the train and tests sets a batch at a time
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# get the first batch of training examples so we can examine them
iterator = iter(train_loader)
images, labels = next(iterator)

print(images.shape)  # the shape should be (batch size, 1 channel (monochrome), image height, image width)
print(labels.shape)  # the shape should be (batch size)


# plot the first image image
plt.imshow(images[0].permute(1, 2, 0).numpy())
plt.show()

# plot 5 more on a grid
figure = plt.figure()
num_of_images = 5
for index in range(1, num_of_images + 1):
    plt.subplot(1, 5, index)
    plt.axis('off')
    plt.imshow(images[index-1].permute(1, 2, 0).numpy())
plt.show()


# define the model
output_size = 100  # output size - one for each digit

kernel_size = 5
# 2 convolutional layers and a final linear layer
model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernel_size),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(num_features=8),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(num_features=32),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),
        nn.BatchNorm2d(num_features=128),
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, output_size)
)

print("Number of model parameters = {}".format(sum(p.numel() for p in model.parameters())))

# loss is cross entropy loss
loss_fn = nn.CrossEntropyLoss()

# the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

# train the model
time_start = time()  # set a timer
epochs = 100  # number of training epochs
training_losses = []
for e in range(epochs):
    epoch_losses = []
    for images, labels in train_loader:
        logits = model(images)  # pass the features forward through the model
        loss = loss_fn(logits, labels)   # compute the loss
        epoch_losses.append(loss)

        optimizer.zero_grad()   # clear the gradients
        loss.backward()  # compute the gradients via backpropagation
        optimizer.step()  # update the weights using the gradients

    epoch_loss = np.array(torch.hstack(epoch_losses).detach().numpy()).mean()
    training_losses.append(epoch_loss)
    print("Epoch {} - Loss: {}".format(e, epoch_loss))

# plot the loss vs epoch
make_loss_plot(epochs, training_losses)

# print out the training time
print("\nTraining Time (in minutes) = {}".format((time() - time_start) / 60 ))



# compute accuracy on the test set
predictions = []
labels_test = []
with torch.no_grad():  # don't need gradients for testing
    for images, labels in test_loader:
        labels_test.append(labels)
        with torch.no_grad():
            logits = model(images)
            predictions.append(predict_by_max_logit(logits))  # make prediction on the class that has the highest value

print("Accuracy = {0:0.1f}%".format(compute_accuracy_from_predictions(torch.hstack(predictions), torch.hstack(labels_test)) * 100.0))

# 22.5% accuracy (0054.3 min) Loss:0.05642 - epochs:050 kernel_size:03 resolution:0128/0128
# 21.9% accuracy (0126.8 min) Loss:0.15917 - epochs:050 kernel_size:03 resolution:0256/0256
# 25.8% accuracy (0121.5 min) Loss:0.18387 - epochs:050 kernel_size:05 resolution:0128/0128
# 22.8% accuracy (0112.3 min) Loss:0.33879 - epochs:050 kernel_size:05 resolution:0256/0256
# 25.6% accuracy (1301.7 min) Loss:0.07352 - epochs:100 kernel_size:08 resolution:0512/0512
# __._% accuracy (___.__ min) Loss:_._____ - epochs:100 kernel_size:05 resolution:0128/0128
# __._% accuracy (___.__ min) Loss:_._____ - epochs:___ kernel_size:__ resolution:____/____
# __._% accuracy (___.__ min) Loss:_._____ - epochs:___ kernel_size:__ resolution:____/____
# __._% accuracy (___.__ min) Loss:_._____ - epochs:___ kernel_size:__ resolution:____/____


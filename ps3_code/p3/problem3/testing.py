import torch
import numpy as np
from torch.utils.data import DataLoader


def test(dataset, model, batch_size):
    # initialize a DataLoader on the dataset with the appropriate batch
    # size and shuffling enabled.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    correct_count = 0
    total = 0
    for images, labels in data_loader:
        total += labels.size(0)
        output = model.classify(images.cuda())

        # Calculate accuracy
        _, predictions = torch.max(output, 1)

        # calculate the number of correctly classified inputs.
        num_correct = (predictions == labels.cuda()).sum().item()

        correct_count += num_correct

    # calculate the float accuracy for the whole dataset.
    accuracy = 100 * correct_count / total
    print("Testing Accuracy: %.3f"%(accuracy))

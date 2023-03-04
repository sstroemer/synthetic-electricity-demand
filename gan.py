import math
import numpy as np

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer1 = nn.Linear(int(input_length), int(input_length)*3)
        self.dense_layer2 = nn.Linear(int(input_length)*3, int(24))
        self.activation1 = nn.ELU()
        self.activation2 = nn.Tanh()

    def forward(self, x):
        return self.activation2(self.dense_layer2(self.activation1(self.dense_layer1(x))))
    

class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense1 = nn.Linear(int(input_length), int(input_length))
        self.dense2 = nn.Linear(int(input_length), 1)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Sigmoid()

    def forward(self, x):
        return self.activation2(self.dense2(self.activation1(self.dense1(x))))


from typing import List, Tuple
import numpy as np

def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):
    input_length = 24 + 7 + 1

    # Models
    generator = Generator(2 + 7 + 1)
    discriminator = Discriminator(24 + 7 + 1)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        samples = [random_sample(weeks) for i in range(batch_size)]
        input = torch.tensor([it[0] for it in samples]).to(torch.float32)
        real_output = torch.tensor([it[1] for it in samples]).to(torch.float32)
        conditioning = input[:, -8:]

        generated_data = torch.cat((generator(input), conditioning), dim=1)

        # Generate examples of even real data
        # true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
        # true_labels = torch.tensor(true_labels).float().reshape(16, 1)
        # true_data = torch.tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, torch.ones(batch_size, 1))
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(torch.cat((real_output, conditioning), dim=1))
        true_discriminator_loss = loss(true_discriminator_out, torch.ones(batch_size, 1))

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size, 1))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        if i % 50 == 0:
            print("G: ", generator_loss)
            print("D: ", discriminator_loss)
            print(generated_data[0,:])


if __name__ == "__main__":
    train()

import torch
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, criterion, optimizer, patience):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.patience = patience

        self.trainingLosses = []
        self.validationLosses = []

    def train(self, train_loader, val_loader, num_epochs):
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the lowest validation loss
        lowest_val_loss = np.inf
        # to count the epochs with no improvement in validation loss
        stagnant_epochs = 0

        for epoch in range(num_epochs):
            for i, (features, labels) in enumerate(train_loader):
                outputs = self.model(features.float())
                loss = self.criterion(outputs, labels.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validate the model
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0
                for features, labels in val_loader:
                    outputs = self.model(features.float())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_loss += self.criterion(outputs, labels.long()).item()

                avg_val_loss = val_loss / len(val_loader)
                valid_losses.append(avg_val_loss)

                self.trainingLosses.append(loss.item())
                self.validationLosses.append(avg_val_loss)

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {100 * correct / total}%')

                # Early stopping
                if avg_val_loss < lowest_val_loss:
                    lowest_val_loss = avg_val_loss
                    stagnant_epochs = 0
                else:
                    stagnant_epochs += 1
                    if stagnant_epochs == self.patience:
                        print("Early stopping due to no improvement in validation loss...")
                        return
    
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.trainingLosses, label='Training loss')
        plt.plot(self.validationLosses, label='Validation loss')
        plt.title('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
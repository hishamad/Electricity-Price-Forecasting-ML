import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class EnergyPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(EnergyPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, train_data, validation_data, labels, validation_labels, epochs=100, batch_size=32, learning_rate=0.001, patience=10):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_mse = []
        val_mse = []
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            epoch_train_loss = 0
            for i in range(0, train_data.size(0), batch_size):
                batch_data, batch_labels = train_data[i:i+batch_size], labels[i:i+batch_size]
                optimizer.zero_grad()
                
                outputs = self(batch_data)

                loss = criterion(outputs, batch_labels)
                epoch_train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_train_loss = epoch_train_loss / (train_data.size(0) // batch_size)
            train_mse.append(avg_train_loss)
            
            # Validation step
            self.eval()
            with torch.no_grad():
                val_outputs = self(validation_data)
                val_loss = criterion(val_outputs, validation_labels)
                val_mse.append(val_loss.item())
                
            print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss.item():.4f}")

            # Early stopping check
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        plt.figure(figsize=(10, 6))
        plt.plot(train_mse, label="Training MSE", color='blue')
        plt.plot(val_mse, label="Validation MSE", color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
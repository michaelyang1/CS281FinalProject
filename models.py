import torch
import torch.nn as nn
import torch.optim as optimize
from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.model_selection import train_test_split

def split_train_test(X, y, train_ratio=0.8):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
    
    # convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # create dataloaders for training and testing
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, X_train, y_train, X_test, y_test

def train_baseline_regression_model(X, y):
    # hyperparameters
    train_loader, test_loader, X_train, y_train, X_test, y_test = split_train_test(X, y)
    
    # create model with appropriate input and output sizes
    model = nn.Sequential(
        # hidden layer 1
        nn.Linear(X_train.shape[0], 32),
        nn.ReLU(),
        
        # hidden layer 2
        nn.Linear(32, 32),
        nn.ReLU(),
        
        # output layer
        nn.Linear(32, 1)
    )
    
    # hyperparameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    
    # train model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            # forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # print progress
        if (epoch + 1) % 10 == 0: # epochs are zero indexed 
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # evaluate model on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = nn.MSELoss()(test_predictions, y_test)
        print(f"Final Test Loss: {test_loss:.4f}")
    
    return model, test_predictions

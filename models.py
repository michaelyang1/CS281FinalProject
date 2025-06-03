import torch
import torch.nn as nn
import torch.optim as optimize
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

def split_train_test(X, y, train_ratio=0.8):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
    gender_train = y_train[['Gender']]
    gender_test = y_test[['Gender']]
    y_train = y_train.drop(columns=['Gender'])
    y_test = y_test.drop(columns=['Gender'])
    
    # convert to torch tensors using .values for pandas DataFrames
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)
    gender_train = torch.tensor(gender_train.values, dtype=torch.float32)
    gender_test = torch.tensor(gender_test.values, dtype=torch.float32)
    print(X_train.shape, y_train.shape, gender_train.shape)
    
    # create dataloaders for training and testing
    train_dataset = TensorDataset(X_train, y_train, gender_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, X_train, y_train, gender_train, X_test, y_test, gender_test

def train_regression_model(X, y, eo=False):
    # hyperparameters
    train_loader, test_loader, X_train, y_train, gender_train, X_test, y_test, gender_test = split_train_test(X, y)
    
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
    if eo:
        criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    num_epochs = 100
    
    # train model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y, batch_gender in train_loader:
            # forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            lambda_fair = 1.0
            if eo:
                # fairness penalty: group-wise mean absolute error
                female_mask = (batch_gender == 0).squeeze()
                male_mask = (batch_gender == 1).squeeze()
                
                male_error = torch.abs(outputs[male_mask] - batch_y[male_mask])
                female_error = torch.abs(outputs[female_mask] - batch_y[female_mask])
                
                # prevent division by zero
                if male_error.numel() > 0 and female_error.numel() > 0:
                    group_gap = torch.abs(male_error.mean() - female_error.mean())
                    loss += lambda_fair * group_gap
            
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
        
        # calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        # update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # print progress
        if (epoch + 1) % 10 == 0:  # epochs are zero indexed 
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # evaluate model on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        criterion = nn.MSELoss()
        if eo:
            criterion = nn.L1Loss()
        overall_loss = criterion(test_predictions, y_test)
        
        gender_test = gender_test.squeeze()
        female_mask = gender_test == 0
        male_mask = gender_test == 1
        
        # group specific predictions and targets
        male_preds = test_predictions[male_mask]
        male_targets = y_test[male_mask]

        female_preds = test_predictions[female_mask]
        female_targets = y_test[female_mask]

        # group specific losses
        male_loss = criterion(male_preds, male_targets) if male_preds.numel() > 0 else torch.tensor(0.0)
        female_loss = criterion(female_preds, female_targets) if female_preds.numel() > 0 else torch.tensor(0.0)

        # Print results
        print(f"Overall Test Loss (MSE): {overall_loss.item():.4f}")
        print(f"Male Test Loss (MSE): {male_loss.item():.4f}")
        print(f"Female Test Loss (MSE): {female_loss.item():.4f}")
    
    return model, test_predictions

def train_classification_model(X, y):
    pass

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
    # Z-score normalization
    X = (X - X.mean()) / X.std()
    
    train_loader, test_loader, X_train, y_train, gender_train, X_test, y_test, gender_test = split_train_test(X, y)

    # create model with appropriate input and output sizes
    model = nn.Sequential(
        # hidden layer 1
        nn.Linear(X_train.shape[1], 32),
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
    
    # early stopping initialization
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
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
        
        # update learning rate based on val loss
        scheduler.step(avg_val_loss)
        
        # perform early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict() # save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                # restore best model
                model.load_state_dict(best_model_state)
                break
        
        # print progress
        if (epoch + 1) % 10 == 0:  # epochs are zero indexed 
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # evaluate model on test set
    overall_loss, male_loss, female_loss = None, None, None
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
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
        overall_loss = overall_loss.item()
        male_loss = male_loss.item()
        female_loss = female_loss.item()
        print(f"Overall Test Loss (L1): {overall_loss:.4f}")
        print(f"Male Test Loss (L1): {male_loss:.4f}")
        print(f"Female Test Loss (L1): {female_loss:.4f}")
    
    return overall_loss, male_loss, female_loss

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), # hidden layer 1
            nn.ReLU(),
            nn.Linear(32, 32), # hidden layer 2
            nn.ReLU(),
            nn.Linear(32, 16), # z is 16-dimensional
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class PHQPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 1)

    def forward(self, z):
        return self.fc(z)

class GenderAdversary(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() # gender output is a probability -> example: 0.56
        )

    def forward(self, z):
        return self.net(z)

def train_adversarial_model(X, y):
    # Z-score normalization
    X = (X - X.mean()) / X.std()
    
    train_loader, test_loader, X_train, y_train, gender_train, X_test, y_test, gender_test = split_train_test(X, y)
    encoder = Encoder(X_train.shape[1])
    phq_predictor = PHQPredictor()
    gender_adversary = GenderAdversary()

    # hyperparameters
    phq_criterion = nn.MSELoss()
    gender_criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(phq_predictor.parameters()), lr=0.001)
    adversarial_optimizer = torch.optim.Adam(gender_adversary.parameters(), lr=0.001)
    
    # early stopping initialization
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    num_epochs = 100
    
    # train model
    for epoch in range(num_epochs):
        # train
        encoder.train()
        phq_predictor.train()
        gender_adversary.train()
        
        train_loss = 0
        lambda_adversary = 1.0
        
        for batch_x, batch_y, batch_gender in train_loader:
            # train gender adversary
            z = encoder(batch_x).detach() # freeze to train adversary only
            adversary_pred = gender_adversary(z)
            adversary_loss = gender_criterion(adversary_pred, batch_gender)
            adversarial_optimizer.zero_grad()
            adversary_loss.backward()
            adversarial_optimizer.step()

            # train encoder and phq predictor
            z = encoder(batch_x)
            phq_pred = phq_predictor(z)
            phq_loss = phq_criterion(phq_pred, batch_y)
            adversary_pred = gender_adversary(z)
            adversary_loss = gender_criterion(adversary_pred, batch_gender)
            # maximize adversary loss -> minimize phq loss
            # encoder is trained to retain PHQ but not gender predictive ability
            total_loss = phq_loss - lambda_adversary * adversary_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        
        # evaluate
        encoder.eval()
        phq_predictor.eval()
        gender_adversary.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                z = encoder(batch_x)
                phq_pred = phq_predictor(z)
                loss = phq_criterion(phq_pred, batch_y)
                val_loss += loss.item()
                
        # calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        # update learning rate based on val loss
        scheduler.step(avg_val_loss)

        # perform early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_encoder = encoder.state_dict()
            best_predictor = phq_predictor.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                encoder.load_state_dict(best_encoder)
                phq_predictor.load_state_dict(best_predictor)
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
        
    # evaluate model on test set
    encoder.eval()
    phq_predictor.eval()
    with torch.no_grad():
        z_test = encoder(X_test)
        y_pred = phq_predictor(z_test)

        criterion = nn.L1Loss()
        overall_loss = criterion(y_pred, y_test)

        gender_test = gender_test.squeeze()
        female_mask = gender_test == 0
        male_mask = gender_test == 1

        female_loss = criterion(y_pred[female_mask], y_test[female_mask]) if female_mask.any() else torch.tensor(0.0)
        male_loss = criterion(y_pred[male_mask], y_test[male_mask]) if male_mask.any() else torch.tensor(0.0)

        print(f"Overall Test Loss (L1): {overall_loss.item():.4f}")
        print(f"Male Test Loss (L1): {male_loss.item():.4f}")
        print(f"Female Test Loss (L1): {female_loss.item():.4f}")

    return overall_loss.item(), male_loss.item(), female_loss.item()

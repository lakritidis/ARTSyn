import torch.nn as nn

class ctdClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import average_precision_score


# =========================================================
# Classifier Architecture
# =========================================================

class TabularClassifier(nn.Module):
    """
    TabularClassifier
    """
    # This architecture is for ctdGAN_cls2 and ctdGAN_cls2clu
    def __init__(self, input_dim, num_classes, hidden_dims=(128, 256, 256, 128), dropout=0.2, temperature=1.0):
        super().__init__()

        self.temperature = temperature

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[3], num_classes)
        )
    '''
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 256), dropout=0.2, temperature=1.0):
        super().__init__()

        self.temperature = temperature

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], num_classes)
        )
    '''
    def forward(self, x):
        logits = self.net(x)
        # temperature scaling
        logits = logits / self.temperature

        return logits

# =========================================================
# Training Function
# =========================================================

def train_classifier(x_train, y_train, x_val, y_val, input_dim, num_classes, hidden_dims=(128, 256, 256, 128),
                     batch_size=64, epochs=30, lr=1e-3, weight_decay=1e-4, patience=5, device="cuda"):

    #device = torch.device(device if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------
    # tensors
    # -----------------------------------------------------

    #x_train = torch.tensor(x_train, dtype=torch.float32)
    #y_train = torch.tensor(y_train, dtype=torch.long)

    #x_val = torch.tensor(x_val, dtype=torch.float32)
    #y_val = torch.tensor(y_val, dtype=torch.long)

    # dataloaders
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False )

    # class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # model
    model = TabularClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims, dropout=0.2, temperature=1.0).to(device)

    # optimizer + loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # early stopping
    best_model = None
    best_val_loss = float("inf")
    patience_counter = 0

    # training loop
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()

        val_loss = 0.0

        all_probs = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()

                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

                all_targets.append(yb.cpu().numpy())

        val_loss /= len(val_loader)

        #all_probs = np.concatenate(all_probs)
        #all_targets = np.concatenate(all_targets)

        # PR-AUC
        #if num_classes == 2:
        #    pr_auc = average_precision_score(all_targets, all_probs[:, 1])
        #else:
        #    pr_auc = average_precision_score(all_targets, all_probs, average="macro")

        #print(f"Epoch {epoch+1:03d} | " f"Train Loss: {train_loss:.4f} | " f"Val Loss: {val_loss:.4f} | " f"PR-AUC: {pr_auc:.4f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            # print("Early stopping triggered at epoch ", epoch)
            break

    # load best model
    model.load_state_dict(best_model)

    return model

#Example usage:
#model = train_classifier(X_train, y_train, X_val, y_val,
#                         input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)),
#                        batch_size=64, epochs=30, lr=1e-3)
'''
Then inside your GAN training:

# fake samples from generator
x_fake = G(z, y_cond)

# classifier prediction
logits = classifier(x_fake)

# classification loss
loss_cls = F.cross_entropy(logits, y_cond)

# generator adversarial loss
loss_adv = ...

# total generator loss
loss_G = loss_adv + lambda_cls * loss_cls

A few important recommendations:

Freeze classifier initially
for p in classifier.parameters():
    p.requires_grad = False
Use .eval() mode during GAN training
classifier.eval()
Start with:
lambda_cls = 0.05
Increase slowly during training.
Standardize numerical features before training the classifier.

For tabular GANs, this matters enormously.
'''
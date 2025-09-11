import torch
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import sys
import copy
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('..','..','models','transformer')))
sys.path.append(os.path.abspath(os.path.join('..','..','models','temporal_cnn')))

from transformer_model import TimeSeriesTransformer
from temporal_cnn_model import TemporalCNN

def train(
    model,train_loader,test_loader,val_loader,
    epochs,device,save_path,
    plot_path,scaling_factors,y_name,patience=5,
    lr=1e-4,optimizer=None,
    criterion=None
):
    if criterion is None:
        criterion = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test_loss = float('inf')
    best_test_r2 = float('-inf')
    best_test_targets = None
    best_test_preds = None
    train_losses = []
    test_losses = []
    train_r2s = []
    test_r2s = []
    # load the scaling factors for our model
    counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        for i, (X_batch,y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            X_batch,y_batch = X_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            train_preds.append(outputs.detach().cpu())
            train_targets.append(y_batch.detach().cpu())
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        epoch_train_r2 = r2_score_torch(train_targets, train_preds)
        train_r2s.append(epoch_train_r2)
        # Validation
        model.eval()
        test_loss = 0.0
        training_loss_on_eval = 0.0
        test_preds, test_targets = [], []
        with torch.no_grad():
            # let's see what our training loss is, just to see how dropout is
            # affecting performance
            for X_train,y_train in train_loader:
                X_train,y_train = X_train.to(device),y_train.to(device)
                train_outputs = model(X_train)
                loss = criterion(train_outputs.squeeze(), y_train)
                training_loss_on_eval += loss.item() * X_train.size(0)
            # now get our real validation loss
            for X_test,y_test in test_loader:
                X_test,y_test = X_test.to(device),y_test.to(device)
                test_outputs = model(X_test)
                loss = criterion(test_outputs.squeeze(), y_test)
                test_loss += loss.item() * X_test.size(0)
                test_preds.append(test_outputs.cpu())
                test_targets.append(y_test.cpu())
        epoch_train_loss_on_eval = training_loss_on_eval / len(train_loader.dataset)
        epoch_train_r2_on_eval = r2_score_torch(train_targets, train_preds)
        epoch_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        test_preds = torch.cat(test_preds)
        test_targets = torch.cat(test_targets)
        epoch_test_r2 = r2_score_torch(test_targets, test_preds)
        test_r2s.append(epoch_test_r2)
        print(
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Train R2: {epoch_train_r2:.4f}, "
            f"Test Loss: {epoch_test_loss:.4f}, "
            f"Test R2: {epoch_test_r2:.4f}"
        )
        # Check for early stopping
        # also save the model if validation loss improves
        if epoch_test_loss < best_test_loss:
            best_model = copy.deepcopy(model)
            best_test_loss = epoch_test_loss
            torch.save(
                model.state_dict(),
                os.path.join(save_path, 'best_model.pth')
            )
            # save the best validation R2
            best_test_r2 = epoch_test_r2
            # convert the preds and targets back to their original scale
            test_preds_denormalized = denormalize(
                test_preds, y_name, scaling_factors
            )
            test_targets_denormalized = denormalize(
                test_targets, y_name, scaling_factors
            )
            best_test_rmse = calc_rmse(
                test_targets_denormalized,
                test_preds_denormalized
            )
            best_test_bias = calc_bias(
                test_targets_denormalized,
                test_preds_denormalized
            )
            best_test_targets = copy.deepcopy(test_targets)
            best_test_preds = copy.deepcopy(test_preds)
            best_test_targets_denormalized = copy.deepcopy(
                test_targets_denormalized
            )
            best_test_preds_denormalized = copy.deepcopy(
                test_preds_denormalized
            )
            print(f"Saved best model at epoch {epoch+1}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        print(f"Counter: {counter}/{patience}")
    # plot losses and R2 after training
    fig,ax1 = plt.subplots(figsize=(10,6))
    # plot losses
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(train_losses, label='Train Loss', color='tab:blue')
    ax1.plot(test_losses, label='Test Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # plot R2 scores
    ax2 = ax1.twinx()
    ax2.set_ylabel('R2 Score', color='tab:green')
    ax2.plot(train_r2s, label='Train R2', color='tab:green', linestyle='--')
    ax2.plot(test_r2s, label='Test R2', color='tab:red', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    lines_1,labels_1 = ax1.get_legend_handles_labels()
    lines_2,labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    plt.savefig(
        os.path.join(plot_path, 'training_plot.png'),
        bbox_inches='tight'
    )
    # plot testing preds vs. target for our best model
    plot_obs_pred(
        best_test_targets_denormalized,
        best_test_preds_denormalized,
        best_test_r2,
        best_test_loss,
        best_test_rmse,
        best_test_bias,
        os.path.join(plot_path, 'best_test_preds_vs_targets.png')
    )
    # evaluate on the validation set 
    # now get our real validation loss
    best_model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_val,y_val in val_loader:
            X_val,y_val = X_val.to(device),y_val.to(device)
            val_outputs = best_model(X_val)
            loss = criterion(val_outputs.squeeze(), y_val)
            val_loss += loss.item() * X_val.size(0)
            val_preds.append(val_outputs.cpu())
            val_targets.append(y_val.cpu())
        total_val_loss = val_loss / len(val_loader.dataset)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_r2 = r2_score_torch(val_targets, val_preds)
        val_preds_denormalized = denormalize(
            val_preds, y_name, scaling_factors
        )
        val_targets_denormalized = denormalize(
            val_targets, y_name, scaling_factors
        )
        val_rmse = calc_rmse(
            val_targets_denormalized,
            val_preds_denormalized
        )
        val_bias = calc_bias(
            val_targets_denormalized,
            val_preds_denormalized
        )
    # plot validation preds vs. target for our best model
    plot_obs_pred(
        val_targets_denormalized,
        val_preds_denormalized,
        val_r2,
        total_val_loss,
        val_rmse,
        val_bias,
        os.path.join(plot_path, 'best_val_preds_vs_targets.png')
    )

def plot_obs_pred(
    targets, preds, r2, loss, rmse, bias,
    fname
):
    # plot validation preds vs. target for our best model
    plt.figure(figsize=(10,6))
    plt.scatter(
        targets,
        preds,
        alpha=0.5
    )
    plt.plot(
        [
            targets.min(),
            targets.max()
        ],
        [
            targets.min(),
            targets.max()
        ],
        color='red', linestyle='--', label='One-to-One Line'
    )
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.text(
        0.05, 0.95,
        f'Best Test R2: {r2:.4f}',
        transform=plt.gca().transAxes,
        fontsize=12, verticalalignment='top'
    )
    plt.text(
        0.05, 0.90,
        f'Best Test Loss: {loss:.4f}',
        transform=plt.gca().transAxes,
        fontsize=12, verticalalignment='top'
    )
    plt.text(
        0.05, 0.85,
        f'Best Test RMSE: {rmse:.4f}',
        transform=plt.gca().transAxes,
        fontsize=12, verticalalignment='top'
    )
    plt.text(
        0.05, 0.80,
        f'Best Test Bias: {bias:.4f}',
        transform=plt.gca().transAxes,
        fontsize=12, verticalalignment='top'
    )
    plt.legend()
    plt.savefig(
        os.path.join(fname),
        bbox_inches='tight'
    )

def r2_score_torch(y_true, y_pred):
    # Flatten y_pred if needed
    if y_pred.dim() > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)  # remove the singleton dim
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()  # returns a Python float

def calc_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE) between true and predicted values.
    """
    if y_pred.dim() > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)  # remove the singleton dim
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return rmse.item()  # returns a Python float

def calc_bias(y_true, y_pred):
    """
    Calculate the bias between true and predicted values.
    Bias is defined as the mean of the differences between true and predicted values.
    """
    if y_pred.dim() > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)  # remove the singleton dim
    bias = torch.mean(y_true - y_pred)
    return bias.item()  # returns a Python float

def denormalize(tensor, feature_name, stats_df):
    row = stats_df.loc[feature_name]
    mean = row['mean']
    std = row['std']
    return tensor * std + mean


if __name__ == "__main__":
    # fill in here
    # things that we need to autofill the names to load the datasets
    model_type = 'transformer'
    split_type = 'spatial'
    start_date = '20030101'
    end_date = '20231231'
    y_name = 'Insitu'
    x_name = 'ModisDaymetStaticLatlon'
    obs_name = 'lfmc'
    # dataset that we are using for training and testing
    gen_dataset_fname = (
        '/scratch/users/trobinet/long_lfmc/'
        'trent_datasets/lfmc_model/data/splits/'
        '{model_type}_{split_type}_{dataset_type}_'
        '{start_date}_{end_date}_y_{y_name}_x_{x_name}.npy'
    ).format(
        model_type=model_type,
        split_type=split_type,
        dataset_type='{dataset_type}',
        start_date=start_date,
        end_date=end_date,
        y_name=y_name,
        x_name=x_name
    )
    train_dataset_fname = gen_dataset_fname.format(
        dataset_type='train'
    )
    test_dataset_fname = gen_dataset_fname.format(
        dataset_type='test'
    )
    val_dataset_fname = gen_dataset_fname.format(
        dataset_type='val'
    )
    # filename for the scaling factors
    scale_df_fname = (
        '/scratch/users/trobinet/long_lfmc/'
        'trent_datasets/lfmc_model/data/norm_df/'
        '{model_type}_{split_type}_'
        '{start_date}_{end_date}_y_{y_name}_x_{x_name}.csv'
    ).format(
        model_type=model_type,
        split_type=split_type,
        start_date=start_date,
        end_date=end_date,
        y_name=y_name,
        x_name=x_name
    )
    # what type of model are we training here?
    # epochs to train for
    epochs = 200
    # patience (early stopping, number of epochs with no improvement)
    patience = 10
    # batch size for training
    batch_size = 64
    # transformer hyperparameters
    d_model = 32
    nhead = 2
    num_layers = 2
    dim_feedforward = d_model * 2
    dropout = 0.2
    lr = 1e-4
    # temporal CNN hyperparameters
    #d_model = 32
    #num_layers = 3
    #kernel_size = 5
    #dilation_base = 2
    #dropout = 0.2
    #lr = 1e-4
    # format names based on user input
    input_name = (
        '_'.join(
            train_dataset_fname.split('/')[-1].split('.')[0].split('_')[0:2]
        ) + '_' + '_'.join(
            train_dataset_fname.split('/')[-1].split('.')[0].split('_')[3:]
        )
    )
    if model_type == 'transformer':
        specific_name = (
            f"{input_name}_d{d_model}_n{nhead}_l{num_layers}_"
            f"df{dim_feedforward}_dr{dropout}_bs{batch_size}"
            f"_lr{lr}_ep{epochs}"
        )
    elif model_type == 'temporal_cnn':
        specific_name = (
            f"{input_name}_d{d_model}_l{num_layers}_"
            f"ks{kernel_size}_db{dilation_base}_dr{dropout}_bs{batch_size}"
            f"_lr{lr}_ep{epochs}"
        )
    # path to save checkpoints and model
    checkpoint_path = (
        '/scratch/users/trobinet/long_lfmc/'
        'trent_datasets/lfmc_model/checkpoints/'
        f'{specific_name}/'
    )
    # path to save plots
    plot_path = (
        '/scratch/users/trobinet/long_lfmc/'
        'trent_datasets/lfmc_model/outputs/viz/'
        f'{specific_name}/'
    )
    print('Starting training process')
    # load dataset
    print(f"Loading train dataset from {train_dataset_fname}")
    train_data = torch.load(train_dataset_fname)
    train_X = train_data['X']
    train_y = train_data['y']
    print(f"Loading test dataset from {test_dataset_fname}")
    test_data = torch.load(test_dataset_fname)
    test_X = test_data['X']
    test_y = test_data['y']
    print(f"Loading validation dataset from {val_dataset_fname}")
    val_data = torch.load(val_dataset_fname)
    val_X = val_data['X']
    val_y = val_data['y']
    print(
        f"Train dataset loaded with shape {train_X.shape} for X and"
        f"{train_y.shape} for y"
    )
    print(
        f"Test dataset loaded with shape {test_X.shape} for X and"
        f"{test_y.shape} for y"
    )
    print(
        f"Validation dataset loaded with shape {val_X.shape} for X and"
        f"{val_y.shape} for y"
    )
    print('loading scaling factors from {}'.format(scale_df_fname))
    # load the scaling factors
    scaling_factors = pd.read_csv(scale_df_fname)
    scaling_factors.set_index('feature', inplace=True)
    # set random seed for reproducibility
    torch.manual_seed(0)
    train_dataset = TensorDataset(
        train_X, train_y
    )
    test_dataset = TensorDataset(
        test_X, test_y
    )
    val_dataset = TensorDataset(
        val_X, val_y
    )
    # compute some statistics on our train and test; might explain these wildly
    # different values in training and testing?
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    # instantiate model
    print("Instantiating model")
    #if not torch.cuda.is_available():
    #    raise RuntimeError("CUDA is not available. Please run on a GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if model_type == 'transformer':
        model = TimeSeriesTransformer(
            input_dim=train_X.shape[2],
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)
    elif model_type == 'temporal_cnn':
        model = TemporalCNN(
            input_dim=train_X.shape[2],
            d_model=d_model,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout
        ).to(device)
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
    # set up directory to save checkpoints and model
    os.makedirs(checkpoint_path, exist_ok=True)
    # set up directory to save plots
    os.makedirs(plot_path, exist_ok=True)
    print(f"Model will be saved to {checkpoint_path}")
    print(f"Plots will be saved to {plot_path}")
    print(f"Training on {device}")
    # train model
    train(
        model,
        train_loader,
        test_loader,
        val_loader,
        epochs,
        device,
        checkpoint_path,
        plot_path,
        scaling_factors,
        obs_name,
        patience=patience,
        lr=lr
    )



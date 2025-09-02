import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import h5py


    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    from tqdm import tqdm
    return (
        DataLoader,
        Dataset,
        MinMaxScaler,
        StandardScaler,
        h5py,
        mean_absolute_error,
        mean_squared_error,
        nn,
        np,
        optim,
        pd,
        plt,
        torch,
        tqdm,
    )


@app.cell
def _(MinMaxScaler, StandardScaler, h5py, np, pd):
    class NCMAPSSPreprocessor:
        def __init__(self, scaler_type="minmax"):
            """
            CMAPSS data preprocessor for LSTM models using H5 files

            Args:
                scaler_type: 'minmax' or 'standard' scaling
            """
            self.scaler_type = scaler_type
            self.feature_scaler = None

        def load_raw_h5_data(self, filename):
            """Load raw CMAPSS dataset from H5 file (your original format)"""
            with h5py.File(filename, "r") as hdf:
                # Development set
                W_dev = np.array(hdf.get("W_dev"))  # W
                X_s_dev = np.array(hdf.get("X_s_dev"))  # X_s
                X_v_dev = np.array(hdf.get("X_v_dev"))  # X_v
                T_dev = np.array(hdf.get("T_dev"))  # T
                Y_dev = np.array(hdf.get("Y_dev"))  # RUL
                A_dev = np.array(hdf.get("A_dev"))  # Auxiliary

                # Test set
                W_test = np.array(hdf.get("W_test"))  # W
                X_s_test = np.array(hdf.get("X_s_test"))  # X_s
                X_v_test = np.array(hdf.get("X_v_test"))  # X_v
                T_test = np.array(hdf.get("T_test"))  # T
                Y_test = np.array(hdf.get("Y_test"))  # RUL
                A_test = np.array(hdf.get("A_test"))  # Auxiliary

                # Variable names
                W_var = np.array(hdf.get("W_var"))
                X_s_var = np.array(hdf.get("X_s_var"))
                X_v_var = np.array(hdf.get("X_v_var"))
                T_var = np.array(hdf.get("T_var"))
                A_var = np.array(hdf.get("A_var"))

                # Convert to string lists
                W_var = list(np.array(W_var, dtype="U20"))
                X_s_var = list(np.array(X_s_var, dtype="U20"))
                X_v_var = list(np.array(X_v_var, dtype="U20"))
                T_var = list(np.array(T_var, dtype="U20"))
                A_var = list(np.array(A_var, dtype="U20"))

            # Create DataFrames
            X_train = pd.DataFrame(
                data=np.hstack((A_dev, W_dev, X_s_dev)),
                columns=(A_var + W_var + X_s_var),
            )
            y_train = pd.DataFrame(data=Y_dev, columns=["RUL"])

            X_test = pd.DataFrame(
                data=np.hstack((A_test, W_test, X_s_test)),
                columns=(A_var + W_var + X_s_var),
            )
            y_test = pd.DataFrame(data=Y_test, columns=["RUL"])

            return X_train, y_train, X_test, y_test

        def load_processed_h5_data(self, filename):
            """Load preprocessed data from H5 file"""
            with h5py.File(filename, "r") as f:
                X_train = pd.DataFrame(np.array(f["X_train"]))
                y_train = pd.DataFrame(np.array(f["y_train"]))
                X_test = pd.DataFrame(np.array(f["X_test"]))
                y_test = pd.DataFrame(np.array(f["y_test"]))

                # Get column names
                X_columns = f["X_columns"][:].astype(str)
                y_columns = f["y_columns"][:].astype(str)

                X_train.columns = X_columns
                y_train.columns = y_columns
                X_test.columns = X_columns
                y_test.columns = y_columns

            return X_train, y_train, X_test, y_test

        def preprocess_and_save(self, raw_filename, processed_filename):
            """Complete preprocessing pipeline from raw to processed H5 file"""
            print("Loading raw data...")
            X_train, y_train, X_test, y_test = self.load_raw_h5_data(raw_filename)

            print("Aggregating by (unit, cycle)...")
            # Temporarily add the RUL column to X
            X_train["RUL"] = y_train["RUL"].values
            X_test["RUL"] = y_test["RUL"].values

            # Get feature columns (exclude unit, cycle and RUL)
            features = [c for c in X_train.columns if c not in ["unit", "cycle", "RUL"]]

            agg_funcs = {f: ["mean", "std"] for f in features}

            def aggregate_cycles(X):
                # Group features
                X_agg = (
                    X.groupby(["unit", "cycle"])
                    .agg(agg_funcs)
                )
                # Flatten multi-level cols
                X_agg.columns = ["_".join(col).strip() for col in X_agg.columns.values]
                X_agg = X_agg.reset_index()

                # Take one RUL per cycle and drop unit and cycle columns
                y_agg = (
                    X.groupby(["unit", "cycle"])["RUL"]
                    .first()
                    .reset_index()
                    .drop(["unit", "cycle"], axis='columns')
                )
                return X_agg, y_agg

            X_train, y_train = aggregate_cycles(X_train)
            X_test, y_test = aggregate_cycles(X_test)

            print("Scaling features...")

            # Convert column names to Python str
            X_train.columns = X_train.columns.astype(str)
            X_test.columns  = X_test.columns.astype(str)
            y_train.columns = y_train.columns.astype(str)
            y_test.columns  = y_test.columns.astype(str)

            # Get feature columns (exclude unit and cycle)
            features = [c for c in X_train.columns if c not in ["unit", "cycle"]]

            # Initialize and fit scaler on training data
            if self.scaler_type == "minmax":
                self.feature_scaler = MinMaxScaler()
                self.target_scaler = MinMaxScaler()
            else:
                self.feature_scaler = StandardScaler()
                self.target_scaler = StandardScaler()

            # Scale features
            X_train[features] = self.feature_scaler.fit_transform(X_train[features])
            X_test[features]  = self.feature_scaler.transform(X_test[features])

            # Scale RUL
            y_train["RUL"] = self.target_scaler.fit_transform(y_train[["RUL"]])
            y_test["RUL"]  = self.target_scaler.transform(y_test[["RUL"]])

            print("Saving processed data...")
            # Save processed data
            with h5py.File(processed_filename, "w") as f:
                f.create_dataset("X_train", data=X_train.values, compression="gzip")
                f.create_dataset("y_train", data=y_train.values, compression="gzip")
                f.create_dataset("X_test", data=X_test.values, compression="gzip")
                f.create_dataset("y_test", data=y_test.values, compression="gzip")

                f.create_dataset(
                    "X_columns",
                    data=X_train.columns.tolist(),
                    dtype=h5py.string_dtype(),
                )
                f.create_dataset("y_columns", data=["RUL"], dtype=h5py.string_dtype())

            print(f"Processed data saved to {processed_filename}")
            return X_train, y_train, X_test, y_test
    return (NCMAPSSPreprocessor,)


@app.cell
def _(Dataset, np, torch):
    class NCMAPSSDatasetLazy(Dataset):
        """
        Lazy dataset for training.
        Converts input DataFrames to numpy arrays for fast indexing.
        Sequences are generated on-the-fly (lazy).
        """

        def __init__(self, X_df, y_df, sequence_length, features, sampling_rate=1):
            self.sequence_length = sequence_length
            self.features = features
            self.sampling_rate = sampling_rate

            # Convert DataFrame to numpy arrays once
            self.units = X_df["unit"].values
            self.X = X_df[features].values.astype(np.float32)
            self.y = y_df["RUL"].values.astype(np.float32)

            # Precompute valid sequence index windows
            self.sequence_indices = self._calculate_sequence_indices()

        def _calculate_sequence_indices(self):
            indices = []
            unique_units = np.unique(self.units)
            for unit_id in unique_units:
                unit_idx = np.where(self.units == unit_id)[0]
                for i in range(0, len(unit_idx) - self.sequence_length + 1, self.sampling_rate):
                    indices.append(unit_idx[i:i+self.sequence_length])
            return indices

        def __len__(self):
            return len(self.sequence_indices)

        def __getitem__(self, idx):
            idxs = self.sequence_indices[idx]
            sequence = self.X[idxs]
            target = self.y[idxs[-1]]
            return torch.from_numpy(sequence), torch.tensor([target], dtype=torch.float32)

    class NCMAPSSDatasetTestLazy(Dataset):
        """
        Lazy dataset for testing.
        Converts input DataFrames to numpy arrays for fast indexing.
        Sequences are generated on-the-fly (lazy).
        """

        def __init__(self, X_df, y_df, sequence_length, features, sampling_rate=1):
            self.sequence_length = sequence_length
            self.features = features
            self.sampling_rate = sampling_rate

            # Convert DataFrame to numpy arrays once
            self.units = X_df["unit"].values.astype(np.float32)
            self.cycles = X_df["cycle"].values.astype(np.float32)
            self.X = X_df[features].values.astype(np.float32)
            self.y = y_df["RUL"].values.astype(np.float32)

            # Precompute valid sequence index windows
            self.sequence_indices = self._calculate_sequence_indices()

        def _calculate_sequence_indices(self):
            indices = []
            unique_units = np.unique(self.units)
            for unit_id in unique_units:
                unit_idx = np.where(self.units == unit_id)[0]
                for i in range(0, len(unit_idx) - self.sequence_length + 1, self.sampling_rate):
                    indices.append(unit_idx[i:i+self.sequence_length])
            return indices

        def __len__(self):
            return len(self.sequence_indices)

        def __getitem__(self, idx):
            idxs = self.sequence_indices[idx]
            sequence = self.X[idxs]
            target = self.y[idxs[-1]]
            unit = self.units[idxs[-1]]
            cycle = self.cycles[idxs[-1]]
            return torch.from_numpy(sequence), torch.tensor([target], dtype=torch.float32), torch.tensor([unit]), torch.tensor([cycle])
    return NCMAPSSDatasetLazy, NCMAPSSDatasetTestLazy


@app.cell
def _(
    DataLoader,
    NCMAPSSDatasetLazy,
    NCMAPSSDatasetTestLazy,
    NCMAPSSPreprocessor,
    h5py,
    np,
    pd,
    torch,
):
    def preprocess_ncmapss_h5(
        raw_filename, processed_filename, scaler_type="minmax"
    ):
        """Complete preprocessing from raw H5 to processed H5"""
        preprocessor = NCMAPSSPreprocessor(
            scaler_type=scaler_type
        )
        X_train, y_train, X_test, y_test = preprocessor.preprocess_and_save(
            raw_filename, processed_filename
        )
        return preprocessor

    def create_lazy_dataloaders_from_h5(
        processed_filename, sequence_length=30, batch_size=32, num_workers=0, sampling_rate=1
    ):
        """Create memory-efficient lazy loading dataloaders"""

        print("Loading processed data...")
        # Load data (this is the only time full data is in memory)
        with h5py.File(processed_filename, "r") as f:
            X_train = pd.DataFrame(np.array(f["X_train"]))
            y_train = pd.DataFrame(np.array(f["y_train"]))
            X_test = pd.DataFrame(np.array(f["X_test"]))
            y_test = pd.DataFrame(np.array(f["y_test"]))

            # Get column names
            X_columns = f["X_columns"][:].astype(str)
            y_columns = f["y_columns"][:].astype(str)

            X_train.columns = X_columns
            y_train.columns = y_columns
            X_test.columns = X_columns
            y_test.columns = y_columns

        print(f"Creating lazy datasets with sampling rate: {sampling_rate}...")
        # Get feature columns
        features = [col for col in X_train.columns if col not in ["unit", "cycle", "Fc_mean", "Fc_std", "hs_mean", "hs_std"]]

        # Create lazy datasets
        train_dataset = NCMAPSSDatasetLazy(X_train, y_train, sequence_length, features, sampling_rate)
        # TODO: Refactor to have proper val dataset
        val_dataset = NCMAPSSDatasetLazy(X_test, y_test, sequence_length, features, sampling_rate)
        test_dataset = NCMAPSSDatasetTestLazy(X_test, y_test, sequence_length, features, sampling_rate)

        print(f"Features: {len(features)}")
        print(f"Training sequences available: {len(train_dataset)}")
        print(f"Test sequences available: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True,
            prefetch_factor=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True,
            prefetch_factor=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True,
            prefetch_factor=2
        )

        return train_loader, val_loader, test_loader, features
    return create_lazy_dataloaders_from_h5, preprocess_ncmapss_h5


@app.cell
def _(preprocess_ncmapss_h5):
    raw_filename = "N-CMAPSS_DS02-006.h5"
    raw_filename = f"data/17. Turbofan Engine Degradation Simulation Data Set 2/data_set/{raw_filename}"
    processed_filename = "data/processed/data.h5"

    preprocessor = preprocess_ncmapss_h5(raw_filename, processed_filename)
    return preprocessor, processed_filename


@app.cell
def _(create_lazy_dataloaders_from_h5, processed_filename):
    train_loader, val_loader, test_loader, features = create_lazy_dataloaders_from_h5(
        processed_filename, sequence_length=5, batch_size=8, num_workers=4, sampling_rate=1
    )
    return features, test_loader, train_loader, val_loader


@app.cell
def _(features):
    print(features)
    return


@app.cell
def _(nn, torch):
    class SimpleLSTM(nn.Module):
        """Simple LSTM model for RUL prediction"""

        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=True):
            """
            Args:
                input_size: Number of input features
                hidden_size: Number of LSTM hidden units
                num_layers: Number of LSTM layers
                dropout: Dropout rate for regularization
            """
            super(SimpleLSTM, self).__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional

            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )

            # Dropout layer
            self.dropout = nn.Dropout(dropout)

            # Output layer
            self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

        def forward(self, x):
            """
            Forward pass
            Args:
                x: Input tensor of shape (batch_size, sequence_length, input_size)
            Returns:
                Output tensor of shape (batch_size, 1)
            """
            # Initialize hidden and cell states
            batch_size = x.size(0)
            if self.bidirectional:
                h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
            else:
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

            # LSTM forward pass
            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

            # Use the last time step output
            last_output = lstm_out[:, -1, :]

            # Apply dropout
            last_output = self.dropout(last_output)

            # Final prediction
            output = self.fc(last_output)

            return output
    return (SimpleLSTM,)


@app.cell
def _(
    mean_absolute_error,
    mean_squared_error,
    nn,
    np,
    optim,
    plt,
    preprocessor,
    torch,
    tqdm,
):
    def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
        """
        Train the LSTM model
        """
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        train_losses = []
        val_losses = []

        print(f"Training on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        for epoch in tqdm(range(num_epochs)):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(device)
                targets = targets.to(device)

                # Forward pass
                optimizer.zero_grad()
                predictions = model(sequences)
                loss = criterion(predictions, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                # Print progress every 100 batches
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')

            avg_train_loss = train_loss / num_batches
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(device)
                    targets = targets.to(device)

                    predictions = model(sequences)
                    loss = criterion(predictions, targets)

                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)

            # Step the scheduler
            scheduler.step(avg_val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        return train_losses, val_losses

    def evaluate_model(model, test_loader, device='cuda'):
        """
        Evaluate the model on test data
        """
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                pred = model(sequences)

                predictions.extend(pred.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()

        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)

        print(f"Test Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return predictions, actuals, {'mse': mse, 'rmse': rmse, 'mae': mae}

    def plot_training_history(train_losses, val_losses, save_path="training_history.png"):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()

    def plot_predictions(predictions, actuals, max_points=1000, save_path="predictions.png"):
        """Plot predictions vs actual values"""
        # Sample points if too many
        if len(predictions) > max_points:
            indices = np.random.choice(len(predictions), max_points, replace=False)
            pred_sample = predictions[indices]
            actual_sample = actuals[indices]
        else:
            pred_sample = predictions
            actual_sample = actuals

        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.subplot(2, 1, 1)
        plt.scatter(actual_sample, pred_sample, alpha=0.6)
        plt.plot([actual_sample.min(), actual_sample.max()], 
                 [actual_sample.min(), actual_sample.max()], 'r--', lw=2)
        plt.xlabel('Actual RUL')
        plt.ylabel('Predicted RUL')
        plt.title('Predicted vs Actual RUL')
        plt.grid(True)

        # Error histogram
        plt.subplot(2, 1, 2)
        errors = pred_sample - actual_sample
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


    def evaluate_model(model, test_loader, device='cuda'):
        """
        Evaluate the model on test data, keeping track of unit and cycle
        """
        model.eval()
        predictions = []
        actuals = []
        units = []
        cycles = []

        with torch.no_grad():
            for sequences, targets, unit, cycle in test_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                pred = model(sequences)

                predictions.extend(pred.cpu().numpy().flatten())
                actuals.extend(targets.cpu().numpy().flatten())
                units.extend(unit.numpy().flatten())
                cycles.extend(cycle.numpy().flatten())

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        units = np.array(units, dtype=int)
        cycles = np.array(cycles, dtype=int)

        # Calculate overall metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)

        print(f"Test Results (scaled):")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return predictions, actuals, units, cycles, {'mse': mse, 'rmse': rmse, 'mae': mae}

    def plot_predictions_by_unit(predictions, actuals, units, cycles, save_path="predictions_by_unit.png", max_units=9):
        """
        Plot predictions vs actual RUL per unit
        """
        unique_units = np.unique(units)
        n_units = min(len(unique_units), max_units)

        plt.figure(figsize=(15, 12))

        for i, unit_id in enumerate(unique_units[:n_units]):
            mask = units == unit_id
            unit_cycles = cycles[mask]
            unit_preds = predictions[mask]
            unit_actuals = actuals[mask]

            # Sort by cycle for proper line plotting
            sort_idx = np.argsort(unit_cycles)
            unit_cycles = unit_cycles[sort_idx]
            unit_preds = unit_preds[sort_idx]
            unit_actuals = unit_actuals[sort_idx]

            # Inverse transform of RUL
            unit_preds = preprocessor.target_scaler.inverse_transform(unit_preds.reshape(-1, 1))
            unit_actuals = preprocessor.target_scaler.inverse_transform(unit_actuals.reshape(-1, 1))

            plt.subplot(int(np.ceil(n_units/3)), 3, i+1)
            plt.plot(unit_cycles, unit_actuals, label="Actual RUL", color="blue")
            plt.plot(unit_cycles, unit_preds, label="Predicted RUL", color="orange", linestyle="--")
            plt.xlabel("Cycle")
            plt.ylabel("RUL")
            plt.title(f"Unit {int(unit_id)}")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    return (
        evaluate_model,
        plot_predictions_by_unit,
        plot_training_history,
        train_model,
    )


@app.cell
def _(
    SimpleLSTM,
    evaluate_model,
    features,
    plot_predictions_by_unit,
    plot_training_history,
    test_loader,
    torch,
    train_loader,
    train_model,
    val_loader,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model parameters
    input_size = len(features)
    hidden_size = 128
    num_layers = 3
    dropout = 0.5
    num_epochs=100
    learning_rate=0.001
    bidirectional = False

    print(f"Input features: {input_size}")

    # Create model
    model = SimpleLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )

    print(f"Model architecture:")
    print(model)

    # Train model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Evaluate model
    # predictions, actuals, metrics = evaluate_model(model, test_loader, device)
    preds, actuals, units, cycles, metrics = evaluate_model(model, test_loader, device)

    # Plot results
    # plot_predictions(predictions, actuals)
    plot_predictions_by_unit(preds, actuals, units, cycles)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        },
        'metrics': metrics
    }, 'lstm_rul_model.pth')

    print("Model saved as 'lstm_rul_model.pth'")
    return


if __name__ == "__main__":
    app.run()

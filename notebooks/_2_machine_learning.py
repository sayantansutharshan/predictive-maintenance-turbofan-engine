import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import h5py

    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    import torch
    from torch.utils.data import Dataset, DataLoader

    return (
        DataLoader,
        Dataset,
        MinMaxScaler,
        StandardScaler,
        h5py,
        np,
        pd,
        torch,
    )


@app.cell
def _(MinMaxScaler, StandardScaler, h5py, np, pd):
    class NCMAPSSPreprocessor:
        def __init__(self, sequence_length=30, scaler_type="minmax"):
            """
            CMAPSS data preprocessor for LSTM models using H5 files

            Args:
                sequence_length: Length of sequences for LSTM input
                scaler_type: 'minmax' or 'standard' scaling
            """
            self.sequence_length = sequence_length
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

            print("Scaling features...")
            # Get feature columns (exclude unit and cycle)
            features = [
                feature
                for feature in X_train.columns
                if feature not in ["unit", "cycle"]
            ]

            # Initialize and fit scaler on training data
            if self.scaler_type == "minmax":
                self.feature_scaler = MinMaxScaler()
            else:
                self.feature_scaler = StandardScaler()

            X_train_scaled = self.feature_scaler.fit_transform(X_train[features])
            X_test_scaled = self.feature_scaler.transform(X_test[features])

            # Update DataFrames with scaled features
            X_train[features] = X_train_scaled
            X_test[features] = X_test_scaled

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

        def create_sequences_from_dataframe(self, X_df, y_df):
            """Create sequences for LSTM from DataFrame format"""
            # Get feature columns (exclude unit and cycle)
            features = [col for col in X_df.columns if col not in ["unit", "cycle"]]

            sequences = []
            targets = []
            unit_ids = []

            # Get unique units
            units = X_df["unit"].unique()

            for unit_id in units:
                # Get data for this unit
                unit_mask = X_df["unit"] == unit_id
                unit_X = X_df[unit_mask]
                unit_y = y_df[unit_mask]

                # Create sequences for this unit
                for i in range(len(unit_X) - self.sequence_length + 1):
                    # Get sequence of features
                    seq = unit_X[features].iloc[i : i + self.sequence_length].values
                    sequences.append(seq)

                    # Target is RUL at the end of sequence
                    target = unit_y["RUL"].iloc[i + self.sequence_length - 1]
                    targets.append(target)

                    unit_ids.append(unit_id)

            return np.array(sequences), np.array(targets), np.array(unit_ids)

        def create_test_sequences_from_dataframe(self, X_df, y_df=None):
            """Create test sequences (last sequence per unit) from DataFrame format"""
            features = [col for col in X_df.columns if col not in ["unit", "cycle"]]

            sequences = []
            targets = []
            unit_ids = []

            units = X_df["unit"].unique()

            for unit_id in units:
                unit_mask = X_df["unit"] == unit_id
                unit_X = X_df[unit_mask]

                # Take the last sequence_length cycles
                if len(unit_X) >= self.sequence_length:
                    seq = unit_X[features].iloc[-self.sequence_length :].values
                    sequences.append(seq)
                    unit_ids.append(unit_id)

                    # Get target if available
                    if y_df is not None:
                        unit_y = y_df[unit_mask].sort_values(
                            X_df[unit_mask]["cycle"].index
                        )
                        target = unit_y["RUL"].iloc[-1]  # Last RUL value
                        targets.append(target)

            sequences = np.array(sequences)
            targets = np.array(targets) if targets else None

            return sequences, targets, np.array(unit_ids)

    return (NCMAPSSPreprocessor,)


@app.cell
def _(Dataset, torch):
    class NCMAPSSDataset(Dataset):
        """PyTorch Dataset for NCMAPSS data"""

        def __init__(self, sequences, targets=None):
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets) if targets is not None else None

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            if self.targets is not None:
                return self.sequences[idx], self.targets[idx]
            else:
                return self.sequences[idx]

    return


@app.cell
def _(Dataset, np, torch):
    class NCMAPSSDatasetLazy(Dataset):
        """Memory-efficient lazy loading PyTorch Dataset for NCMAPSS data"""

        def __init__(self, X_df, y_df, sequence_length, features):
            """
            Args:
                X_df: Feature DataFrame (kept in memory as reference)
                y_df: Target DataFrame
                sequence_length: Length of sequences for LSTM
                features: List of feature column names
            """
            self.X_df = X_df
            self.y_df = y_df
            self.sequence_length = sequence_length
            self.features = features

            # Convert to float32 to save memory
            self.X_df[features] = self.X_df[features].astype(np.float32)
            self.y_df = self.y_df.astype(np.float32)

            # Pre-calculate sequence indices (lightweight metadata only)
            self.sequence_indices = self._calculate_sequence_indices()

            print(f"Lazy dataset created with {len(self.sequence_indices)} sequences")

        def _calculate_sequence_indices(self):
            """Calculate where each valid sequence starts (metadata only)"""
            indices = []
            units = self.X_df["unit"].unique()

            for unit_id in units:
                unit_mask = self.X_df["unit"] == unit_id
                unit_length = unit_mask.sum()

                # Only store indices, not actual data
                unit_indices = self.X_df[unit_mask].index.tolist()

                # Calculate valid sequence starting positions
                for i in range(unit_length - self.sequence_length + 1):
                    indices.append(
                        {
                            "unit_id": unit_id,
                            "start_idx": unit_indices[i],
                            "end_idx": unit_indices[i + self.sequence_length - 1],
                            "unit_position": i,
                        }
                    )

            return indices

        def __len__(self):
            return len(self.sequence_indices)

        def __getitem__(self, idx):
            """Generate sequence on-demand (lazy loading)"""
            seq_info = self.sequence_indices[idx]
            unit_id = seq_info["unit_id"]
            unit_position = seq_info["unit_position"]

            # Get unit data (still efficient since data is sorted)
            unit_mask = self.X_df["unit"] == unit_id
            unit_X = self.X_df[unit_mask]
            unit_y = self.y_df[unit_mask]

            # Extract sequence (only creates small sequence, not full dataset)
            start_pos = unit_position
            end_pos = start_pos + self.sequence_length

            sequence = unit_X[self.features].iloc[start_pos:end_pos].values
            target = unit_y["RUL"].iloc[end_pos - 1]

            # Convert to tensors on-the-fly
            return torch.FloatTensor(sequence), torch.FloatTensor([target])

    class NCMAPSSDatasetTestLazy(Dataset):
        """Lazy loading for test data (last sequence per unit)"""

        def __init__(self, X_df, y_df, sequence_length, features):
            self.X_df = X_df
            self.y_df = y_df
            self.sequence_length = sequence_length
            self.features = features

            # Convert to float32
            self.X_df[features] = self.X_df[features].astype(np.float32)
            if y_df is not None:
                self.y_df = self.y_df.astype(np.float32)

            # Calculate test sequence indices (one per unit)
            self.test_indices = self._calculate_test_indices()

            print(f"Lazy test dataset created with {len(self.test_indices)} sequences")

        def _calculate_test_indices(self):
            """Calculate indices for last sequence of each unit"""
            indices = []
            units = self.X_df["unit"].unique()

            for unit_id in units:
                unit_mask = self.X_df["unit"] == unit_id
                unit_length = unit_mask.sum()

                # Only include units with enough data
                if unit_length >= self.sequence_length:
                    indices.append(
                        {
                            "unit_id": unit_id,
                            "start_position": unit_length - self.sequence_length,
                        }
                    )

            return indices

        def __len__(self):
            return len(self.test_indices)

        def __getitem__(self, idx):
            """Generate last sequence for unit on-demand"""
            test_info = self.test_indices[idx]
            unit_id = test_info["unit_id"]
            start_pos = test_info["start_position"]

            # Get unit data
            unit_mask = self.X_df["unit"] == unit_id
            unit_X = self.X_df[unit_mask]

            # Extract last sequence
            sequence = (
                unit_X[self.features]
                .iloc[start_pos : start_pos + self.sequence_length]
                .values
            )

            # Get target if available
            if self.y_df is not None:
                unit_y = self.y_df[unit_mask]
                target = unit_y["RUL"].iloc[-1]  # Last RUL value
                return torch.FloatTensor(sequence), torch.FloatTensor([target])
            else:
                return torch.FloatTensor(sequence)
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
    def preprocess_cmapss_h5(
        raw_filename, processed_filename, sequence_length=30, scaler_type="minmax"
    ):
        """Complete preprocessing from raw H5 to processed H5"""
        preprocessor = NCMAPSSPreprocessor(
            sequence_length=sequence_length, scaler_type=scaler_type
        )
        X_train, y_train, X_test, y_test = preprocessor.preprocess_and_save(
            raw_filename, processed_filename
        )
        return preprocessor

    def create_lstm_dataloaders_from_h5(
        processed_filename, sequence_length=30, batch_size=32, scaler_type="minmax"
    ):
        """Create LSTM dataloaders from processed H5 file"""
        preprocessor = NCMAPSSPreprocessor(
            sequence_length=sequence_length, scaler_type=scaler_type
        )

        print("Loading processed data...")
        X_train, y_train, X_test, y_test = preprocessor.load_processed_h5_data(
            processed_filename
        )

        print("Creating sequences...")
        # Create training sequences
        X_train_seq, y_train_seq, train_units = (
            preprocessor.create_sequences_from_dataframe(X_train, y_train)
        )

        # Create test sequences (last sequence per unit)
        X_test_seq, y_test_seq, test_units = (
            preprocessor.create_test_sequences_from_dataframe(X_test, y_test)
        )

        print(f"Training sequences shape: {X_train_seq.shape}")
        print(f"Training targets shape: {y_train_seq.shape}")
        print(f"Test sequences shape: {X_test_seq.shape}")
        print(f"Test targets shape: {y_test_seq.shape}")
        print(f"Number of features: {X_train_seq.shape[2]}")

        # Create PyTorch datasets
        train_dataset = NCMAPSSDatasetLazy(X_train_seq, y_train_seq)
        test_dataset = NCMAPSSDatasetLazy(X_test_seq, y_test_seq)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, preprocessor, y_test_seq

    # Updated preprocessing functions
    def create_lazy_dataloaders_from_h5(
        processed_filename, sequence_length=30, batch_size=32, num_workers=0
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

        print("Creating lazy datasets...")
        # Get feature columns
        features = [col for col in X_train.columns if col not in ["unit", "cycle"]]

        # Create lazy datasets (minimal memory usage)
        train_dataset = NCMAPSSDatasetLazy(X_train, y_train, sequence_length, features)
        test_dataset = NCMAPSSDatasetTestLazy(X_test, y_test, sequence_length, features)

        print(f"Features: {len(features)}")
        print(f"Training sequences available: {len(train_dataset)}")
        print(f"Test sequences available: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,  # Use multiple workers for better performance
            pin_memory=True if torch.cuda.is_available() else False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return train_loader, test_loader, features
    return (create_lazy_dataloaders_from_h5,)


@app.cell
def _():
    raw_filename = "N-CMAPSS_DS01-005.h5"
    raw_filename = f"data/17. Turbofan Engine Degradation Simulation Data Set 2/data_set/{raw_filename}"
    processed_filename = "data/processed/data.h5"
    return (processed_filename,)


@app.cell
def _(create_lazy_dataloaders_from_h5, processed_filename):
    train_loader_1, test_loader_1, features_1 = create_lazy_dataloaders_from_h5(
        processed_filename, sequence_length=30, batch_size=32, num_workers=2
    )
    return


if __name__ == "__main__":
    app.run()

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

    import altair as alt
    return MinMaxScaler, StandardScaler, alt, h5py, mo, np, pd


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
            X_train["RUL"] = y_train["RUL"].values
            X_test["RUL"] = y_test["RUL"].values
    
            # Drop 'unit' and 'cycle' for aggregation
            features = [c for c in X_train.columns if c not in ["unit", "cycle", "RUL"]]
    
            agg_funcs = {f: ["mean", "std"] for f in features}
    
            def aggregate_cycles(X):
                # Group sensors/settings
                X_agg = (
                    X.groupby(["unit", "cycle"])
                    .agg(agg_funcs)
                )
                # Flatten multi-level cols
                X_agg.columns = ["_".join(col).strip() for col in X_agg.columns.values]
                X_agg = X_agg.reset_index()
    
                # Take one RUL per cycle (all equal, so .first() works)
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

            # Ensure column names are Python str
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
        
            # Scale RUL only
            y_train["RUL"] = self.target_scaler.fit_transform(y_train[["RUL"]])
            y_test["RUL"]  = self.target_scaler.transform(y_test[["RUL"]])

            # # Update DataFrames with scaled features
            # y_train["RUL"] = y_train_scaled
            # y_test["RUL"] = y_test_scaled

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
def _():
    raw_filename = "N-CMAPSS_DS02-006.h5"
    raw_filename = f"data/17. Turbofan Engine Degradation Simulation Data Set 2/data_set/{raw_filename}"
    processed_filename = "data/processed/data.h5"
    return processed_filename, raw_filename


@app.cell
def _(NCMAPSSPreprocessor, processed_filename, raw_filename):
    preprocessor = NCMAPSSPreprocessor()

    X_train, y_train, X_test, y_test = preprocessor.preprocess_and_save(f"../{raw_filename}", f"../{processed_filename}")
    # X_train, y_train, X_test, y_test = preprocessor.load_processed_h5_data(f"../{processed_filename}")
    return X_train, y_train


@app.cell
def _(y_train):
    y_train.describe()
    return


@app.cell
def _(X_train):
    X_train.columns
    return


@app.cell
def _(X_train):
    X_train['unit'].unique()
    return


@app.cell
def _(X_train, alt, mo):
    # 2. Sensor degradation over time (example with key sensors)
    key_sensors = ['T24', 'T30', 'T50', 'Ps30', 'Nf', 'Nc', 'Wf']
    df_sensors = X_train[X_train['unit'] == 2].melt(
        id_vars=['unit', 'cycle'],
        var_name='sensor',
        value_name='value'
    )

    chart2 = alt.Chart(df_sensors.sample(n=50000) if len(df_sensors) > 50000 else df_sensors).mark_line(
        opacity=0.3,
        size=0.5
    ).encode(
        x=alt.X('cycle:Q', title='Cycle'),
        y=alt.Y('value:Q', title='Sensor Value'),
        color=alt.Color('unit:N', legend=None),
        facet=alt.Facet('sensor:N', columns=4, title='Sensor Degradation Over Time')
    ).resolve_scale(
        x='shared',
        y='independent'
    ).properties(
        width=200,
        height=150
    )

    mo.ui.altair_chart(chart2)
    return


if __name__ == "__main__":
    app.run()

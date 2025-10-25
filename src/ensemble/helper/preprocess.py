import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_transform_pipeline(data):
    
    df = pl.DataFrame([data["user_input"]])
    df = df.drop(["nameOrig", "nameDest"])
    target = "isFraud"
    features = df.columns
    features = [c for c in features if c != target]

    cat_cols = ["type"]  # adjust if more categorical features exist
    le_dict = {}  # store label encoders for each column
    for col in cat_cols:
        # Convert to numpy for LabelEncoder
        le = LabelEncoder()
        df = df.with_columns([
            pl.Series(col, le.fit_transform(df[col].to_numpy()))
        ])
        le_dict[col] = le

    df = df.with_columns([
        (pl.col('newbalanceOrig') - pl.col('oldbalanceOrg')).alias('delta_orig'),
        (pl.col('newbalanceDest') - pl.col('oldbalanceDest')).alias('delta_dest'),
        (pl.col('amount') / (pl.col('oldbalanceOrg') + 1e-6)).alias('amount_ratio'),
        (pl.col('amount') > 0.8).cast(pl.Int8).alias('high_amount')
    ])

    num_cols = [col for col in df.columns if col not in cat_cols + [target]]
    scaler = StandardScaler()

    # Polars does not directly support sklearn scaling, so convert numerics to numpy, scale, then back
    df_num_scaled = scaler.fit_transform(df[num_cols].to_numpy())

    # Create scaled column names
    scaled_num_cols = [f"{col}_scaled" for col in num_cols]

    df = df.with_columns([
        pl.Series(name, df_num_scaled[:, i]) for i, name in enumerate(scaled_num_cols)
    ])

    df = df.drop(num_cols)
    X = df.select([col for col in df.columns if col != target])

    return X.to_pandas() 
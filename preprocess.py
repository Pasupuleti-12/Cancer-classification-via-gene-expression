def preprocess_data(data_file, label_file):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    X = pd.read_csv(data_file)
    y_df = pd.read_csv(label_file)

    # Drop non-numeric columns (like IDs)
    X = X.select_dtypes(include=['number'])

    # Get label column
    if y_df.shape[1] >= 2:
        y = y_df.iloc[:, 1]
    else:
        y = y_df.iloc[:, 0]

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, label_encoder

import pandas as pd


def load_data(file_path):
    """Load the customer shopping dataset."""
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean and transform the customer shopping data."""

    # Fill missing Review Rating values
    df['Review Rating'] = df['Review Rating'].fillna(
        df.groupby('Category')['Review Rating'].transform('median')
    )

    # Standardize column names
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')

    # Rename purchase amount column
    df = df.rename(
        columns={'purchase_amount_(usd)': 'purchase_amount'}
    )

    # Create age groups
    labels = [
        'Young Adult',
        'Adult',
        'Middle-aged',
        'Senior'
    ]

    df['age_group'] = pd.qcut(
        df['age'],
        q=4,
        labels=labels
    ).astype(str)

    # Convert purchase frequency into number of days
    frequency_mapping = {
        'Fortnightly': 14,
        'Weekly': 7,
        'Monthly': 30,
        'Quarterly': 90,
        'Bi-Weekly': 14,
        'Annually': 365,
        'Every 3 Months': 90
    }

    df['purchase_frequency_days'] = (
        df['frequency_of_purchases'].map(frequency_mapping)
    )

    # Remove redundant column
    if 'promo_code_used' in df.columns:
        df = df.drop('promo_code_used', axis=1)

    return df


if __name__ == "__main__":

    file_path = "data/customer_shopping_behavior.csv"

    df = load_data(file_path)

    print("Original shape:", df.shape)

    df = clean_data(df)

    print("Cleaned shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 5 rows:")
    print(df.head())
# DS-Containers/app/generate_test_data.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_simple_expense_data(output_dir: str = 'data', num_records: int = 200, seed: int = 42):
    """
    Generates a simple CSV file with synthetic restaurant expense data.
    This version is simplified for local testing and does not rely on date arguments.
    """
    # Use a fixed date for consistency in local testing
    fixed_date = datetime(2025, 7, 1) # Example: July 1, 2025

    np.random.seed(seed) # Seed for reproducibility

    products = [
        'Fresh Vegetables', 'Meats (Beef, Chicken)', 'Dairy Products', 'Spices & Herbs',
        'Grains (Rice, Pasta)', 'Beverages (Soft Drinks, Juices)', 'Bakery Items',
        'Cleaning Supplies', 'Disposable Goods', 'Seafood', 'Fruits'
    ]
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Local Farm', 'Wholesale Foods']
    payment_methods = ['Credit Card', 'Bank Transfer', 'Cash']

    data = {
        'date': [fixed_date + timedelta(days=np.random.randint(0, 7)) for _ in range(num_records)],
        'product': np.random.choice(products, num_records),
        'quantity': np.random.randint(1, 50, num_records),
        'unit_price': np.random.uniform(0.5, 100.0, num_records).round(2),
        'supplier': np.random.choice(suppliers, num_records),
        'payment_method': np.random.choice(payment_methods, num_records),
        'notes': np.random.choice(['None', 'Urgent', 'Bulk Order', 'Special Request'], num_records, p=[0.7, 0.1, 0.1, 0.1])
    }

    df = pd.DataFrame(data)
    df['total_price'] = (df['quantity'] * df['unit_price']).round(2)

    # Determine the output directory relative to where this script is run
    # Assuming this script is in 'app/' and 'data/' is at the 'DS-Containers/' level
    script_dir = os.path.dirname(__file__)
    final_output_dir = os.path.join(script_dir, '..', output_dir)
    
    os.makedirs(final_output_dir, exist_ok=True)
    
    filename = os.path.join(final_output_dir, f"test_expense_data.csv")
    df.to_csv(filename, index=False)
    print(f"Generated {num_records} records to: {filename}")
    return filename

if __name__ == "__main__":
    print("Running simple data generation for local testing...")
    generate_simple_expense_data()
    print("Local data generation complete.")

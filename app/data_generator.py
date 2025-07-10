# DS-Containers/app/data_generator.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_expense_data(output_dir: str, num_records: int = 200, seed: int = 42) -> str:
    """
    Generates synthetic restaurant expense data and saves it to a CSV.
    
    Args:
        output_dir (str): The directory where the CSV file should be saved.
                          This path should be absolute or relative to the script's execution context.
        num_records (int): Number of records to generate.
        seed (int): Seed for reproducibility.
    
    Returns:
        str: The full path to the generated CSV file.
    """
    current_run_date = datetime.now()
    np.random.seed(seed)

    products = [
        'Fresh Vegetables', 'Meats (Beef, Chicken)', 'Dairy Products', 'Spices & Herbs',
        'Grains (Rice, Pasta)', 'Beverages (Soft Drinks, Juices)', 'Bakery Items',
        'Cleaning Supplies', 'Disposable Goods', 'Seafood', 'Fruits'
    ]
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Local Farm', 'Wholesale Foods']
    payment_methods = ['Credit Card', 'Bank Transfer', 'Cash']

    data = {
        'date': [current_run_date + timedelta(days=np.random.randint(0, 7)) for _ in range(num_records)],
        'product': np.random.choice(products, num_records),
        'quantity': np.random.randint(1, 50, num_records),
        'unit_price': np.random.uniform(0.5, 100.0, num_records).round(2),
        'supplier': np.random.choice(suppliers, num_records),
        'payment_method': np.random.choice(payment_methods, num_records),
        'notes': np.random.choice(['None', 'Urgent', 'Bulk Order', 'Special Request'], num_records, p=[0.7, 0.1, 0.1, 0.1])
    }

    df = pd.DataFrame(data)
    df['total_price'] = (df['quantity'] * df['unit_price']).round(2)

    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"expense_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv") # Unique filename per run
    df.to_csv(filename, index=False)
    print(filename) # Print only the filename for easy capture by shell scripts
    return filename

if __name__ == "__main__":
    import sys
    # When run directly, it expects an output directory as an argument.
    # For local testing, you would typically run it like:
    # python data_generator.py ../data
    
    if len(sys.argv) > 1:
        output_directory_arg = sys.argv[1]
        generate_expense_data(output_directory_arg)
    else:
        print("Usage: python data_generator.py <output_directory_path>")
        print("Example: python data_generator.py ../data")
        sys.exit(1) # Exit if no output directory is provided

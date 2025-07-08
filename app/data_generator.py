# DS-Containers/app/data_generator.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_weekly_expense_data(start_date_str, num_records=200, output_dir='data'):
    """
    Generates synthetic weekly restaurant expense data and saves it to a CSV.
    output_dir is relative to the container's WORKDIR (/app), but mapped to host's data/
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = start_date + timedelta(days=6) # A full week

    np.random.seed(int(start_date.strftime('%Y%m%d'))) # Seed for reproducibility per week

    products = [
        'Fresh Vegetables', 'Meats (Beef, Chicken)', 'Dairy Products', 'Spices & Herbs',
        'Grains (Rice, Pasta)', 'Beverages (Soft Drinks, Juices)', 'Bakery Items',
        'Cleaning Supplies', 'Disposable Goods', 'Seafood', 'Fruits'
    ]
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Local Farm', 'Wholesale Foods']
    payment_methods = ['Credit Card', 'Bank Transfer', 'Cash']

    data = {
        'date': [start_date + timedelta(days=np.random.randint(0, 7)) for _ in range(num_records)],
        'product': np.random.choice(products, num_records),
        'quantity': np.random.randint(1, 50, num_records),
        'unit_price': np.random.uniform(0.5, 100.0, num_records).round(2),
        'supplier': np.random.choice(suppliers, num_records),
        'payment_method': np.random.choice(payment_methods, num_records),
        'notes': np.random.choice(['None', 'Urgent', 'Bulk Order', 'Special Request'], num_records, p=[0.7, 0.1, 0.1, 0.1])
    }

    df = pd.DataFrame(data)
    df['total_price'] = (df['quantity'] * df['unit_price']).round(2)

    # Ensure output directory exists (relative to container's /app, which maps to host's data/)
    # The 'data' directory is mounted at '/data' in the container
    # So, we need to write to /data/ instead of just 'data'
    container_output_dir = os.path.join('/data', output_dir) # Corrected path within container
    os.makedirs(container_output_dir, exist_ok=True)

    filename = os.path.join(container_output_dir, f"weekly_expense_{start_date.strftime('%Y-%m-%d')}.csv")
    df.to_csv(filename, index=False)
    print(f"Generated {num_records} records for week starting {start_date_str} to {filename}")
    return filename
#!/Users/jimcorrell/development/neho/scoutiq/data/lahman/lahman_1871-2024u_csv/venv/bin/python3
"""
Load Lahman Baseball Database CSV files into PostgreSQL
"""
import os
import csv
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys

# Database configuration
DB_NAME = "lahman"
DB_USER = os.getenv("PGUSER", os.getenv("USER"))
DB_PASSWORD = os.getenv("PGPASSWORD", "")
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5432")

# CSV directory
CSV_DIR = os.path.dirname(os.path.abspath(__file__))

# Tables to skip
SKIP_FILES = ["readme2024u.txt"]

def get_postgres_type(value, column_name):
    """Infer PostgreSQL data type from sample value"""
    if not value or value == '':
        return 'TEXT'
    
    # Try integer
    try:
        int(value)
        # Check if it's a year-like field
        if 'year' in column_name.lower() or column_name.lower() in ['yearid']:
            return 'INTEGER'
        # Check if it's an ID field that might be large
        if 'id' in column_name.lower():
            return 'INTEGER'
        return 'INTEGER'
    except ValueError:
        pass
    
    # Try float
    try:
        float(value)
        return 'DOUBLE PRECISION'
    except ValueError:
        pass
    
    # Default to TEXT
    return 'TEXT'

def infer_column_types(csv_file, sample_rows=100):
    """Infer column types by sampling the CSV file"""
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        
        # Initialize type tracking
        type_votes = {col: {} for col in columns}
        
        # Sample rows
        for i, row in enumerate(reader):
            if i >= sample_rows:
                break
            for col in columns:
                col_type = get_postgres_type(row[col], col)
                type_votes[col][col_type] = type_votes[col].get(col_type, 0) + 1
        
        # Determine final types (prefer TEXT if mixed)
        column_types = {}
        for col in columns:
            if not type_votes[col]:
                column_types[col] = 'TEXT'
            else:
                # If TEXT appears in votes, use TEXT (most permissive)
                if 'TEXT' in type_votes[col]:
                    column_types[col] = 'TEXT'
                # If DOUBLE PRECISION appears, use it (more general than INTEGER)
                elif 'DOUBLE PRECISION' in type_votes[col]:
                    column_types[col] = 'DOUBLE PRECISION'
                else:
                    column_types[col] = 'INTEGER'
        
        return column_types

def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        
        if exists:
            print(f"Database '{DB_NAME}' already exists.")
            response = input("Drop and recreate? (yes/no): ")
            if response.lower() == 'yes':
                cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(DB_NAME)))
                print(f"Dropped database '{DB_NAME}'")
            else:
                print("Using existing database.")
                cursor.close()
                conn.close()
                return
        
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
        print(f"Created database '{DB_NAME}'")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")
        sys.exit(1)

def create_table_and_load(conn, csv_file):
    """Create table and load data from CSV file"""
    table_name = os.path.splitext(os.path.basename(csv_file))[0].lower()
    
    print(f"\nProcessing {table_name}...")
    
    # Infer column types
    column_types = infer_column_types(csv_file)
    
    cursor = conn.cursor()
    
    try:
        # Drop table if exists
        cursor.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
            sql.Identifier(table_name)
        ))
        
        # Create table
        columns_def = [
            sql.SQL("{} {}").format(
                sql.Identifier(col),
                sql.SQL(col_type)
            )
            for col, col_type in column_types.items()
        ]
        
        create_table_query = sql.SQL("CREATE TABLE {} ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(columns_def)
        )
        
        cursor.execute(create_table_query)
        print(f"  Created table '{table_name}'")
        
        # Load data using COPY
        with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
            # Skip header
            next(f)
            cursor.copy_expert(
                sql.SQL("COPY {} FROM STDIN WITH (FORMAT CSV, NULL '')").format(
                    sql.Identifier(table_name)
                ),
                f
            )
        
        # Get row count
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(
            sql.Identifier(table_name)
        ))
        count = cursor.fetchone()[0]
        print(f"  Loaded {count:,} rows into '{table_name}'")
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        print(f"  Error processing {table_name}: {e}")
        raise
    
    finally:
        cursor.close()

def main():
    # Create database
    create_database()
    
    # Connect to the lahman database
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print(f"\nConnected to database '{DB_NAME}'")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Get all CSV files
    csv_files = [
        os.path.join(CSV_DIR, f)
        for f in os.listdir(CSV_DIR)
        if f.endswith('.csv') and f not in SKIP_FILES
    ]
    csv_files.sort()
    
    print(f"\nFound {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            create_table_and_load(conn, csv_file)
        except Exception as e:
            print(f"Failed to process {csv_file}: {e}")
            continue
    
    conn.close()
    
    print("\n" + "="*60)
    print("Database setup complete!")
    print(f"Database name: {DB_NAME}")
    print(f"Connect with: psql -d {DB_NAME}")
    print("="*60)

if __name__ == "__main__":
    main()

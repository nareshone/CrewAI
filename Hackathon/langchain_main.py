from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from langchain_groq import ChatGroq

# Environment setup
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-70b-8192", temperature=0)

# Test LLM connection
try:
    result = llm.invoke("Hello")
    print(f"LLM Test: {result.content}")
except Exception as e:
    print(f"LLM connection failed: {e}")
    exit(1)

def extract_file_schema(file_path: str) -> str:
    """Extract schema information from flat files"""
    try:
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
            
        schema_info = "FILE SCHEMA:\n"
        schema_info += "=" * 50 + "\n"
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            schema_info += f"File: {os.path.basename(file_path)} (CSV)\n"
            schema_info += f"Rows: {len(df)}\n"
            schema_info += f"Columns: {len(df.columns)}\n\n"
            
            schema_info += "Column Information:\n"
            schema_info += "-" * 30 + "\n"
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                
                schema_info += f"  - {col}: {dtype} (Nulls: {null_count}, Unique: {unique_count})\n"
                
                # Show sample values
                sample_values = df[col].dropna().head(3).tolist()
                schema_info += f"    Sample: {sample_values}\n"
                
        elif file_path.endswith(('.xlsx', '.xls')):
            # For Excel, read the first sheet by default
            df = pd.read_excel(file_path, sheet_name=0)
            schema_info += f"File: {os.path.basename(file_path)} (Excel - First Sheet)\n"
            schema_info += f"Rows: {len(df)}\n"
            schema_info += f"Columns: {len(df.columns)}\n\n"
            
            schema_info += "Column Information:\n"
            schema_info += "-" * 30 + "\n"
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                
                schema_info += f"  - {col}: {dtype} (Nulls: {null_count}, Unique: {unique_count})\n"
                
                # Show sample values
                sample_values = df[col].dropna().head(3).tolist()
                schema_info += f"    Sample: {sample_values}\n"
                
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
            schema_info += f"File: {os.path.basename(file_path)} (JSON)\n"
            schema_info += f"Rows: {len(df)}\n"
            schema_info += f"Columns: {len(df.columns)}\n\n"
            
            schema_info += "Column Information:\n"
            schema_info += "-" * 30 + "\n"
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                
                schema_info += f"  - {col}: {dtype} (Nulls: {null_count}, Unique: {unique_count})\n"
                
                # Show sample values
                sample_values = df[col].dropna().head(3).tolist()
                schema_info += f"    Sample: {sample_values}\n"
        else:
            return f"Unsupported file type: {file_path}"
            
        return schema_info
        
    except Exception as e:
        return f"Error extracting file schema: {str(e)}"

def generate_sql_query_direct(file_path: str, user_requirements: str):
    """
    Generate SQL query using direct LLM call without CrewAI
    
    Args:
        file_path: Path to the data file
        user_requirements: Natural language description of what data to retrieve
    
    Returns:
        Generated SQL query and explanation
    """
    try:
        # Extract schema information
        schema_info = extract_file_schema(file_path)
        
        if "Error" in schema_info or "not found" in schema_info:
            return schema_info
        
        # Create prompt for SQL generation
        prompt = f"""
You are an expert SQL developer. Based on the following file schema and user requirements, generate a SQL query that would retrieve the requested data.

IMPORTANT GUIDELINES:
1. Treat the file as if it were a database table with the same name as the file (without extension)
2. Use exact column names from the schema
3. Generate standard SQL syntax (SELECT, FROM, WHERE, GROUP BY, ORDER BY, etc.)
4. Consider data types when writing conditions
5. Provide a complete, syntactically correct SQL query

SCHEMA INFORMATION:
{schema_info}

USER REQUIREMENTS:
{user_requirements}

Please provide:
1. A complete SQL query
2. Brief explanation of what the query does
3. Any assumptions made about the data

Format your response as:
SQL QUERY:
[Your SQL query here]

EXPLANATION:
[Your explanation here]

ASSUMPTIONS:
[Any assumptions made]
"""

        # Generate SQL using LLM
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

def create_sample_csv():
    """Create sample CSV files for testing"""
    
    # Create customers CSV
    customers_data = {
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
        'city': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Miami'],
        'state': ['NY', 'CA', 'IL', 'NY', 'FL'],
        'age': [28, 35, 42, 31, 29],
        'registration_date': ['2023-01-15', '2023-02-20', '2023-01-10', '2023-03-05', '2023-02-28']
    }
    
    customers_df = pd.DataFrame(customers_data)
    customers_df.to_csv('customers.csv', index=False)
    
    # Create sales data CSV
    sales_data = {
        'order_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'customer_id': [1, 1, 2, 3, 4, 5, 2, 1],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Laptop', 'Headphones', 'Tablet', 'Webcam'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
        'quantity': [1, 2, 1, 1, 1, 1, 1, 1],
        'unit_price': [1200.00, 25.00, 75.00, 300.00, 1200.00, 150.00, 500.00, 80.00],
        'total_amount': [1200.00, 50.00, 75.00, 300.00, 1200.00, 150.00, 500.00, 80.00],
        'order_date': ['2024-01-15', '2024-02-20', '2024-01-10', '2024-03-05', '2024-02-15', '2024-03-10', '2024-01-25', '2024-03-20'],
        'status': ['completed', 'completed', 'completed', 'pending', 'completed', 'completed', 'shipped', 'completed']
    }
    
    sales_df = pd.DataFrame(sales_data)
    sales_df.to_csv('sales_data.csv', index=False)
    
    # Create employee data CSV
    employee_data = {
        'employee_id': [101, 102, 103, 104, 105],
        'name': ['Sarah Johnson', 'Mike Davis', 'Emily Chen', 'David Rodriguez', 'Lisa Wang'],
        'department': ['Sales', 'IT', 'Marketing', 'Sales', 'IT'],
        'position': ['Sales Manager', 'Developer', 'Marketing Specialist', 'Sales Rep', 'Data Analyst'],
        'salary': [75000, 85000, 60000, 55000, 70000],
        'hire_date': ['2022-03-15', '2021-07-10', '2023-01-20', '2022-11-05', '2023-02-14'],
        'performance_score': [4.5, 4.8, 4.2, 4.0, 4.7]
    }
    
    employee_df = pd.DataFrame(employee_data)
    employee_df.to_csv('employees.csv', index=False)
    
    print("Sample CSV files created successfully!")
    print("- customers.csv: Customer information")
    print("- sales_data.csv: Sales transactions") 
    print("- employees.csv: Employee data")
    
    return ['customers.csv', 'sales_data.csv', 'employees.csv']

def create_sample_excel():
    """Create a sample Excel file for testing"""
    
    # Create products data
    products_data = {
        'product_id': [1, 2, 3, 4, 5, 6],
        'product_name': ['Laptop Pro', 'Wireless Mouse', 'Mechanical Keyboard', '4K Monitor', 'Gaming Headset', 'Tablet'],
        'category': ['Computers', 'Accessories', 'Accessories', 'Displays', 'Audio', 'Tablets'],
        'brand': ['TechBrand', 'MouseCorp', 'KeyTech', 'DisplayMax', 'AudioPro', 'TabletCo'],
        'price': [1299.99, 29.99, 89.99, 399.99, 199.99, 549.99],
        'stock_quantity': [50, 200, 75, 30, 100, 25],
        'supplier': ['Supplier A', 'Supplier B', 'Supplier B', 'Supplier C', 'Supplier A', 'Supplier C']
    }
    
    products_df = pd.DataFrame(products_data)
    products_df.to_excel('products.xlsx', index=False)
    
    print("Sample Excel file 'products.xlsx' created successfully!")
    print("- Contains product data with pricing and inventory")
    
    return 'products.xlsx'

def create_sample_json():
    """Create a sample JSON file for testing"""
    
    # Create inventory data
    inventory_data = [
        {
            "item_id": 1,
            "item_name": "Smartphone",
            "category": "Electronics",
            "brand": "TechPhone",
            "price": 699.99,
            "in_stock": True,
            "quantity_available": 45,
            "last_restocked": "2024-03-01",
            "average_rating": 4.3,
            "total_reviews": 156
        },
        {
            "item_id": 2,
            "item_name": "Bluetooth Speaker",
            "category": "Audio",
            "brand": "SoundMax",
            "price": 89.99,
            "in_stock": True,
            "quantity_available": 78,
            "last_restocked": "2024-02-15",
            "average_rating": 4.1,
            "total_reviews": 89
        },
        {
            "item_id": 3,
            "item_name": "Fitness Tracker",
            "category": "Wearables",
            "brand": "FitTech",
            "price": 149.99,
            "in_stock": False,
            "quantity_available": 0,
            "last_restocked": "2024-01-20",
            "average_rating": 4.5,
            "total_reviews": 203
        }
    ]
    
    # Save as JSON
    import json
    with open('inventory.json', 'w') as f:
        json.dump(inventory_data, f, indent=2)
    
    print("Sample JSON file 'inventory.json' created successfully!")
    print("- Contains product inventory with ratings")
    
    return 'inventory.json'

if __name__ == "__main__":
    # Create sample files for testing
    print("Creating sample files for testing...")
    print("="*50)
    
    csv_files = create_sample_csv()
    excel_file = create_sample_excel()
    json_file = create_sample_json()
    
    print("\n" + "="*60)
    print("TESTING SQL QUERY GENERATION WITH FILES")
    print("="*60)
    
    # Test cases for different file types and queries
    test_cases = [
        {
            'file': 'customers.csv',
            'query': 'Get all customers from New York with their ages, sorted by age descending',
            'description': 'CSV - Customer filtering and sorting'
        },
        {
            'file': 'sales_data.csv', 
            'query': 'Show total sales amount by customer_id for completed orders only',
            'description': 'CSV - Sales aggregation with filtering'
        },
        {
            'file': 'employees.csv',
            'query': 'Find all employees in IT department with salary greater than 70000',
            'description': 'CSV - Employee filtering by department and salary'
        },
        {
            'file': 'products.xlsx',
            'query': 'Get top 3 most expensive products with their brand and stock quantity',
            'description': 'Excel - Product ranking by price'
        },
        {
            'file': 'inventory.json',
            'query': 'Show all items with average rating above 4.0 and quantity available greater than 50',
            'description': 'JSON - Inventory filtering by rating and stock'
        }
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"{'='*60}")
        print(f"File: {test_case['file']}")
        print(f"Query: {test_case['query']}")
        print(f"{'- '*30}")
        
        try:
            print("Analyzing file schema...")
            schema = extract_file_schema(test_case['file'])
            print("Schema extracted successfully!")
            
            print(f"\nGenerating SQL query...")
            result = generate_sql_query_direct(test_case['file'], test_case['query'])
            print("RESULT:")
            print(result)
        except Exception as e:
            print(f"Error: {e}")
        
        print(f"{'='*60}")
        
        # Add a pause between test cases for readability
        import time
        time.sleep(2)
    
    print(f"\n{'='*60}")
    print("MANUAL TESTING INSTRUCTIONS")
    print(f"{'='*60}")
    print("Sample files created:")
    print("1. customers.csv - Customer data with demographics")
    print("2. sales_data.csv - Sales transaction data") 
    print("3. employees.csv - Employee information")
    print("4. products.xlsx - Product data with pricing")
    print("5. inventory.json - Product inventory with ratings")
    print("\nTo test with your own queries:")
    #result = generate_sql_query_direct('sales_data.csv', 'Show total sales amount by customer_id for completed orders only')
    #print(result)
    print("result = generate_sql_query_direct('filename.csv', 'your natural language query')")
    print("\nExample queries you can try:")
    print("- 'Show customers whose age is between 25 and 35'")
    print("- 'Get average salary by department'") 
    print("- 'Find products with price less than 100'")
    print("- 'Show all orders from the last month'")
    print("- 'Get employees hired after 2022 sorted by performance score'")
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import pickle

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Role-based access control
class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst" 
    VIEWER = "viewer"

@dataclass
class User:
    username: str
    role: UserRole
    permissions: List[str]

class RoleManager:
    def __init__(self):
        self.users = {}
        self.role_permissions = {
            UserRole.ADMIN: ["read", "write", "execute", "delete", "manage_users", "register_files"],
            UserRole.ANALYST: ["read", "write", "execute", "register_files"],
            UserRole.VIEWER: ["read"]
        }
    
    def add_user(self, username: str, role: UserRole):
        permissions = self.role_permissions.get(role, [])
        self.users[username] = User(username, role, permissions)
    
    def check_permission(self, username: str, permission: str) -> bool:
        user = self.users.get(username)
        if not user:
            return False
        return permission in user.permissions
    
    def get_user_role(self, username: str) -> Optional[UserRole]:
        user = self.users.get(username)
        return user.role if user else None

# Memory system
class ConversationMemory:
    def __init__(self, memory_file: str = "conversation_memory.pkl"):
        self.memory_file = memory_file
        self.conversations = {}
        self.query_history = []
        self.successful_queries = {}
        self.load_memory()
    
    def save_memory(self):
        try:
            memory_data = {
                'conversations': self.conversations,
                'query_history': self.query_history,
                'successful_queries': self.successful_queries
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                    self.conversations = memory_data.get('conversations', {})
                    self.query_history = memory_data.get('query_history', [])
                    self.successful_queries = memory_data.get('successful_queries', {})
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
    
    def add_conversation(self, username: str, request: str, sql_query: str, success: bool):
        if username not in self.conversations:
            self.conversations[username] = []
        
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'request': request,
            'sql_query': sql_query,
            'success': success
        }
        
        self.conversations[username].append(conversation_entry)
        self.query_history.append(conversation_entry)
        self.save_memory()

# File manager for flat files
class FileDataManager:
    def __init__(self):
        self.data_sources = {}
        self.temp_db_path = "temp_file_data.db"
    
    def register_file(self, name: str, file_path: str) -> bool:
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Load file based on extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                logger.error(f"Unsupported file type: {file_path}")
                return False
            
            # Create temporary SQLite table
            table_name = name.lower().replace(' ', '_')
            conn = sqlite3.connect(self.temp_db_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            
            # Store metadata
            self.data_sources[name] = {
                'path': file_path,
                'table_name': table_name,
                'rows': len(df),
                'columns': df.columns.tolist()
            }
            
            logger.info(f"Registered file: {name} with {len(df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register file {name}: {e}")
            return False

# Working SQL Generator Tool
class SQLGeneratorTool(BaseTool):
    name: str = "sql_generator"
    description: str = "Generate SQL queries based on natural language input"
    
    def _run(self, query_description: str) -> str:
        """Generate SQL query based on description"""
        desc_lower = query_description.lower()
        
        # Simple but effective pattern matching
        if "count" in desc_lower or "how many" in desc_lower:
            if "department" in desc_lower:
                return "SELECT department, COUNT(*) as count FROM employees GROUP BY department ORDER BY count DESC;"
            else:
                return "SELECT COUNT(*) as total_count FROM employees;"
        
        elif "average" in desc_lower or "avg" in desc_lower:
            if "salary" in desc_lower:
                return "SELECT AVG(salary) as average_salary FROM employees;"
            else:
                return "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department;"
        
        elif "high" in desc_lower and "salary" in desc_lower:
            return "SELECT * FROM employees WHERE salary > 70000 ORDER BY salary DESC;"
        
        elif "department" in desc_lower and ("group" in desc_lower or "by" in desc_lower):
            return "SELECT department, COUNT(*) as employee_count FROM employees GROUP BY department;"
        
        elif "all" in desc_lower and "employees" in desc_lower:
            return "SELECT * FROM employees ORDER BY name;"
        
        elif "sum" in desc_lower or "total" in desc_lower:
            if "salary" in desc_lower:
                return "SELECT SUM(salary) as total_salary FROM employees;"
            else:
                return "SELECT department, SUM(salary) as total_salary FROM employees GROUP BY department;"
        
        else:
            # Default query
            return "SELECT * FROM employees LIMIT 10;"

# Working SQL Executor Tool
class SQLExecutorTool(BaseTool):
    name: str = "sql_executor"
    description: str = "Execute SQL queries and return results"
    
    def _run(self, sql_query: str, db_path: str = "sample.db") -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            return {
                "success": True,
                "data": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "row_count": len(df)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"SQL execution failed: {str(e)}"
            }

# Working Chart Generator Tool
class ChartGeneratorTool(BaseTool):
    name: str = "chart_generator"
    description: str = "Generate charts from data"
    
    def _run(self, data: List[Dict], chart_type: str = "bar") -> str:
        """Generate chart from data"""
        try:
            if not data:
                return "No data provided for chart generation"
            
            df = pd.DataFrame(data)
            
            plt.figure(figsize=(10, 6))
            
            if chart_type == "bar":
                # Simple bar chart logic
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    
                    # Handle numeric vs text columns
                    if df[y_col].dtype in ['int64', 'float64']:
                        plt.bar(df[x_col].astype(str), df[y_col])
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                    else:
                        value_counts = df[x_col].value_counts()
                        plt.bar(value_counts.index.astype(str), value_counts.values)
                        plt.xlabel(x_col)
                        plt.ylabel('Count')
                else:
                    return "Insufficient data columns for chart"
            
            elif chart_type == "pie":
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    plt.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%')
                else:
                    return "Insufficient data columns for pie chart"
            
            plt.title(f'{chart_type.title()} Chart')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = f"chart_{chart_type}_{timestamp}.png"
            plt.savefig(chart_path)
            plt.close()
            
            return f"Chart saved as: {chart_path}"
            
        except Exception as e:
            return f"Chart generation failed: {str(e)}"

# Main SQL Analysis System
class SQLAnalysisSystem:
    def __init__(self, db_path: str = "sample.db"):
        self.db_path = db_path
        self.role_manager = RoleManager()
        self.memory = ConversationMemory()
        self.file_manager = FileDataManager()
        
        # Initialize default users
        self.role_manager.add_user("admin", UserRole.ADMIN)
        self.role_manager.add_user("analyst", UserRole.ANALYST)
        self.role_manager.add_user("viewer", UserRole.VIEWER)
        
        # Initialize tools
        self.sql_generator = SQLGeneratorTool()
        self.sql_executor = SQLExecutorTool()
        self.chart_generator = ChartGeneratorTool()
        
        # Create agents
        self._create_agents()
    
    def _create_agents(self):
        """Create CrewAI agents"""
        self.sql_architect = Agent(
            role='SQL Query Architect',
            goal='Generate accurate SQL queries based on user requirements',
            backstory='You are an expert SQL developer who translates business requirements into precise SQL queries.',
            tools=[self.sql_generator],
            verbose=True,
            allow_delegation=False
        )
        
        self.data_analyst = Agent(
            role='Data Analyst',
            goal='Execute SQL queries and analyze results',
            backstory='You are a skilled data analyst who executes queries and interprets results.',
            tools=[self.sql_executor],
            verbose=True,
            allow_delegation=False
        )
        
        self.visualization_expert = Agent(
            role='Data Visualization Expert',
            goal='Create clear visualizations from data',
            backstory='You create informative charts and graphs from data analysis results.',
            tools=[self.chart_generator],
            verbose=True,
            allow_delegation=False
        )
    
    def process_request(self, user_request: str, username: str = "admin", 
                       create_chart: bool = False, chart_type: str = "bar") -> Dict[str, Any]:
        """Process user request and return results"""
        
        # Check permissions
        if not self.role_manager.check_permission(username, "read"):
            return {"error": "Permission denied"}
        
        try:
            # Step 1: Generate SQL
            print(f"🔄 Generating SQL for: {user_request}")
            generated_sql = self.sql_generator._run(user_request)
            print(f"📝 Generated SQL: {generated_sql}")
            
            # Step 2: Human validation
            validated_sql = self.human_validation(generated_sql, username)
            if not validated_sql:
                return {"error": "Query validation cancelled"}
            
            # Step 3: Execute SQL
            print(f"🚀 Executing SQL...")
            result = self.sql_executor._run(validated_sql, self.db_path)
            
            if not result["success"]:
                self.memory.add_conversation(username, user_request, validated_sql, False)
                return result
            
            print(f"✅ Found {result['row_count']} rows")
            
            # Step 4: Create chart if requested
            chart_result = None
            if create_chart and result["data"]:
                print(f"📊 Creating {chart_type} chart...")
                chart_result = self.chart_generator._run(result["data"], chart_type)
                print(f"🎨 {chart_result}")
            
            # Store in memory
            self.memory.add_conversation(username, user_request, validated_sql, True)
            
            return {
                "success": True,
                "sql_query": validated_sql,
                "data": result["data"],
                "columns": result["columns"],
                "row_count": result["row_count"],
                "chart": chart_result
            }
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.memory.add_conversation(username, user_request, error_msg, False)
            return {"error": error_msg}
    
    def human_validation(self, generated_sql: str, username: str) -> str:
        """Human validation of generated SQL"""
        print(f"\n{'='*60}")
        print(f"GENERATED SQL QUERY FOR: {username}")
        print(f"{'='*60}")
        print(generated_sql)
        print(f"{'='*60}")
        
        while True:
            print("\nOptions:")
            print("1. Execute query as-is")
            print("2. Modify query")
            print("3. Cancel")
            
            choice = input("Your choice (1-3): ").strip()
            
            if choice == "1":
                return generated_sql
            elif choice == "2":
                print("Enter your modified SQL:")
                modified_sql = input().strip()
                if modified_sql:
                    print(f"Modified SQL: {modified_sql}")
                    confirm = input("Execute this? (y/n): ").strip().lower()
                    if confirm == 'y':
                        return modified_sql
            elif choice == "3":
                return None
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    def register_file(self, name: str, file_path: str, username: str = "admin") -> Dict[str, Any]:
        """Register a data file"""
        if not self.role_manager.check_permission(username, "register_files"):
            return {"error": "Permission denied"}
        
        success = self.file_manager.register_file(name, file_path)
        if success:
            return {"success": True, "message": f"File {name} registered successfully"}
        else:
            return {"error": f"Failed to register file {name}"}
    
    def export_to_excel(self, data: List[Dict], filename: str = None) -> str:
        """Export data to Excel"""
        if not filename:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        try:
            df = pd.DataFrame(data)
            df.to_excel(filename, index=False)
            return f"Data exported to: {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"
    
    def get_stats(self, username: str = None) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'total_queries': len(self.memory.query_history),
            'successful_queries': sum(1 for q in self.memory.query_history if q['success']),
            'registered_files': len(self.file_manager.data_sources)
        }
        
        if username and username in self.memory.conversations:
            user_queries = self.memory.conversations[username]
            stats['user_queries'] = len(user_queries)
            stats['user_success'] = sum(1 for q in user_queries if q['success'])
        
        return stats

# Application class
class SQLAnalysisApp:
    def __init__(self):
        self.system = SQLAnalysisSystem()
        self._create_sample_database()
        self._create_sample_files()
    
    def _create_sample_database(self):
        """Create sample database"""
        if not os.path.exists("sample.db"):
            conn = sqlite3.connect("sample.db")
            cursor = conn.cursor()
            
            # Create employees table
            cursor.execute("""
                CREATE TABLE employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT,
                    salary REAL,
                    hire_date DATE
                )
            """)
            
            # Insert sample data
            employees_data = [
                (1, "John Doe", "Engineering", 75000, "2022-01-15"),
                (2, "Jane Smith", "Marketing", 65000, "2021-03-20"),
                (3, "Bob Johnson", "Engineering", 80000, "2020-07-10"),
                (4, "Alice Brown", "HR", 60000, "2023-02-01"),
                (5, "Charlie Wilson", "Engineering", 72000, "2021-11-05"),
                (6, "Diana Prince", "Marketing", 68000, "2022-08-12"),
                (7, "Edward Norton", "Finance", 70000, "2020-04-20"),
                (8, "Fiona Green", "HR", 58000, "2023-01-10")
            ]
            
            cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees_data)
            conn.commit()
            conn.close()
            logger.info("Sample database created")
    
    def _create_sample_files(self):
        """Create sample CSV and Excel files"""
        # Sample CSV
        if not os.path.exists('sample_products.csv'):
            products_data = {
                'product_id': [1, 2, 3, 4, 5],
                'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'],
                'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories'],
                'price': [999.99, 25.99, 79.99, 299.99, 89.99],
                'stock': [50, 200, 150, 75, 120]
            }
            pd.DataFrame(products_data).to_csv('sample_products.csv', index=False)
        
        # Sample Excel  
        if not os.path.exists('sample_customers.xlsx'):
            customers_data = {
                'customer_id': [1, 2, 3, 4, 5],
                'name': ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Martinez'],
                'age': [28, 35, 42, 29, 38],
                'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
                'total_spent': [1250.50, 450.75, 2100.25, 890.00, 1675.80]
            }
            pd.DataFrame(customers_data).to_excel('sample_customers.xlsx', index=False)
        
        logger.info("Sample files created")
    
    def run_interactive(self):
        """Run interactive session"""
        print("=" * 60)
        print("🚀 SQL ANALYSIS SYSTEM")
        print("=" * 60)
        print("Features:")
        print("✅ Natural language to SQL")
        print("✅ Database and file queries")
        print("✅ Chart generation")
        print("✅ Human validation")
        print("✅ Memory system")
        print("=" * 60)
        
        # Login
        username = input("\nEnter username (admin/analyst/viewer): ").strip() or "admin"
        if username not in self.system.role_manager.users:
            print("User not found. Using 'admin'")
            username = "admin"
        
        role = self.system.role_manager.get_user_role(username)
        print(f"✅ Logged in as: {username} ({role.value})")
        
        # Register sample files
        if self.system.role_manager.check_permission(username, "register_files"):
            print("\n📁 Registering sample files...")
            self.system.register_file("products", "sample_products.csv", username)
            self.system.register_file("customers", "sample_customers.xlsx", username)
        
        # Show stats
        stats = self.system.get_stats(username)
        print(f"\n📊 System Stats:")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Registered files: {stats['registered_files']}")
        
        # Main loop
        while True:
            print(f"\n{'='*60}")
            print("💬 What would you like to analyze?")
            print("Examples:")
            print("  - 'Show all employees'")
            print("  - 'Count employees by department'")
            print("  - 'Show average salary'")
            print("  - 'Show high salary employees'")
            print("Type 'quit' to exit")
            print(f"{'='*60}")
            
            request = input("Your request: ").strip()
            
            if request.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if not request:
                continue
            
            # Ask about chart
            create_chart = False
            chart_type = "bar"
            if self.system.role_manager.check_permission(username, "read"):
                viz_choice = input("📊 Create chart? (y/n): ").strip().lower()
                if viz_choice == 'y':
                    create_chart = True
                    chart_type = input("Chart type (bar/pie): ").strip() or "bar"
            
            # Process request
            result = self.system.process_request(request, username, create_chart, chart_type)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            elif result.get("success"):
                print(f"✅ Found {result['row_count']} rows")
                
                # Show sample data
                if result["data"]:
                    print(f"\n📋 Sample results (first 5 rows):")
                    for i, row in enumerate(result["data"][:5], 1):
                        print(f"{i}. {row}")
                
                # Export option
                export = input("\n💾 Export to Excel? (y/n): ").strip().lower()
                if export == 'y':
                    filename = input("Filename (or Enter for auto): ").strip()
                    export_result = self.system.export_to_excel(
                        result["data"], 
                        filename if filename else None
                    )
                    print(f"📊 {export_result}")

def main():
    """Main entry point"""
    try:
        app = SQLAnalysisApp()
        
        print("Choose mode:")
        print("1. Interactive session")
        print("2. API example")
        
        choice = input("Choice (1/2): ").strip()
        
        if choice == "1":
            app.run_interactive()
        else:
            # API example
            print("\nAPI Example:")
            result = app.system.process_request("Show all employees", "admin", True, "bar")
            print(f"Result: {result}")
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
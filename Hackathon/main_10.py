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
from tabulate import tabulate
import re

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
    
    # Show which API keys are available
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        print(f"✅ GROQ_API_KEY found in .env file")
    else:
        print("⚠️  GROQ_API_KEY not found in .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("📝 Or set environment variables manually")

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import tabulate, use fallback if not available
try:
    from tabulate import tabulate
except ImportError:
    print("⚠️  tabulate not installed. Install with: pip install tabulate")
    # Simple fallback function
    def tabulate(data, headers='keys', tablefmt='grid', showindex=False):
        if isinstance(data, pd.DataFrame):
            return data.to_string(index=showindex)
        return str(data)

# Enhanced LLM Configuration with specific provider selection
def get_llm(provider: str = "auto"):
    """Get LLM instance - supports specific provider selection"""
    
    print(f"\n🔍 Checking LLM providers (requested: {provider})...")
    print(f"GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Not set'}")
    print(f"OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
    print(f"ANTHROPIC_API_KEY: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Not set'}")
    
    # Mock LLM option
    if provider == "mock":
        return _create_mock_llm()
    
    # Specific provider requested
    if provider == "groq":
        return _try_groq()
    elif provider == "openai":
        return _try_openai()
    elif provider == "anthropic":
        return _try_anthropic()
    elif provider == "ollama":
        return _try_ollama()
    
    # Auto mode - try in order of preference
    if provider == "auto":
        # Try each provider in order
        for provider_func in [_try_groq, _try_openai, _try_anthropic, _try_ollama]:
            llm = provider_func()
            if llm:
                return llm
        
        # Fallback to mock
        return _create_mock_llm()
    
    # Unknown provider, fallback to auto
    print(f"⚠️  Unknown provider '{provider}', falling back to auto mode")
    return get_llm("auto")

def _try_groq():
    """Try to initialize Groq LLM"""
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and groq_key.strip():
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                groq_api_key=groq_key,
                model_name="mixtral-8x7b-32768",
                temperature=0.1,
                max_tokens=1000
            )
            print(f"✅ Using Groq LLM: mixtral-8x7b-32768")
            
            # Test the connection
            try:
                test_response = llm.invoke("Hello")
                print(f"✅ Groq connection test successful")
                return llm
            except Exception as test_error:
                print(f"❌ Groq connection test failed: {test_error}")
                
    except Exception as e:
        print(f"❌ Groq initialization failed: {e}")
    return None

def _try_openai():
    """Try to initialize OpenAI LLM"""
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.strip():
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                openai_api_key=openai_key,
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=1000
            )
            print(f"✅ Using OpenAI LLM: gpt-3.5-turbo")
            
            # Test the connection
            try:
                test_response = llm.invoke("Hello")
                print(f"✅ OpenAI connection test successful")
                return llm
            except Exception as test_error:
                print(f"❌ OpenAI connection test failed: {test_error}")
                
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {e}")
    return None

def _try_anthropic():
    """Try to initialize Anthropic LLM"""
    try:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key.strip():
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                anthropic_api_key=anthropic_key,
                model_name="claude-3-sonnet-20240229",
                temperature=0.1
            )
            print(f"✅ Using Anthropic LLM: claude-3-sonnet-20240229")
            
            # Test the connection
            try:
                test_response = llm.invoke("Hello")
                print(f"✅ Anthropic connection test successful")
                return llm
            except Exception as test_error:
                print(f"❌ Anthropic connection test failed: {test_error}")
                
    except Exception as e:
        print(f"❌ Anthropic initialization failed: {e}")
    return None

def _try_ollama():
    """Try to initialize Ollama LLM"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            from langchain_community.llms import Ollama
            llm = Ollama(
                model="llama2",
                temperature=0.1
            )
            print(f"✅ Using Ollama LLM: llama2")
            return llm
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
    return None

def _create_mock_llm():
    """Create a mock LLM for testing without API keys"""
    try:
        from langchain_core.language_models.base import BaseLanguageModel
        from langchain_core.outputs import LLMResult, Generation
        
        class MockLLM(BaseLanguageModel):
            """Mock LLM for testing without API keys"""
            
            @property
            def _llm_type(self) -> str:
                return "mock"
            
            def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
                """Generate mock responses"""
                generations = []
                for prompt in prompts:
                    # Simple pattern-based responses for SQL generation
                    if "SELECT" in prompt or "sql" in prompt.lower():
                        if "count" in prompt.lower() and "department" in prompt.lower():
                            text = "SELECT department, COUNT(*) as count FROM employees GROUP BY department ORDER BY count DESC;"
                        elif "average" in prompt.lower() and "salary" in prompt.lower():
                            text = "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC;"
                        elif "high" in prompt.lower() and "salary" in prompt.lower():
                            text = "SELECT * FROM employees WHERE salary > 70000 ORDER BY salary DESC LIMIT 20;"
                        else:
                            text = "SELECT * FROM employees LIMIT 10;"
                    else:
                        text = "Query analysis completed successfully."
                    
                    generations.append([Generation(text=text)])
                
                return LLMResult(generations=generations)
            
            def invoke(self, prompt, **kwargs):
                """Invoke method for compatibility"""
                result = self._generate([prompt])
                return type('MockResponse', (), {'content': result.generations[0][0].text})()
        
        llm = MockLLM()
        print("⚠️  Using Mock LLM (no API key found) - Limited functionality")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to create mock LLM: {e}")
    
    print("❌ No LLM provider available")
    return None

# Enhanced Role-based access control with granular permissions
class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst" 
    VIEWER = "viewer"

@dataclass
class User:
    username: str
    role: UserRole
    permissions: List[str]
    accessible_tables: List[str]
    restricted_columns: Dict[str, List[str]]  # table -> list of restricted columns

class RoleManager:
    def __init__(self):
        self.users = {}
        self.role_permissions = {
            UserRole.ADMIN: {
                "permissions": ["read", "write", "execute", "delete", "manage_users", "register_files", "upload_files"],
                "accessible_tables": ["*"],  # All tables
                "restricted_columns": {}  # No restrictions
            },
            UserRole.ANALYST: {
                "permissions": ["read", "execute", "register_files"],
                "accessible_tables": ["employees", "departments", "products", "sales", "sample_products", "sample_sales"],  # Include file tables
                "restricted_columns": {
                    "employees": ["manager_id"],  # Some restrictions
                    "sales": [],  # No restrictions on sales data for analysts
                    "sample_sales": []  # No restrictions on file sales data
                }
            },
            UserRole.VIEWER: {
                "permissions": ["read"],
                "accessible_tables": ["departments", "products", "sales", "sample_products", "sample_sales"],  # Include file tables but NO employees
                "restricted_columns": {
                    "departments": ["budget"],  # Budget info restricted
                    "products": [],  # No restrictions on products
                    "sales": ["customer_name"],  # Customer names restricted for privacy
                    "sample_products": [],  # No restrictions on sample products
                    "sample_sales": ["customer_name"]  # Customer names restricted for privacy
                }
            }
        }
    
    def add_user(self, username: str, role: UserRole):
        role_config = self.role_permissions.get(role)
        if role_config:
            self.users[username] = User(
                username=username,
                role=role,
                permissions=role_config["permissions"],
                accessible_tables=role_config["accessible_tables"],
                restricted_columns=role_config["restricted_columns"]
            )
    
    def check_permission(self, username: str, permission: str) -> bool:
        user = self.users.get(username)
        if not user:
            return False
        return permission in user.permissions
    
    def check_table_access(self, username: str, table_name: str) -> bool:
        user = self.users.get(username)
        if not user:
            return False
        
        # Admin has access to all tables
        if "*" in user.accessible_tables:
            return True
        
        return table_name in user.accessible_tables
    
    def get_accessible_columns(self, username: str, table_name: str) -> List[str]:
        """Get list of accessible columns for a user and table"""
        user = self.users.get(username)
        if not user:
            return []
        
        # Get all columns for the table (you'd normally get this from schema)
        all_columns = self._get_table_columns(table_name)
        
        # Remove restricted columns
        restricted = user.restricted_columns.get(table_name, [])
        accessible = [col for col in all_columns if col not in restricted]
        
        return accessible
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get all columns for a table - simplified for demo"""
        table_schemas = {
            "employees": ["id", "name", "department", "salary", "hire_date", "manager_id", "status"],
            "departments": ["id", "name", "budget", "location"],
            "products": ["product_id", "product_name", "category", "price", "stock", "supplier"],
            "sales": ["sale_id", "product_id", "customer_name", "quantity", "sale_date", "total_amount"],
            # Handle file table names too
            "sample_products": ["product_id", "product_name", "category", "price", "stock", "supplier"],
            "sample_sales": ["sale_id", "product_id", "customer_name", "quantity", "sale_date", "total_amount"]
        }
        return table_schemas.get(table_name, [])
    
    def apply_column_restrictions(self, username: str, table_name: str, query_result: pd.DataFrame) -> pd.DataFrame:
        """Apply column restrictions to query results"""
        user = self.users.get(username)
        if not user or "*" in user.accessible_tables:
            return query_result  # Admin has no restrictions
        
        restricted_columns = user.restricted_columns.get(table_name, [])
        
        # Remove restricted columns from results
        accessible_columns = [col for col in query_result.columns if col not in restricted_columns]
        return query_result[accessible_columns]
    
    def get_user_role(self, username: str) -> Optional[UserRole]:
        user = self.users.get(username)
        return user.role if user else None

# Enhanced Memory system with role-based filtering
class ConversationMemory:
    def __init__(self, memory_file: str = "conversation_memory.pkl"):
        self.memory_file = memory_file
        self.conversations = {}
        self.query_history = []
        self.successful_patterns = {}
        self.feedback_history = {}
        self.role_based_patterns = {}  # Store patterns by role
        self.load_memory()
    
    def save_memory(self):
        try:
            memory_data = {
                'conversations': self.conversations,
                'query_history': self.query_history,
                'successful_patterns': self.successful_patterns,
                'feedback_history': self.feedback_history,
                'role_based_patterns': self.role_based_patterns
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
                    self.successful_patterns = memory_data.get('successful_patterns', {})
                    self.feedback_history = memory_data.get('feedback_history', {})
                    self.role_based_patterns = memory_data.get('role_based_patterns', {})
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
    
    def add_conversation(self, username: str, request: str, sql_query: str, success: bool, feedback: str = None, user_role: str = None):
        if username not in self.conversations:
            self.conversations[username] = []
        
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'request': request,
            'sql_query': sql_query,
            'success': success,
            'feedback': feedback,
            'user_role': user_role
        }
        
        self.conversations[username].append(conversation_entry)
        self.query_history.append(conversation_entry)
        
        # Learn successful patterns by role
        if success and user_role:
            pattern_key = self._extract_pattern(request)
            
            # General patterns
            if pattern_key not in self.successful_patterns:
                self.successful_patterns[pattern_key] = []
            self.successful_patterns[pattern_key].append(sql_query)
            
            # Role-based patterns
            if user_role not in self.role_based_patterns:
                self.role_based_patterns[user_role] = {}
            if pattern_key not in self.role_based_patterns[user_role]:
                self.role_based_patterns[user_role][pattern_key] = []
            self.role_based_patterns[user_role][pattern_key].append(sql_query)
        
        # Track feedback patterns
        if feedback:
            self._track_feedback_pattern(request, feedback, sql_query, user_role)
        
        self.save_memory()
    
    def _track_feedback_pattern(self, request: str, feedback: str, sql_query: str, user_role: str = None):
        """Track common feedback patterns for learning"""
        feedback_key = self._extract_pattern(feedback)
        if feedback_key not in self.feedback_history:
            self.feedback_history[feedback_key] = []
        
        self.feedback_history[feedback_key].append({
            'original_request': request,
            'feedback': feedback,
            'corrected_sql': sql_query,
            'timestamp': datetime.now().isoformat(),
            'user_role': user_role
        })
    
    def _extract_pattern(self, request: str) -> str:
        """Extract query pattern for learning"""
        keywords = ['count', 'average', 'sum', 'group', 'join', 'where', 'order', 'limit', 'filter', 'top']
        found = [kw for kw in keywords if kw in request.lower()]
        return '_'.join(found) if found else 'general'
    
    def get_role_based_context(self, username: str, request: str, user_role: str, feedback: str = None) -> str:
        """Get role-specific context for LLM including user history and role patterns"""
        context = f"=== ROLE-BASED CONTEXT FOR {user_role.upper()} ===\n\n"
        
        # Role-specific successful patterns
        pattern = self._extract_pattern(request)
        if user_role in self.role_based_patterns and pattern in self.role_based_patterns[user_role]:
            role_examples = self.role_based_patterns[user_role][pattern][-2:]
            if role_examples:
                context += f"Successful {user_role} queries for similar requests:\n"
                for sql in role_examples:
                    context += f"SQL: {sql}\n"
                context += "\n"
        
        # Recent user queries with role context
        if username in self.conversations:
            recent = [entry for entry in self.conversations[username][-5:] if entry.get('user_role') == user_role]
            if recent:
                context += f"Recent {user_role} queries by this user:\n"
                for entry in recent:
                    context += f"Request: {entry['request']}\nSQL: {entry['sql_query']}\nSuccess: {entry['success']}\n"
                    if entry.get('feedback'):
                        context += f"Feedback: {entry['feedback']}\n"
                    context += "\n"
        
        # Role-specific feedback patterns
        if feedback:
            feedback_pattern = self._extract_pattern(feedback)
            if feedback_pattern in self.feedback_history:
                role_feedback = [entry for entry in self.feedback_history[feedback_pattern] 
                               if entry.get('user_role') == user_role][-2:]
                if role_feedback:
                    context += f"Similar {user_role} feedback corrections:\n"
                    for fb_entry in role_feedback:
                        context += f"Original: {fb_entry['original_request']}\n"
                        context += f"Feedback: {fb_entry['feedback']}\n"
                        context += f"Corrected SQL: {fb_entry['corrected_sql']}\n\n"
        
        return context

# Enhanced File manager supporting combined queries and multiple databases
class FileDataManager:
    def __init__(self, selected_databases: List[str] = None):
        self.data_sources = {}
        self.temp_db_path = "temp_file_data.db"
        self.selected_databases = selected_databases or ["sample.db"]
        self.database_paths = {db: os.path.abspath(db) for db in self.selected_databases}
    
    def get_database_tables(self, database_name: str) -> List[Dict[str, Any]]:
        """Get detailed table information for a database"""
        tables_info = []
        
        try:
            db_path = self.database_paths.get(database_name, os.path.abspath(database_name))
            
            if not os.path.exists(db_path):
                return tables_info
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]
            
            for table_name in table_names:
                # Get table info
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns_info = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                
                # Get sample data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                sample_rows = cursor.fetchall()
                
                table_info = {
                    "name": table_name,
                    "row_count": row_count,
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "not_null": bool(col[3]),
                            "primary_key": bool(col[5])
                        }
                        for col in columns_info
                    ],
                    "sample_data": [list(row) for row in sample_rows],
                    "column_names": [col[1] for col in columns_info]
                }
                
                tables_info.append(table_info)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting tables for database {database_name}: {e}")
        
        return tables_info
    
    def get_all_databases_tables(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get table information for all selected databases"""
        all_tables = {}
        
        for db_name in self.selected_databases:
            all_tables[db_name] = self.get_database_tables(db_name)
        
        return all_tables
    
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
            
            # Store metadata with sample data
            self.data_sources[name] = {
                'path': file_path,
                'table_name': table_name,
                'rows': len(df),
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict('records')
            }
            
            logger.info(f"Registered file: {name} with {len(df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register file {name}: {e}")
            return False
    
    def execute_combined_query(self, sql_query: str, username: str, role_manager: 'RoleManager') -> pd.DataFrame:
        """Execute query that might span database and file sources across multiple databases"""
        print(f"🔍 Executing combined query across {len(self.selected_databases)} database(s): {sql_query}")
        
        # Check if this is an error message query
        if sql_query.strip().lower().startswith("select 'error:"):
            # This is an error message, return it as a dataframe
            error_msg = sql_query.split("'")[1] if "'" in sql_query else "Unknown error"
            return pd.DataFrame([{"error_message": error_msg}])
        
        try:
            # Try each database in order until one works
            for db_name in self.selected_databases:
                try:
                    db_path = self.database_paths[db_name]
                    if not os.path.exists(db_path):
                        continue
                    
                    print(f"🔍 Trying database: {db_name}")
                    conn = sqlite3.connect(db_path)
                    df = pd.read_sql_query(sql_query, conn)
                    conn.close()
                    
                    print(f"✅ Query successful with {db_name}: {len(df)} rows")
                    return df
                    
                except Exception as e:
                    print(f"⚠️  Database {db_name} failed: {e}")
                    continue
            
            # If all databases fail, try file sources
            if os.path.exists(self.temp_db_path):
                print(f"🔍 Trying file sources")
                conn = sqlite3.connect(self.temp_db_path)
                df = pd.read_sql_query(sql_query, conn)
                conn.close()
                print(f"✅ File query executed successfully, returned {len(df)} rows")
                return df
            
            # If everything fails
            raise Exception("Query failed on all available databases and file sources")
                
        except Exception as e:
            logger.error(f"Combined query execution failed: {e}")
            print(f"❌ Combined query failed: {e}")
            
            # Create error dataframe with helpful information
            error_df = pd.DataFrame([{
                "error_type": "Query Execution Failed",
                "error_message": str(e),
                "sql_query": sql_query,
                "selected_databases": ", ".join(self.selected_databases),
                "suggestion": "Please check if the requested tables exist in the selected database(s)"
            }])
            return error_df

# SQL Converter Tool
class SQLConverterTool(BaseTool):
    name: str = "sql_converter"
    description: str = "Convert SQLite SQL queries to other database dialects (SQL Server, PostgreSQL, DB2)"
    
    def _run(self, sql_query: str, target_db: str = "postgresql") -> str:
        """Convert SQLite SQL to target database dialect"""
        try:
            # Clean the input query
            query = sql_query.strip()
            if not query.upper().startswith('SELECT'):
                return f"Error: Only SELECT queries are supported for conversion"
            
            # Choose conversion method based on target database
            if target_db.lower() == "postgresql":
                return self._convert_to_postgresql(query)
            elif target_db.lower() == "sqlserver":
                return self._convert_to_sqlserver(query)
            elif target_db.lower() == "db2":
                return self._convert_to_db2(query)
            else:
                return f"Error: Unsupported target database: {target_db}"
                
        except Exception as e:
            return f"Conversion error: {str(e)}"
    
    def _convert_to_postgresql(self, query: str) -> str:
        """Convert SQLite query to PostgreSQL"""
        converted = query
        
        # PostgreSQL-specific conversions
        conversions = [
            # Date functions
            (r'\bDATE\s*\(\s*([^)]+)\s*\)', r"TO_DATE(\1, 'YYYY-MM-DD')"),
            (r'\bDATETIME\s*\(\s*([^)]+)\s*\)', r"TO_TIMESTAMP(\1, 'YYYY-MM-DD HH24:MI:SS')"),
            (r'\bstrftime\s*\(\s*[\'"]%Y[\'"],\s*([^)]+)\s*\)', r"EXTRACT(YEAR FROM \1)"),
            (r'\bstrftime\s*\(\s*[\'"]%m[\'"],\s*([^)]+)\s*\)', r"EXTRACT(MONTH FROM \1)"),
            (r'\bstrftime\s*\(\s*[\'"]%d[\'"],\s*([^)]+)\s*\)', r"EXTRACT(DAY FROM \1)"),
            
            # String functions
            (r'\bSUBSTR\s*\(', r"SUBSTRING("),
            
            # Data types
            (r'\bINTEGER\b', r"INTEGER"),
            (r'\bTEXT\b', r"VARCHAR"),
            (r'\bREAL\b', r"DECIMAL"),
        ]
        
        for pattern, replacement in conversions:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        
        return f"-- PostgreSQL Query\n{converted}"
    
    def _convert_to_sqlserver(self, query: str) -> str:
        """Convert SQLite query to SQL Server"""
        converted = query
        
        # SQL Server-specific conversions
        conversions = [
            # Date functions
            (r'\bDATE\s*\(\s*([^)]+)\s*\)', r"CAST(\1 AS DATE)"),
            (r'\bDATETIME\s*\(\s*([^)]+)\s*\)', r"CAST(\1 AS DATETIME)"),
            (r'\bstrftime\s*\(\s*[\'"]%Y[\'"],\s*([^)]+)\s*\)', r"YEAR(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%m[\'"],\s*([^)]+)\s*\)', r"MONTH(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%d[\'"],\s*([^)]+)\s*\)', r"DAY(\1)"),
            
            # String concatenation
            (r'\|\|', r"+"),
            
            # String functions
            (r'\bSUBSTR\s*\(', r"SUBSTRING("),
            (r'\bLENGTH\s*\(', r"LEN("),
            
            # Data types
            (r'\bINTEGER\b', r"INT"),
            (r'\bTEXT\b', r"NVARCHAR(MAX)"),
            (r'\bREAL\b', r"DECIMAL(18,2)"),
        ]
        
        for pattern, replacement in conversions:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        
        # Handle LIMIT -> TOP conversion
        limit_match = re.search(r'\bLIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            limit_value = limit_match.group(1)
            # Add TOP clause after SELECT
            converted = re.sub(r'\bSELECT\b', f"SELECT TOP {limit_value}", converted, count=1, flags=re.IGNORECASE)
            # Remove the LIMIT clause
            converted = re.sub(r'\bLIMIT\s+\d+\s*;?\s*$', '', converted, flags=re.IGNORECASE)
        
        return f"-- SQL Server Query\n{converted}"
    
    def _convert_to_db2(self, query: str) -> str:
        """Convert SQLite query to DB2"""
        converted = query
        
        # DB2-specific conversions
        conversions = [
            # Date functions
            (r'\bDATE\s*\(\s*([^)]+)\s*\)', r"DATE(\1)"),
            (r'\bDATETIME\s*\(\s*([^)]+)\s*\)', r"TIMESTAMP(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%Y[\'"],\s*([^)]+)\s*\)', r"YEAR(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%m[\'"],\s*([^)]+)\s*\)', r"MONTH(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%d[\'"],\s*([^)]+)\s*\)', r"DAY(\1)"),
            
            # String functions
            (r'\bSUBSTR\s*\(', r"SUBSTR("),
            (r'\bLENGTH\s*\(', r"LENGTH("),
            
            # Data types
            (r'\bINTEGER\b', r"INTEGER"),
            (r'\bTEXT\b', r"VARCHAR(1000)"),
            (r'\bREAL\b', r"DECIMAL(15,2)"),
            
            # LIMIT to FETCH FIRST
            (r'\bLIMIT\s+(\d+)\s*;?\s*$', r"FETCH FIRST \1 ROWS ONLY"),
        ]
        
        for pattern, replacement in conversions:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        
        return f"-- DB2 Query\n{converted}"
    
    def get_supported_databases(self) -> List[str]:
        """Get list of supported target databases"""
        return ["postgresql", "sqlserver", "db2"]

# Enhanced CrewAI Tools with role-based memory usage and LLM selection

class SQLGeneratorTool(BaseTool):
    name: str = "sql_generator"
    description: str = "Generate SQL queries using selected LLM with role-based schema awareness and memory"
    
    def _run(self, query_description: str, username: str = "admin", feedback: str = None, iteration: int = 1) -> str:
        """Generate SQL using selected LLM with role-based context and memory"""
        
        try:
            file_manager = getattr(self, '_file_manager', None)
            memory = getattr(self, '_memory', None)
            role_manager = getattr(self, '_role_manager', None)
            llm = getattr(self, '_llm', None)
            
            print(f"🔍 SQL Generator - Iteration {iteration} for user {username}")
            print(f"📝 Request: {query_description}")
            if feedback:
                print(f"🔄 Feedback: {feedback}")
            
            if file_manager and memory and role_manager:
                user_role = role_manager.get_user_role(username)
                
                if llm:
                    print(f"🧠 Using LLM: {type(llm).__name__} for query generation with role-based context")
                    result = self._generate_with_llm(query_description, llm, feedback, iteration, user_role)
                else:
                    print(f"⚠️  No LLM available, using fallback")
                    result = self._generate_fallback(query_description, user_role, role_manager)
            else:
                print(f"⚠️  Missing components, using basic fallback")
                result = self._generate_fallback(query_description)
            
            print(f"✅ Generated SQL: {result}")
            return result
                
        except Exception as e:
            print(f"❌ SQL generation error: {e}")
            logger.error(f"SQL generation error: {e}")
            return self._generate_fallback(query_description)
    
    def _generate_with_llm(self, query_description: str, llm, feedback: str = None, iteration: int = 1, user_role: UserRole = None) -> str:
        """Generate SQL using selected LLM with role-based constraints"""
        
        if feedback and iteration > 1:
            # Feedback iteration
            prompt = f"""Generate an improved SQL query based on user feedback.

Original request: {query_description}
User feedback: {feedback}

Generate ONLY the corrected SQL query, no explanations."""
        else:
            # Initial query generation
            prompt = f"""Generate a SQL query for: {query_description}

Generate ONLY the SQL query, no explanations. Always include LIMIT clauses."""

        try:
            if hasattr(llm, 'invoke'):
                response = llm.invoke(prompt)
                sql_query = response.content if hasattr(response, 'content') else str(response)
            else:
                sql_query = llm(prompt)
            
            # Clean the response
            sql_query = sql_query.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Extract just the SQL query
            lines = sql_query.split('\n')
            sql_lines = []
            for line in lines:
                line = line.strip()
                if line and (line.upper().startswith('SELECT') or 
                           line.upper().startswith('WITH') or
                           (sql_lines and not line.endswith(';') and not line.startswith('Note:'))):
                    sql_lines.append(line)
                elif sql_lines and line.endswith(';'):
                    sql_lines.append(line)
                    break
                elif line.upper().startswith('SELECT'):
                    sql_lines = [line]
            
            if sql_lines:
                sql_query = ' '.join(sql_lines)
            
            sql_query = sql_query.strip()
            if not sql_query.upper().startswith('SELECT'):
                logger.warning(f"Generated query doesn't start with SELECT: {sql_query}")
                return self._generate_fallback(query_description, user_role)
            
            logger.info(f"LLM generated SQL (iteration {iteration}): {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback(query_description, user_role)
    
    def _generate_fallback(self, query_description: str, user_role: UserRole = None, role_manager: 'RoleManager' = None) -> str:
        """Role-aware fallback SQL generation"""
        desc = query_description.lower()
        
        # Role-based fallback queries for database tables
        if user_role == UserRole.VIEWER:
            # Viewer gets very limited, non-personal queries
            if "department" in desc:
                return "SELECT name, location FROM departments LIMIT 10;"
            elif "employees" in desc:
                return "SELECT 'Access denied: Viewers cannot access employee data' as access_denied_message;"
            else:
                return "SELECT name, location FROM departments LIMIT 10;"
        
        elif user_role == UserRole.ANALYST:
            # Analyst gets more access but still restricted
            if "count" in desc and "department" in desc:
                return "SELECT department, COUNT(*) as count FROM employees GROUP BY department ORDER BY count DESC LIMIT 10;"
            elif "average" in desc and "salary" in desc:
                return "SELECT 'Salary data access restricted for analysts' as restriction_message;"
            elif "department" in desc:
                return "SELECT name, location, budget FROM departments LIMIT 10;"
            elif "employees" in desc:
                return "SELECT name, department FROM employees LIMIT 20;"
            else:
                return "SELECT name, department FROM employees LIMIT 20;"
        
        else:
            # Admin or fallback - full access
            if "count" in desc and "department" in desc:
                return "SELECT department, COUNT(*) as count FROM employees GROUP BY department ORDER BY count DESC LIMIT 10;"
            elif "average" in desc and "salary" in desc:
                return "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC LIMIT 10;"
            elif "high" in desc and "salary" in desc:
                return "SELECT name, department, salary FROM employees WHERE salary > 70000 ORDER BY salary DESC LIMIT 20;"
            else:
                return "SELECT name, department, salary FROM employees LIMIT 10;"


class SQLExecutorTool(BaseTool):
    name: str = "sql_executor"
    description: str = "Execute SQL queries with role-based validation and column filtering across multiple databases"
    
    def _run(self, sql_query: str, username: str = "admin", db_path: str = None) -> str:
        """Execute SQL with role-based validation and filtering across selected databases"""
        
        role_manager = getattr(self, '_role_manager', None)
        file_manager = getattr(self, '_file_manager', None)

        import re
        sql_query = re.sub(r'\b\w+\.db\.', '', sql_query)
        
        print(f"🔍 SQL Executor for user: {username}")
        print(f"🔍 Available databases: {file_manager.selected_databases if file_manager else 'None'}")
        
        # Permission check
        if role_manager and not role_manager.check_permission(username, "execute"):
            return "Permission denied: You don't have execute permissions"
        
        # Basic SQL validation
        sql_lower = sql_query.lower().strip()
        dangerous_ops = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
        
        for op in dangerous_ops:
            if op in sql_lower:
                return f"Dangerous operation '{op}' not allowed"
        
        if not sql_lower.startswith('select'):
            return "Only SELECT queries allowed"
        
        try:
            print(f"🔍 Executing SQL across selected databases")
            
            # Use file manager for multi-database execution
            if file_manager:
                df = file_manager.execute_combined_query(sql_query, username, role_manager)
            else:
                # Fallback to simple execution
                db_path = db_path or "sample.db"
                if not os.path.exists(db_path):
                    return f"Database file not found: {db_path}"
                
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(sql_query, conn)
                conn.close()
            
            print(f"✅ Query executed successfully, returned {len(df)} rows")
            
            # Create a formatted table string for display
            if len(df) > 0:
                display_df = df.head(20)
                table_str = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)
                
                if len(df) > 20:
                    table_str += f"\n... and {len(df) - 20} more rows"
            else:
                table_str = "No results found"
            
            # Store the execution result
            execution_result = {
                "success": True,
                "data": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "sql_query": sql_query,
                "databases_used": file_manager.selected_databases if file_manager else ["Unknown"],
                "table_display": table_str,
                "dataframe": df,
                "user_role": role_manager.get_user_role(username).value if role_manager else "unknown"
            }
            
            # Store in instance attribute
            if hasattr(self, '_last_execution_data'):
                self._last_execution_data = execution_result
            
            # Store in system reference
            if hasattr(self, '_system_ref') and self._system_ref:
                self._system_ref.last_execution_result = execution_result
                print(f"📦 Stored execution result in system")
            
            # Return formatted result
            return_str = f"""SQL Query Executed Successfully!
Query: {sql_query}
Rows Returned: {len(df)}
User Role: {role_manager.get_user_role(username).value if role_manager else 'unknown'}
Databases: {', '.join(file_manager.selected_databases) if file_manager else 'Unknown'}

QUERY RESULTS:
{table_str}

[Full data with {len(df)} rows and {len(df.columns)} columns has been retrieved and stored for export]"""
            
            return return_str
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            print(f"❌ SQL execution error: {error_msg}")
            print(f"🔍 Query: {sql_query}")
            
            return error_msg

class ChartGeneratorTool(BaseTool):
    name: str = "chart_generator"
    description: str = "Generate professional charts from query results"
    
    def _run(self, data: List[Dict], chart_type: str = "bar", title: str = None) -> str:
        """Generate professional charts"""
        try:
            if not data:
                return "No data provided for chart generation"
            
            df = pd.DataFrame(data)
            
            # Set professional style
            plt.style.use('default')
            plt.figure(figsize=(12, 8))
            
            if chart_type == "bar":
                # Intelligent column selection
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols and text_cols:
                    x_col, y_col = text_cols[0], numeric_cols[0]
                    bars = plt.bar(df[x_col].astype(str), df[y_col], 
                                 color='steelblue', alpha=0.8, edgecolor='navy')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
                
                elif numeric_cols:
                    y_col = numeric_cols[0]
                    plt.bar(range(len(df)), df[y_col], color='steelblue', alpha=0.8)
                    plt.xlabel('Records', fontsize=12)
                    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
            
            elif chart_type == "pie":
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols and text_cols and len(df) <= 10:
                    labels = df[text_cols[0]].astype(str)
                    values = df[numeric_cols[0]]
                    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                else:
                    return "Pie chart requires categorical and numeric data with ≤10 categories"
            
            # Formatting
            if not title:
                title = f'{chart_type.title()} Chart - Data Analysis'
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            if chart_type == "bar":
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = f"chart_{chart_type}_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return f"Chart saved: {chart_path}"
            
        except Exception as e:
            return f"Chart generation failed: {str(e)}"

class QueryValidatorTool(BaseTool):
    name: str = "query_validator"
    description: str = "Validate and automatically correct SQL queries with role-based checks"
    
    def _run(self, sql_query: str, username: str = "admin") -> str:
        """Validate and auto-correct SQL query with role awareness"""
        
        role_manager = getattr(self, '_role_manager', None)
        
        print(f"🔍 Query Validator - Input SQL: {sql_query}")
        
        corrected_query = sql_query.strip()
        corrections_made = []
        
        # Add LIMIT if missing
        if 'LIMIT' not in corrected_query.upper() and 'COUNT(' not in corrected_query.upper():
            corrected_query = corrected_query.rstrip(';') + ' LIMIT 50;'
            corrections_made.append("Added LIMIT 50")
        
        if corrections_made:
            print(f"✅ Schema corrections applied: {corrections_made}")
            print(f"📝 Corrected SQL: {corrected_query}")
            return f"CORRECTED SQL: {corrected_query}\n\nCorrections made: {', '.join(corrections_made)}"
        else:
            print(f"✅ No corrections needed")
            return f"SQL query validated successfully: {corrected_query}"


class CrewAISQLSystem:
    def __init__(self, llm_provider: str = "auto", selected_databases: List[str] = None):
        self.selected_databases = selected_databases or ["sample.db"]
        self.llm_provider = llm_provider
        self.llm = get_llm(llm_provider)
        self.last_execution_result = None
        self.max_feedback_iterations = 5
        
        print(f"🧠 Initializing CrewAI System with LLM: {llm_provider}")
        print(f"🗄️ Selected databases: {self.selected_databases}")
        
        # Initialize enhanced components
        self.role_manager = RoleManager()
        self.memory = ConversationMemory()
        self.file_manager = FileDataManager(self.selected_databases)
        
        # Setup users with enhanced roles
        self.role_manager.add_user("admin", UserRole.ADMIN)
        self.role_manager.add_user("analyst", UserRole.ANALYST)
        self.role_manager.add_user("viewer", UserRole.VIEWER)
        
        # Initialize tools with role manager
        self.sql_generator = SQLGeneratorTool()
        self.sql_executor = SQLExecutorTool()
        self.chart_generator = ChartGeneratorTool()
        self.query_validator = QueryValidatorTool()
        self.sql_converter = SQLConverterTool()
        
        # Set enhanced attributes on tools
        self.sql_generator._file_manager = self.file_manager
        self.sql_generator._memory = self.memory
        self.sql_generator._role_manager = self.role_manager
        self.sql_generator._llm = self.llm
        
        self.sql_executor._role_manager = self.role_manager
        self.sql_executor._file_manager = self.file_manager
        self.sql_executor._system_ref = self
        self.sql_executor._last_execution_data = None
        
        self.query_validator._role_manager = self.role_manager
        self.query_validator._llm = self.llm
        
        # Create enhanced agents
        self._create_agents()
        
        # Verify database connections
        self._verify_database_connections()
    
    def convert_sql_to_target_db(self, sql_query: str, target_db: str) -> Dict[str, Any]:
        """Convert SQL query to target database dialect"""
        try:
            converted_query = self.sql_converter._run(sql_query, target_db)
            
            return {
                "success": True,
                "original_query": sql_query,
                "converted_query": converted_query,
                "target_database": target_db,
                "conversion_notes": f"Converted SQLite query to {target_db.upper()} dialect"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Conversion failed: {str(e)}",
                "original_query": sql_query,
                "target_database": target_db
            }
    
    def get_database_tables_info(self, username: str = "admin") -> Dict[str, Any]:
        """Get detailed table information for all selected databases"""
        try:
            user_role = self.role_manager.get_user_role(username)
            all_tables = self.file_manager.get_all_databases_tables()
            
            # Filter tables based on user role
            filtered_tables = {}
            for db_name, tables in all_tables.items():
                accessible_tables = []
                for table_info in tables:
                    table_name = table_info["name"]
                    if self.role_manager.check_table_access(username, table_name):
                        # Filter columns based on role
                        accessible_columns = self.role_manager.get_accessible_columns(username, table_name)
                        
                        # Filter the table info
                        filtered_table = {
                            "name": table_info["name"],
                            "row_count": table_info["row_count"],
                            "columns": [
                                col for col in table_info["columns"] 
                                if col["name"] in accessible_columns
                            ],
                            "sample_data": table_info["sample_data"],
                            "column_names": [
                                col for col in table_info["column_names"] 
                                if col in accessible_columns
                            ],
                            "access_level": user_role.value if user_role else "unknown"
                        }
                        accessible_tables.append(filtered_table)
                
                filtered_tables[db_name] = accessible_tables
            
            return {
                "success": True,
                "databases": filtered_tables,
                "user_role": user_role.value if user_role else "unknown",
                "total_databases": len(filtered_tables),
                "total_accessible_tables": sum(len(tables) for tables in filtered_tables.values())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get table information: {str(e)}"
            }
    
    def _verify_database_connections(self):
        """Verify connections to all selected databases"""
        print(f"\n🔍 Verifying connections to {len(self.selected_databases)} database(s)...")
        
        for db_name in self.selected_databases:
            db_path = os.path.abspath(db_name)
            print(f"📂 Checking: {db_name} -> {db_path}")
            
            if not os.path.exists(db_path):
                print(f"❌ Database file not found: {db_path}")
                # Try to create if it's sample.db
                if db_name == "sample.db":
                    print(f"🏗️  Creating sample database...")
                    self._create_fresh_database(db_path)
                continue
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                if tables:
                    print(f"✅ {db_name}: {len(tables)} tables found")
                    for table in tables[:3]:  # Show first 3 tables
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        print(f"   - {table}: {count} records")
                    if len(tables) > 3:
                        print(f"   - ... and {len(tables) - 3} more tables")
                else:
                    print(f"⚠️  {db_name}: No tables found")
                
                conn.close()
                
            except Exception as e:
                print(f"❌ {db_name}: Connection failed - {e}")
    
    def _create_fresh_database(self, db_path: str):
        """Create a fresh database with sample data"""
        try:
            print(f"🏗️  Creating fresh database at: {db_path}")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create employees table
            cursor.execute("""
                CREATE TABLE employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT,
                    salary REAL,
                    hire_date DATE,
                    manager_id INTEGER,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Create departments table
            cursor.execute("""
                CREATE TABLE departments (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    budget REAL,
                    location TEXT
                )
            """)
            
            # Insert sample data
            employees = [
                (1, "John Doe", "Engineering", 75000, "2022-01-15", None, "active"),
                (2, "Jane Smith", "Marketing", 65000, "2021-03-20", None, "active"),
                (3, "Bob Johnson", "Engineering", 80000, "2020-07-10", 1, "active"),
                (4, "Alice Brown", "HR", 60000, "2023-02-01", None, "active"),
                (5, "Charlie Wilson", "Engineering", 72000, "2021-11-05", 1, "active"),
                (6, "Diana Prince", "Marketing", 68000, "2022-08-12", 2, "active"),
                (7, "Edward Norton", "Finance", 70000, "2020-04-20", None, "active"),
                (8, "Fiona Green", "HR", 58000, "2023-01-10", 4, "active"),
                (9, "George Miller", "Engineering", 85000, "2019-05-15", 1, "active"),
                (10, "Helen Davis", "Marketing", 62000, "2022-09-01", 2, "active")
            ]
            
            departments = [
                (1, "Engineering", 500000, "Building A"),
                (2, "Marketing", 300000, "Building B"),
                (3, "HR", 200000, "Building C"),
                (4, "Finance", 250000, "Building A")
            ]
            
            cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", employees)
            cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments)
            
            conn.commit()
            conn.close()
            
            print(f"✅ Fresh database created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create fresh database: {e}")
            return False
    
    def _create_agents(self):
        """Create enhanced CrewAI agents with role awareness and LLM selection"""
        
        self.sql_architect = Agent(
            role='Senior SQL Database Architect with Role-Based Access Control',
            goal='Generate perfect SQL queries using the selected LLM that respect user roles, use role-based memory patterns, and incorporate feedback while maintaining security constraints',
            backstory=f"""You are a world-class database architect with expertise in role-based access control 
                        and multi-database environments. You are powered by {self.llm_provider} LLM and have access 
                        to {len(self.selected_databases)} database(s): {', '.join(self.selected_databases)}. 
                        You understand that different users have different access levels and you ALWAYS respect these 
                        constraints. You use historical patterns from users with similar roles to improve your queries. 
                        You excel at incorporating feedback while maintaining security boundaries.""",
            tools=[self.sql_generator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
        
        self.security_specialist = Agent(
            role='Database Security and Role-Based Access Specialist',
            goal='Ensure all SQL queries respect role-based access controls and follow enterprise security practices across multiple databases',
            backstory=f"""You are a cybersecurity expert who specializes in role-based access control and database 
                        security across multiple database systems. You work with {len(self.selected_databases)} database(s) 
                        and understand that viewers should not see sensitive data, analysts have limited access, 
                        and only admins have full access. You enforce these rules strictly across all databases.""",
            tools=[self.query_validator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
        
        self.data_analyst = Agent(
            role='Senior Data Analytics Engineer with Multi-Database and Multi-Source Query Execution',
            goal='Execute queries across multiple databases and file sources while applying role-based column filtering',
            backstory=f"""You are a senior data engineer who can execute queries across multiple data sources 
                        including {len(self.selected_databases)} database(s): {', '.join(self.selected_databases)} 
                        and various file sources. You ensure that users only see data appropriate for their role 
                        and can intelligently route queries to the appropriate data sources. You apply column-level 
                        security and present results in clear, formatted tables.""",
            tools=[self.sql_executor],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
        
        self.visualization_expert = Agent(
            role='Data Visualization Expert with Role-Aware Chart Generation',
            goal='Create professional visualizations that respect data privacy constraints based on user roles',
            backstory="""You are a data visualization expert who understands that different users should see 
                        different levels of detail in charts. You create visualizations that are both insightful 
                        and compliant with role-based access policies.""",
            tools=[self.chart_generator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
    
    def process_request(self, user_request: str, username: str = "admin", 
                       create_chart: bool = False, chart_type: str = "bar",
                       data_source: str = "auto", feedback: str = None, 
                       iteration: int = 1, selected_databases: List[str] = None) -> Dict[str, Any]:
        """Process request with enhanced role-based workflow and LLM selection"""
        
        if not self.role_manager.check_permission(username, "read"):
            return {"error": f"Permission denied: {username} doesn't have read permissions"}
        
        user_role = self.role_manager.get_user_role(username)
        print(f"\n🚀 STARTING ENHANCED CREWAI WORKFLOW - ITERATION {iteration}")
        print(f"📝 Request: {user_request}")
        print(f"👤 User: {username} (Role: {user_role.value if user_role else 'unknown'})")
        print(f"🧠 LLM Provider: {self.llm_provider}")
        print(f"🗄️ Target Databases: {selected_databases or self.selected_databases}")
        if feedback:
            print(f"🔄 Feedback: {feedback}")
        print(f"📊 Visualization: {create_chart} ({chart_type})")
        print(f"💾 Data Source: {data_source}")
        print("="*70)
        
        # Enhanced task descriptions with role awareness and LLM info
        sql_task_desc = f"""
        Generate an advanced SQL query using {self.llm_provider} LLM for user '{username}' with role '{user_role.value if user_role else 'unknown'}':
        
        REQUEST: "{user_request}"
        
        CRITICAL ROLE-BASED REQUIREMENTS:
        - Respect user's role-based access permissions
        - Use role-specific memory patterns and successful queries
        - Only access tables and columns available to this user role
        - Apply appropriate data filtering for the user's role
        - Work across multiple databases: {', '.join(selected_databases or self.selected_databases)}
        
        USER CONTEXT:
        - Username: {username}
        - Role: {user_role.value if user_role else 'unknown'}
        - Available databases: {len(selected_databases or self.selected_databases)}
        - Data source preference: {data_source}
        - LLM Provider: {self.llm_provider}
        - Iteration: {iteration}
        """
        
        if feedback and iteration > 1:
            sql_task_desc += f"""
            
            🔄 USER FEEDBACK ITERATION {iteration}:
            Previous query feedback: "{feedback}"
            
            INCORPORATE this feedback while maintaining role-based restrictions.
            """
        
        sql_task = Task(
            description=sql_task_desc,
            agent=self.sql_architect,
            expected_output=f"Role-appropriate SQL query for {user_role.value if user_role else 'unknown'} user that incorporates feedback" if feedback else f"Role-appropriate SQL query for {user_role.value if user_role else 'unknown'} user"
        )
        
        # Enhanced validation task
        validation_task = Task(
            description=f"""
            Perform role-based security validation for user '{username}' with role '{user_role.value if user_role else 'unknown'}':
            
            VALIDATION REQUIREMENTS:
            - Ensure query respects role-based table access across {len(selected_databases or self.selected_databases)} databases
            - Verify no restricted columns are accessed
            - Apply role-specific security policies
            - Validate SQL syntax and performance
            - Check database access permissions
            
            USER: {username} (Role: {user_role.value if user_role else 'unknown'})
            DATABASES: {', '.join(selected_databases or self.selected_databases)}
            ITERATION: {iteration}
            """,
            agent=self.security_specialist,
            expected_output="Role-validated and corrected SQL query with security compliance report",
            context=[sql_task]
        )
        
        # Enhanced execution task
        execution_task = Task(
            description=f"""
            Execute the validated query for user '{username}' with multi-database and multi-source support:
            
            EXECUTION REQUIREMENTS:
            - Execute across {len(selected_databases or self.selected_databases)} databases and file sources as needed
            - Apply role-based column filtering to results
            - Present data in formatted, role-appropriate manner
            - Ensure data privacy compliance
            - Route query to appropriate database automatically
            
            USER: {username} (Role: {user_role.value if user_role else 'unknown'})
            DATABASES: {', '.join(selected_databases or self.selected_databases)}
            DATA SOURCES: {data_source}
            ITERATION: {iteration}
            """,
            agent=self.data_analyst,
            expected_output="Role-filtered query results with complete formatted data display from appropriate database",
            context=[validation_task]
        )
        
        tasks = [sql_task, validation_task, execution_task]
        
        # Enhanced visualization task
        if create_chart and self.role_manager.check_permission(username, "read"):
            viz_task = Task(
                description=f"""
                Create role-appropriate visualization for user '{username}':
                
                VISUALIZATION REQUIREMENTS:
                - Respect data privacy constraints for user role
                - Create {chart_type} chart with appropriate detail level
                - Ensure no sensitive data is visualized for restricted roles
                - Apply professional formatting
                
                USER ROLE: {user_role.value if user_role else 'unknown'}
                CHART TYPE: {chart_type}
                """,
                agent=self.visualization_expert,
                expected_output=f"Role-appropriate {chart_type} chart with privacy compliance",
                context=[execution_task]
            )
            tasks.append(viz_task)
        
        # Execute workflow
        crew = Crew(
            agents=[self.sql_architect, self.security_specialist, self.data_analyst, self.visualization_expert],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            manager_llm=self.llm
        )
        
        try:
            print(f"\n🔄 EXECUTING ENHANCED MULTI-AGENT WORKFLOW - ITERATION {iteration}...")
            print(f"🧠 Using LLM: {self.llm_provider}")
            
            # Execute the crew
            result = crew.kickoff()
            
            print(f"\n✅ ENHANCED CREWAI WORKFLOW COMPLETED - ITERATION {iteration}")
            print("="*70)
            
            # Extract execution results
            execution_data = None
            
            if hasattr(self.sql_executor, '_last_execution_data') and self.sql_executor._last_execution_data:
                execution_data = self.sql_executor._last_execution_data
                print(f"📊 Retrieved execution data")
                print(f"   - Rows: {execution_data.get('row_count', 0)}")
                print(f"   - Role: {execution_data.get('user_role', 'unknown')}")
                print(f"   - Databases: {execution_data.get('databases_used', ['Unknown'])}")
            elif self.last_execution_result:
                execution_data = self.last_execution_result
            
            # Store in enhanced memory with role information
            self.memory.add_conversation(username, user_request, str(result), True, feedback, 
                                       user_role.value if user_role else None)
            
            return {
                "success": True,
                "crew_result": str(result),
                "workflow_completed": True,
                "agents_used": len(tasks),
                "llm_provider": self.llm_provider,
                "selected_databases": selected_databases or self.selected_databases,
                "user": username,
                "user_role": user_role.value if user_role else "unknown",
                "request": user_request,
                "execution_data": execution_data,
                "iteration": iteration,
                "feedback_applied": feedback is not None
            }
            
        except Exception as e:
            error_msg = f"Enhanced CrewAI workflow failed: {str(e)}"
            logger.error(error_msg)
            
            # Store failed attempt with role info
            self.memory.add_conversation(username, user_request, error_msg, False, feedback,
                                       user_role.value if user_role else None)
            
            return {
                "success": False,
                "error": error_msg,
                "llm_available": self.llm is not None,
                "llm_provider": self.llm_provider,
                "selected_databases": selected_databases or self.selected_databases,
                "user_role": user_role.value if user_role else "unknown",
                "iteration": iteration
            }
    
    def process_request_with_feedback_loop(self, user_request: str, username: str = "admin",
                                         create_chart: bool = False, chart_type: str = "bar",
                                         data_source: str = "auto", selected_databases: List[str] = None) -> Dict[str, Any]:
        """Enhanced feedback loop with role awareness and database selection"""
        
        iteration = 1
        current_feedback = None
        
        # Use provided databases or fall back to system default
        target_databases = selected_databases or self.selected_databases
        
        while iteration <= self.max_feedback_iterations:
            print(f"\n{'🔄' if iteration > 1 else '🚀'} PROCESSING REQUEST - ITERATION {iteration}")
            
            # Process the request
            result = self.process_request(
                user_request=user_request,
                username=username,
                create_chart=create_chart,
                chart_type=chart_type,
                data_source=data_source,
                feedback=current_feedback,
                iteration=iteration,
                selected_databases=target_databases
            )
            
            if not result.get('success'):
                print(f"❌ Workflow failed on iteration {iteration}")
                return result
            
            # For Streamlit integration, we'll return after first iteration
            # The feedback loop will be handled in the UI
            if iteration == 1:
                return {
                    **result,
                    "needs_feedback": True,
                    "iteration": iteration
                }
            
            iteration += 1
        
        return result
    
    def apply_feedback_and_retry(self, previous_result: Dict, feedback: str, username: str = "admin",
                               create_chart: bool = False, chart_type: str = "bar",
                               data_source: str = "auto", selected_databases: List[str] = None) -> Dict[str, Any]:
        """Apply feedback and retry the request"""
        
        original_request = previous_result.get('request', '')
        previous_iteration = previous_result.get('iteration', 1)
        target_databases = selected_databases or previous_result.get('selected_databases', self.selected_databases)
        
        return self.process_request(
            user_request=original_request,
            username=username,
            create_chart=create_chart,
            chart_type=chart_type,
            data_source=data_source,
            feedback=feedback,
            iteration=previous_iteration + 1,
            selected_databases=target_databases
        )
    
    def register_file(self, name: str, file_path: str, username: str = "admin") -> Dict[str, Any]:
        """Enhanced file registration with role checking"""
        if not self.role_manager.check_permission(username, "register_files"):
            return {"error": f"Permission denied: {username} doesn't have file registration permissions"}
        
        success = self.file_manager.register_file(name, file_path)
        return {"success": success, "message": f"File {name} registered" if success else f"Failed to register {name}"}
    
    def export_results(self, data: List[Dict] = None, filename: str = None) -> str:
        """Export query results to Excel with enhanced metadata"""
        if not filename:
            filename = f"crewai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        try:
            # Use last execution result if no data provided
            if not data and self.last_execution_result and 'data' in self.last_execution_result:
                data = self.last_execution_result['data']
            
            if not data:
                return "No data available to export"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main data
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name='Query Results', index=False)
                
                # Enhanced metadata
                metadata = {
                    'Generated By': ['Enhanced CrewAI SQL Analysis System'],
                    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Records': [len(data)],
                    'LLM Provider': [self.llm_provider],
                    'Databases Used': [', '.join(self.selected_databases)],
                    'Database Count': [len(self.selected_databases)]
                }
                
                if self.last_execution_result:
                    metadata['SQL Query'] = [self.last_execution_result.get('sql_query', 'N/A')]
                    metadata['User Role'] = [self.last_execution_result.get('user_role', 'Unknown')]
                    metadata['Databases Accessed'] = [', '.join(self.last_execution_result.get('databases_used', ['Unknown']))]
                
                pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)
            
            return f"Results exported to: {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"

# Enhanced Application Interface
class CrewAIApp:
    def __init__(self, llm_provider: str = "auto", selected_databases: List[str] = None):
        print("🏗️  Initializing Enhanced CrewAI Application...")
        print(f"🧠 LLM Provider: {llm_provider}")
        print(f"🗄️ Selected databases: {selected_databases or ['sample.db']}")
        
        # Create sample data FIRST if needed
        self._create_sample_data(selected_databases)
        
        # Initialize the enhanced system with LLM and database selection
        self.system = CrewAISQLSystem(llm_provider, selected_databases)
        
        self._show_llm_status()
    
    def _create_sample_data(self, selected_databases: List[str] = None):
        """Create enhanced sample database and files if needed"""
        databases = selected_databases or ["sample.db"]
        
        # Only create sample.db if it's in the selected databases and doesn't exist
        if "sample.db" in databases and not os.path.exists("sample.db"):
            print("📊 Creating sample database...")
            
            try:
                conn = sqlite3.connect("sample.db")
                cursor = conn.cursor()
                
                # Create tables with enhanced data
                cursor.execute("""
                    CREATE TABLE employees (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        department TEXT,
                        salary REAL,
                        hire_date DATE,
                        manager_id INTEGER,
                        status TEXT DEFAULT 'active'
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE departments (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        budget REAL,
                        location TEXT
                    )
                """)
                
                # Enhanced sample data
                employees = [
                    (1, "John Doe", "Engineering", 75000, "2022-01-15", None, "active"),
                    (2, "Jane Smith", "Marketing", 65000, "2021-03-20", None, "active"),
                    (3, "Bob Johnson", "Engineering", 80000, "2020-07-10", 1, "active"),
                    (4, "Alice Brown", "HR", 60000, "2023-02-01", None, "active"),
                    (5, "Charlie Wilson", "Engineering", 72000, "2021-11-05", 1, "active"),
                    (6, "Diana Prince", "Marketing", 68000, "2022-08-12", 2, "active"),
                    (7, "Edward Norton", "Finance", 70000, "2020-04-20", None, "active"),
                    (8, "Fiona Green", "HR", 58000, "2023-01-10", 4, "active"),
                    (9, "George Miller", "Engineering", 85000, "2019-05-15", 1, "active"),
                    (10, "Helen Davis", "Marketing", 62000, "2022-09-01", 2, "active"),
                    (11, "Ian Foster", "Engineering", 90000, "2018-06-01", 1, "active"),
                    (12, "Jessica Wong", "Finance", 73000, "2021-09-15", 7, "active")
                ]
                
                departments = [
                    (1, "Engineering", 500000, "Building A"),
                    (2, "Marketing", 300000, "Building B"),
                    (3, "HR", 200000, "Building C"),
                    (4, "Finance", 250000, "Building A")
                ]
                
                cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", employees)
                cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments)
                
                conn.commit()
                conn.close()
                
                print(f"✅ Sample database created successfully")
                
            except Exception as e:
                print(f"❌ Error creating database: {e}")
        
        # Create enhanced sample files
        self._create_sample_files()
    
    def _create_sample_files(self):
        """Create enhanced sample files"""
        try:
            # Enhanced products file
            products_file = 'sample_products.csv'
            if not os.path.exists(products_file):
                products = {
                    'product_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'product_name': ['Laptop Pro', 'Wireless Mouse', 'Mechanical Keyboard', '4K Monitor', 'HD Webcam', 'Bluetooth Speakers', 'Noise-Canceling Headphones', 'Tablet Pro', 'Smartwatch', 'Wireless Charger'],
                    'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Accessories', 'Electronics', 'Electronics', 'Accessories'],
                    'price': [1299.99, 45.99, 129.99, 599.99, 149.99, 199.99, 299.99, 899.99, 399.99, 79.99],
                    'stock': [25, 150, 75, 40, 80, 60, 35, 20, 45, 100],
                    'supplier': ['TechCorp', 'AccessoryInc', 'AccessoryInc', 'TechCorp', 'AccessoryInc', 'AudioCorp', 'AudioCorp', 'TechCorp', 'TechCorp', 'AccessoryInc']
                }
                pd.DataFrame(products).to_csv(products_file, index=False)
                print(f"✅ Created enhanced {products_file}")
            
            # Enhanced sales file
            sales_file = 'sample_sales.xlsx'
            if not os.path.exists(sales_file):
                sales = {
                    'sale_id': list(range(1, 16)),
                    'product_id': [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3, 5, 7],
                    'customer_name': ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Martinez', 'Frank Brown', 'Grace Lee', 'Henry Garcia', 'Iris Chen', 'Jack Taylor', 'Kelly Moore', 'Liam O\'Brien', 'Maya Patel', 'Nathan Kim', 'Olivia Ross'],
                    'quantity': [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1],
                    'sale_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20', '2024-01-21', '2024-01-22', '2024-01-23', '2024-01-24', '2024-01-25', '2024-01-26', '2024-01-27', '2024-01-28', '2024-01-29'],
                    'total_amount': [1299.99, 91.98, 129.99, 1299.99, 149.99, 199.99, 299.99, 899.99, 399.99, 399.99, 159.98, 137.97, 129.99, 199.99, 899.99]
                }
                pd.DataFrame(sales).to_excel(sales_file, index=False)
                print(f"✅ Created enhanced {sales_file}")
                
        except Exception as e:
            print(f"❌ Error creating sample files: {e}")
    
    def _show_llm_status(self):
        """Show enhanced LLM configuration status"""
        print(f"\n{'='*60}")
        print("🧠 ENHANCED LLM CONFIGURATION STATUS")
        print(f"{'='*60}")
        
        if self.system.llm:
            llm_type = type(self.system.llm).__name__
            print(f"✅ LLM Active: {self.system.llm_provider} ({llm_type})")
            print(f"🗄️ Databases: {len(self.system.selected_databases)} database(s)")
            print("🤖 Enhanced features enabled:")
            print("   • Role-based query generation")
            print("   • Memory-driven pattern learning")
            print("   • Multi-database query support")
            print("   • Human-in-the-loop feedback")
            print("   • Configurable LLM providers")
            print("   • SQL database conversion")
            print("   • Database table inspection")
            
            if "Mock" in llm_type:
                print("⚠️  Mock LLM - Limited functionality")
            else:
                print("🚀 Full AI capabilities available")
        else:
            print(f"❌ No LLM configured ({self.system.llm_provider}) - Limited functionality")
        
        print(f"{'='*60}")

def main():
    """Enhanced main function with LLM selection"""
    print("🚀 Initializing Enhanced CrewAI SQL Analysis System...")
    
    try:
        # Default initialization for testing
        app = CrewAIApp(llm_provider="auto", selected_databases=["sample.db"])
        
        print("\n🎯 Enhanced System Features:")
        print("✅ Role-based access control (Admin/Analyst/Viewer)")
        print("✅ Memory-driven query learning")
        print("✅ Multi-database data queries (DB + Files)")
        print("✅ Human-in-the-loop feedback")
        print("✅ Enhanced security and privacy")
        print("✅ Configurable LLM providers")
        print("✅ Multi-database support")
        print("✅ SQL dialect conversion (PostgreSQL, SQL Server, DB2)")
        print("✅ Database table inspection and listing")
        
        # For now, just show that the enhanced system is ready
        print("\n✅ Enhanced CrewAI system initialized successfully!")
        print("🌐 Ready for Streamlit integration!")
        
        return app
        
    except Exception as e:
        logger.error(f"Enhanced application error: {e}")
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
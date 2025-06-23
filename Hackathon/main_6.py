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

# LLM Configuration with multiple providers
def get_llm():
    """Get LLM instance - tries multiple providers in order of preference"""
    
    print(f"\n🔍 Checking LLM providers...")
    print(f"GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Not set'}")
    print(f"OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
    print(f"ANTHROPIC_API_KEY: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Not set'}")
    
    # Option 1: Groq API (Fast and reliable)
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
    
    # Option 2: OpenAI API
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
    
    # Option 3: Anthropic Claude
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
    
    # Option 4: Ollama (Local)
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
    
    # Option 5: Create a mock LLM for testing without API keys
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

# Enhanced File manager supporting combined queries
class FileDataManager:
    def __init__(self, db_path: str = "sample.db"):
        self.data_sources = {}
        self.temp_db_path = "temp_file_data.db"
        self.main_db_path = os.path.abspath(db_path)
    
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
    
    def get_role_based_schema_info(self, username: str, role_manager: 'RoleManager') -> str:
        """Get schema information filtered by user role"""
        schema_info = "=== AVAILABLE DATA SOURCES (ROLE-FILTERED) ===\n\n"
        
        # Database tables with role-based filtering
        abs_db_path = self.main_db_path
        
        try:
            if not os.path.exists(abs_db_path):
                return schema_info + f"❌ Database file not found: {abs_db_path}\n"
            
            conn = sqlite3.connect(abs_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Filter tables based on user role
            accessible_tables = [table for table in all_tables 
                               if role_manager.check_table_access(username, table)]
            
            if accessible_tables:
                schema_info += f"DATABASE TABLES (ACCESSIBLE TO YOUR ROLE):\n"
                for table_name in accessible_tables:
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    all_columns = cursor.fetchall()
                    
                    # Get accessible columns for this user
                    accessible_columns = role_manager.get_accessible_columns(username, table_name)
                    
                    schema_info += f"\n=== TABLE: {table_name} ===\n"
                    schema_info += "ACCESSIBLE COLUMNS:\n"
                    for col in all_columns:
                        col_name, col_type = col[1], col[2]
                        if col_name in accessible_columns:
                            pk_indicator = " (PRIMARY KEY)" if col[5] == 1 else ""
                            schema_info += f"  {col_name} ({col_type}){pk_indicator}\n"
                    
                    # Show sample data with column restrictions
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                    sample_data = cursor.fetchall()
                    
                    schema_info += f"\nSAMPLE DATA:\n"
                    if sample_data:
                        schema_info += f"  Headers: {accessible_columns}\n"
                        for i, row in enumerate(sample_data, 1):
                            # Filter row data to accessible columns only
                            all_col_names = [col[1] for col in all_columns]
                            filtered_row = [row[all_col_names.index(col)] for col in accessible_columns 
                                          if col in all_col_names]
                            schema_info += f"  Row {i}: {filtered_row}\n"
                    schema_info += "\n"
            
            conn.close()
        except Exception as e:
            schema_info += f"❌ Error reading database schema: {e}\n"
        
        # File sources (also apply role-based filtering)
        accessible_file_tables = [name for name in self.data_sources.keys() 
                                if role_manager.check_table_access(username, self.data_sources[name]['table_name'])]
        
        if accessible_file_tables:
            schema_info += "\nFILE DATA SOURCES (ACCESSIBLE):\n"
            for name in accessible_file_tables:
                info = self.data_sources[name]
                schema_info += f"\n=== FILE: {name} ===\n"
                schema_info += f"SQL Table: {info['table_name']}\n"
                schema_info += f"Columns: {', '.join(info['columns'])}\n"
                schema_info += f"Data types: {info['dtypes']}\n"
                schema_info += f"Sample data:\n"
                for i, row in enumerate(info['sample_data'][:2], 1):
                    schema_info += f"  Row {i}: {row}\n"
                schema_info += "\n"
        
        # Add role-specific SQL generation rules
        user_role = role_manager.get_user_role(username)
        schema_info += "\n" + "="*50 + "\n"
        schema_info += f"ROLE-SPECIFIC SQL GENERATION RULES ({user_role.value.upper() if user_role else 'UNKNOWN'}):\n"
        schema_info += "="*50 + "\n"
        schema_info += "1. ONLY use tables and columns listed above for your role\n"
        schema_info += "2. Respect column access restrictions\n"
        schema_info += "3. Always include LIMIT clause\n"
        schema_info += "4. Use exact column names from accessible schema above\n"
        
        if user_role == UserRole.VIEWER:
            schema_info += "5. VIEWER RESTRICTIONS: No personal data, no salary info, no sensitive data\n"
        elif user_role == UserRole.ANALYST:
            schema_info += "5. ANALYST RESTRICTIONS: Limited personal data access\n"
        
        schema_info += "="*50 + "\n"
        
        return schema_info
    
    def execute_combined_query(self, sql_query: str, username: str, role_manager: 'RoleManager') -> pd.DataFrame:
        """Execute query that might span both database and file sources"""
        print(f"🔍 Executing combined query: {sql_query}")
        
        try:
            # Determine if query involves file tables
            file_tables = [info['table_name'] for info in self.data_sources.values()]
            query_lower = sql_query.lower()
            
            involves_files = any(table in query_lower for table in file_tables)
            involves_db = any(table in query_lower for table in ['employees', 'departments'])
            
            print(f"🔍 Query analysis: involves_files={involves_files}, involves_db={involves_db}")
            print(f"🔍 Available file tables: {file_tables}")
            print(f"🔍 Query: {query_lower}")
            
            if involves_files and involves_db:
                # Complex case: query spans both sources
                return self._execute_cross_source_query(sql_query, username, role_manager)
            elif involves_files:
                # Query only file sources
                print(f"🔍 Executing file-only query")
                return self._execute_file_query(sql_query, username, role_manager)
            else:
                # Query only database
                print(f"🔍 Executing database-only query")
                return self._execute_db_query(sql_query, username, role_manager)
                
        except Exception as e:
            logger.error(f"Combined query execution failed: {e}")
            print(f"❌ Combined query failed, falling back to database: {e}")
            # Fallback to database query
            return self._execute_db_query(sql_query, username, role_manager)
    
    def _execute_file_query(self, sql_query: str, username: str, role_manager: 'RoleManager') -> pd.DataFrame:
        """Execute query on file sources"""
        print(f"🔍 Executing file query: {sql_query}")
        
        # Check if temp database exists
        if not os.path.exists(self.temp_db_path):
            raise Exception("File database not found. Please register files first.")
        
        conn = sqlite3.connect(self.temp_db_path)
        
        try:
            df = pd.read_sql_query(sql_query, conn)
            print(f"✅ File query executed successfully, returned {len(df)} rows")
        except Exception as e:
            print(f"❌ File query failed: {e}")
            raise
        finally:
            conn.close()
        
        return df
    
    def _execute_db_query(self, sql_query: str, username: str, role_manager: 'RoleManager') -> pd.DataFrame:
        """Execute query on database sources with role-based validation"""
        print(f"🔍 Executing database query: {sql_query}")
        
        # Extract table names and validate access
        query_tables = self._extract_table_names_from_query(sql_query)
        print(f"🔍 Tables found in query: {query_tables}")
        
        for table in query_tables:
            if not role_manager.check_table_access(username, table):
                raise Exception(f"Access denied: You don't have permission to access table '{table}'")
        
        conn = sqlite3.connect(self.main_db_path)
        
        try:
            df = pd.read_sql_query(sql_query, conn)
            print(f"✅ Database query executed successfully, returned {len(df)} rows")
        except Exception as e:
            print(f"❌ Database query failed: {e}")
            raise
        finally:
            conn.close()
        
        return df
    
    def _extract_table_names_from_query(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query"""
        query_lower = sql_query.lower()
        tables = []
        
        # Known tables to check for
        known_db_tables = ['employees', 'departments']
        known_file_tables = [info['table_name'] for info in self.data_sources.values()]
        all_tables = known_db_tables + known_file_tables
        
        for table in all_tables:
            if table in query_lower:
                tables.append(table)
        
        return tables
    
    def _execute_cross_source_query(self, sql_query: str, username: str, role_manager: 'RoleManager') -> pd.DataFrame:
        """Execute query that spans both database and file sources"""
        raise Exception("Cross-source queries (database + files in same query) are not yet supported. Please query them separately.")

# Enhanced CrewAI Tools with role-based memory usage

class SQLGeneratorTool(BaseTool):
    name: str = "sql_generator"
    description: str = "Generate SQL queries using LLM with role-based schema awareness and memory"
    
    def _run(self, query_description: str, username: str = "admin", feedback: str = None, iteration: int = 1) -> str:
        """Generate SQL using LLM with role-based context and memory"""
        
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
                # Get role-based schema info
                schema_info = file_manager.get_role_based_schema_info(username, role_manager)
                user_role = role_manager.get_user_role(username)
                
                # Get role-based memory context
                context = memory.get_role_based_context(username, query_description, 
                                                       user_role.value if user_role else "unknown", feedback)
                
                if llm:
                    print(f"🧠 Using LLM for query generation with role-based context")
                    result = self._generate_with_llm(query_description, schema_info, context, llm, feedback, iteration, user_role)
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
    
    def _generate_with_llm(self, query_description: str, schema_info: str, context: str, 
                          llm, feedback: str = None, iteration: int = 1, user_role: UserRole = None) -> str:
        """Generate SQL using LLM with role-based constraints"""
        
        role_constraints = ""
        if user_role == UserRole.VIEWER:
            role_constraints = """
CRITICAL VIEWER ROLE CONSTRAINTS:
- NO access to salary, personal, or sensitive data
- NO access to manager_id, hire_date columns
- ONLY use tables and columns shown in your accessible schema
- Focus on aggregate and non-personal data only
"""
        elif user_role == UserRole.ANALYST:
            role_constraints = """
ANALYST ROLE CONSTRAINTS:
- Limited access to personal data
- NO access to manager_id column in employees table
- Use only tables and columns shown in your accessible schema
"""
        elif user_role == UserRole.ADMIN:
            role_constraints = """
ADMIN ROLE - FULL ACCESS:
- Access to all tables and columns
- Can query sensitive data when necessary
"""
        
        if feedback and iteration > 1:
            # Feedback iteration with role awareness
            prompt = f"""You are an expert SQL developer working on iteration {iteration} for a {user_role.value if user_role else 'unknown'} user.

{schema_info}

{role_constraints}

ROLE-BASED MEMORY CONTEXT:
{context}

ORIGINAL USER REQUEST: {query_description}

CRITICAL - USER FEEDBACK ON PREVIOUS QUERY:
The user provided this feedback: "{feedback}"

You MUST:
1. Analyze the user's feedback within their role constraints
2. Modify the query to address their concerns while respecting role limitations
3. Generate a NEW query that incorporates feedback AND role restrictions

REQUIREMENTS:
1. Generate ONLY the SQL query, no explanations
2. Use EXACT table and column names from YOUR ACCESSIBLE schema
3. Respect all role-based column restrictions
4. Address the user's feedback within role constraints
5. Always include LIMIT clauses

Generate the corrected SQL query:"""
        else:
            # Initial query generation with role awareness
            prompt = f"""You are an expert SQL developer generating queries for a {user_role.value if user_role else 'unknown'} user.

{schema_info}

{role_constraints}

ROLE-BASED MEMORY CONTEXT:
{context}

USER REQUEST: {query_description}

CRITICAL REQUIREMENTS:
1. Generate ONLY the SQL query, no explanations or markdown
2. Use EXACT table and column names from YOUR ACCESSIBLE schema above
3. STRICTLY respect role-based access restrictions
4. NEVER use columns not listed in your accessible schema
5. Always include LIMIT clauses (50 or less)
6. Focus on data appropriate for the user's role

Generate the SQL query now:"""

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
        
        # Check if query is about products/sales (file sources)
        if "product" in desc or "sales" in desc:
            if "product" in desc and "sales" in desc:
                return "SELECT p.product_name, s.quantity, s.total_amount FROM products p JOIN sales s ON p.product_id = s.product_id LIMIT 20;"
            elif "product" in desc:
                if "category" in desc:
                    return "SELECT category, COUNT(*) as count FROM products GROUP BY category LIMIT 10;"
                elif "price" in desc:
                    return "SELECT product_name, price FROM products ORDER BY price DESC LIMIT 10;"
                else:
                    return "SELECT product_name, category, price, stock FROM products LIMIT 20;"
            elif "sales" in desc:
                if "total" in desc or "amount" in desc:
                    return "SELECT customer_name, total_amount, sale_date FROM sales ORDER BY total_amount DESC LIMIT 20;"
                else:
                    return "SELECT customer_name, quantity, sale_date FROM sales LIMIT 20;"
        
        # Role-based fallback queries for database tables
        if user_role == UserRole.VIEWER:
            # Viewer gets very limited, non-personal queries
            if "department" in desc:
                return "SELECT name, location FROM departments LIMIT 10;"
            elif "employees" in desc:
                # Viewers should NOT access employees table
                return "SELECT 'Access denied: Viewers cannot access employee data' as message;"
            else:
                return "SELECT name, location FROM departments LIMIT 10;"
        
        elif user_role == UserRole.ANALYST:
            # Analyst gets more access but still restricted
            if "count" in desc and "department" in desc:
                return "SELECT department, COUNT(*) as count FROM employees GROUP BY department ORDER BY count DESC LIMIT 10;"
            elif "average" in desc and "salary" in desc:
                return "SELECT 'Salary data access restricted for analysts' as message;"
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
    description: str = "Execute SQL queries with role-based validation and column filtering"
    
    def _run(self, sql_query: str, username: str = "admin", db_path: str = None) -> str:
        """Execute SQL with role-based validation and filtering"""
        
        role_manager = getattr(self, '_role_manager', None)
        file_manager = getattr(self, '_file_manager', None)
        stored_db_path = getattr(self, '_db_path', None)
        
        if stored_db_path:
            actual_db_path = stored_db_path
        elif db_path:
            actual_db_path = db_path
        else:
            actual_db_path = os.path.abspath('sample.db')
        
        print(f"🔍 SQL Executor using database: {actual_db_path} for user: {username}")
        
        # Permission check
        if role_manager and not role_manager.check_permission(username, "execute"):
            return "Permission denied: You don't have execute permissions"
        
        # Verify database file exists
        if not os.path.exists(actual_db_path):
            print(f"❌ Database not found at: {actual_db_path}")
            return f"Database file not found: {actual_db_path}"
        
        # Basic SQL validation
        sql_lower = sql_query.lower().strip()
        dangerous_ops = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
        
        for op in dangerous_ops:
            if op in sql_lower:
                return f"Dangerous operation '{op}' not allowed"
        
        if not sql_lower.startswith('select'):
            return "Only SELECT queries allowed"
        
        # Role-based table access validation
        if role_manager:
            # Extract table names from query (simplified)
            query_tables = self._extract_table_names(sql_query)
            print(f"🔍 Tables found in query: {query_tables}")
            
            for table in query_tables:
                if not role_manager.check_table_access(username, table):
                    return f"❌ Access denied: Your role ({role_manager.get_user_role(username).value}) doesn't have permission to access table '{table}'"
        
        try:
            print(f"🔍 Executing SQL on database: {actual_db_path}")
            
            # Check if query involves file sources and execute accordingly
            if file_manager:
                try:
                    df = file_manager.execute_combined_query(sql_query, username, role_manager)
                except Exception as file_error:
                    print(f"⚠️  File query failed, trying database: {file_error}")
                    # Fallback to regular database query
                    conn = sqlite3.connect(actual_db_path)
                    df = pd.read_sql_query(sql_query, conn)
                    conn.close()
            else:
                conn = sqlite3.connect(actual_db_path)
                df = pd.read_sql_query(sql_query, conn)
                conn.close()
            
            print(f"✅ Query executed successfully, returned {len(df)} rows")
            
            # Apply role-based column filtering
            if role_manager and len(df) > 0:
                # Try to identify which table the results came from and apply restrictions
                query_tables = self._extract_table_names(sql_query)
                if len(query_tables) == 1:
                    table_name = query_tables[0]
                    original_columns = len(df.columns)
                    df = role_manager.apply_column_restrictions(username, table_name, df)
                    filtered_columns = len(df.columns)
                    if filtered_columns < original_columns:
                        print(f"🔒 Applied role-based column restrictions for table: {table_name} ({original_columns} -> {filtered_columns} columns)")
            
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
                "database_path": actual_db_path,
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

QUERY RESULTS:
{table_str}

[Full data with {len(df)} rows and {len(df.columns)} columns has been retrieved and stored for export]"""
            
            return return_str
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            print(f"❌ SQL execution error: {error_msg}")
            print(f"🔍 Query: {sql_query}")
            print(f"🔍 Database: {actual_db_path}")
            
            return error_msg
    
    def _extract_table_names(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query (simplified)"""
        query_lower = sql_query.lower()
        tables = []
        
        # Known database tables
        known_db_tables = ['employees', 'departments']
        for table in known_db_tables:
            if table in query_lower:
                tables.append(table)
        
        # Known file tables (get from file manager if available)
        file_manager = getattr(self, '_file_manager', None)
        if file_manager:
            file_tables = [info['table_name'] for info in file_manager.data_sources.values()]
            for table in file_tables:
                if table in query_lower:
                    tables.append(table)
        else:
            # Fallback known file tables
            known_file_tables = ['products', 'sales', 'sample_products', 'sample_sales']
            for table in known_file_tables:
                if table in query_lower:
                    tables.append(table)
        
        return tables

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
        
        # Role-based validation
        if role_manager:
            user_role = role_manager.get_user_role(username)
            if user_role == UserRole.VIEWER:
                # Check for restricted columns
                restricted_patterns = ['salary', 'manager_id', 'hire_date', 'budget']
                for pattern in restricted_patterns:
                    if pattern in corrected_query.lower():
                        corrections_made.append(f"Removed restricted column '{pattern}' for viewer role")
                        # Simple removal - in practice you'd need more sophisticated parsing
                        corrected_query = corrected_query.replace(pattern, '')
                        corrected_query = corrected_query.replace(',,', ',')  # Clean up commas
        
        # Schema correction mappings
        corrections = {
            'employee_id': 'id',
            'employee_name': 'name',
            'first_name': 'name',
            'last_name': 'name',
            'department_name': 'name',
            'department_id': 'id',
        }
        
        # Apply corrections
        for wrong, correct in corrections.items():
            if wrong in corrected_query:
                corrected_query = corrected_query.replace(wrong, correct)
                corrections_made.append(f"{wrong} → {correct}")
        
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
    def __init__(self, db_path: str = "sample.db"):
        self.db_path = os.path.abspath(db_path)
        self.llm = get_llm()
        self.last_execution_result = None
        self.max_feedback_iterations = 5
        
        # Initialize enhanced components
        self.role_manager = RoleManager()
        self.memory = ConversationMemory()
        self.file_manager = FileDataManager(self.db_path)
        
        # Setup users with enhanced roles
        self.role_manager.add_user("admin", UserRole.ADMIN)
        self.role_manager.add_user("analyst", UserRole.ANALYST)
        self.role_manager.add_user("viewer", UserRole.VIEWER)
        
        # Initialize tools with role manager
        self.sql_generator = SQLGeneratorTool()
        self.sql_executor = SQLExecutorTool()
        self.chart_generator = ChartGeneratorTool()
        self.query_validator = QueryValidatorTool()
        
        # Set enhanced attributes on tools
        self.sql_generator._file_manager = self.file_manager
        self.sql_generator._memory = self.memory
        self.sql_generator._role_manager = self.role_manager
        self.sql_generator._llm = self.llm
        
        self.sql_executor._role_manager = self.role_manager
        self.sql_executor._file_manager = self.file_manager
        self.sql_executor._db_path = self.db_path
        self.sql_executor._system_ref = self
        self.sql_executor._last_execution_data = None
        
        self.query_validator._role_manager = self.role_manager
        self.query_validator._llm = self.llm
        
        # Create enhanced agents
        self._create_agents()
    
    def _verify_database_connection(self):
        """Verify database connection"""
        print(f"\n🔍 Verifying database connection...")
        print(f"📂 Expected database path: {self.db_path}")
        
        if not os.path.exists(self.db_path):
            print(f"❌ Database file not found at: {self.db_path}")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                print(f"✅ Database connection verified!")
                print(f"📊 Found tables: {tables}")
                
                if 'employees' in tables:
                    cursor.execute("SELECT COUNT(*) FROM employees")
                    count = cursor.fetchone()[0]
                    print(f"👥 Employees table has {count} records")
                
                conn.close()
                return True
            else:
                print(f"❌ Database exists but has no tables")
                conn.close()
                return False
                
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def _create_fresh_database(self):
        """Create a fresh database with sample data"""
        try:
            print(f"🏗️  Creating fresh database at: {self.db_path}")
            
            conn = sqlite3.connect(self.db_path)
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
        """Create enhanced CrewAI agents with role awareness"""
        
        self.sql_architect = Agent(
            role='Senior SQL Database Architect with Role-Based Access Control',
            goal='Generate perfect SQL queries that respect user roles, use role-based memory patterns, and incorporate feedback while maintaining security constraints',
            backstory="""You are a world-class database architect with expertise in role-based access control. 
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
            goal='Ensure all SQL queries respect role-based access controls and follow enterprise security practices',
            backstory="""You are a cybersecurity expert who specializes in role-based access control and database 
                        security. You understand that viewers should not see sensitive data, analysts have limited 
                        access, and only admins have full access. You enforce these rules strictly.""",
            tools=[self.query_validator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
        
        self.data_analyst = Agent(
            role='Senior Data Analytics Engineer with Multi-Source Query Execution',
            goal='Execute queries across database and file sources while applying role-based column filtering',
            backstory="""You are a senior data engineer who can execute queries across multiple data sources 
                        (databases and files) while ensuring that users only see data appropriate for their role. 
                        You apply column-level security and present results in clear, formatted tables.""",
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
                       data_source: str = "database", feedback: str = None, 
                       iteration: int = 1) -> Dict[str, Any]:
        """Process request with enhanced role-based workflow"""
        
        if not self.role_manager.check_permission(username, "read"):
            return {"error": f"Permission denied: {username} doesn't have read permissions"}
        
        user_role = self.role_manager.get_user_role(username)
        print(f"\n🚀 STARTING ENHANCED CREWAI WORKFLOW - ITERATION {iteration}")
        print(f"📝 Request: {user_request}")
        print(f"👤 User: {username} (Role: {user_role.value if user_role else 'unknown'})")
        if feedback:
            print(f"🔄 Feedback: {feedback}")
        print(f"📊 Visualization: {create_chart} ({chart_type})")
        print(f"💾 Data Source: {data_source}")
        print("="*70)
        
        # Enhanced task descriptions with role awareness
        sql_task_desc = f"""
        Generate an advanced SQL query for user '{username}' with role '{user_role.value if user_role else 'unknown'}':
        
        REQUEST: "{user_request}"
        
        CRITICAL ROLE-BASED REQUIREMENTS:
        - Respect user's role-based access permissions
        - Use role-specific memory patterns and successful queries
        - Only access tables and columns available to this user role
        - Apply appropriate data filtering for the user's role
        
        USER CONTEXT:
        - Username: {username}
        - Role: {user_role.value if user_role else 'unknown'}
        - Data source preference: {data_source}
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
            - Ensure query respects role-based table access
            - Verify no restricted columns are accessed
            - Apply role-specific security policies
            - Validate SQL syntax and performance
            
            USER: {username} (Role: {user_role.value if user_role else 'unknown'})
            ITERATION: {iteration}
            """,
            agent=self.security_specialist,
            expected_output="Role-validated and corrected SQL query with security compliance report",
            context=[sql_task]
        )
        
        # Enhanced execution task
        execution_task = Task(
            description=f"""
            Execute the validated query for user '{username}' with multi-source support:
            
            EXECUTION REQUIREMENTS:
            - Execute across database and file sources as needed
            - Apply role-based column filtering to results
            - Present data in formatted, role-appropriate manner
            - Ensure data privacy compliance
            
            USER: {username} (Role: {user_role.value if user_role else 'unknown'})
            DATA SOURCES: {data_source}
            ITERATION: {iteration}
            """,
            agent=self.data_analyst,
            expected_output="Role-filtered query results with complete formatted data display",
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
                "llm_provider": type(self.llm).__name__ if self.llm else "No LLM",
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
                "user_role": user_role.value if user_role else "unknown",
                "iteration": iteration
            }
    
    def process_request_with_feedback_loop(self, user_request: str, username: str = "admin",
                                         create_chart: bool = False, chart_type: str = "bar",
                                         data_source: str = "database") -> Dict[str, Any]:
        """Enhanced feedback loop with role awareness"""
        
        iteration = 1
        current_feedback = None
        
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
                iteration=iteration
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
                               data_source: str = "database") -> Dict[str, Any]:
        """Apply feedback and retry the request"""
        
        original_request = previous_result.get('request', '')
        previous_iteration = previous_result.get('iteration', 1)
        
        return self.process_request(
            user_request=original_request,
            username=username,
            create_chart=create_chart,
            chart_type=chart_type,
            data_source=data_source,
            feedback=feedback,
            iteration=previous_iteration + 1
        )
    
    def register_file(self, name: str, file_path: str, username: str = "admin") -> Dict[str, Any]:
        """Enhanced file registration with role checking"""
        if not self.role_manager.check_permission(username, "register_files"):
            return {"error": f"Permission denied: {username} doesn't have file registration permissions"}
        
        success = self.file_manager.register_file(name, file_path)
        return {"success": success, "message": f"File {name} registered" if success else f"Failed to register {name}"}
    
    def get_user_accessible_tables(self, username: str) -> List[str]:
        """Get list of tables accessible to the user"""
        user = self.role_manager.users.get(username)
        if not user:
            return []
        
        if "*" in user.accessible_tables:
            # Admin has access to all tables
            return ["employees", "departments", "products", "sales"]
        
        return user.accessible_tables
    
    def get_stats(self, username: str = None) -> Dict[str, Any]:
        """Enhanced statistics with role information"""
        stats = {
            'total_queries': len(self.memory.query_history),
            'successful_queries': sum(1 for q in self.memory.query_history if q['success']),
            'llm_available': self.llm is not None,
            'llm_provider': type(self.llm).__name__ if self.llm else "None",
            'registered_files': len(self.file_manager.data_sources),
            'feedback_sessions': len([q for q in self.memory.query_history if q.get('feedback')])
        }
        
        if username:
            user_role = self.role_manager.get_user_role(username)
            stats['user_role'] = user_role.value if user_role else "unknown"
            stats['accessible_tables'] = self.get_user_accessible_tables(username)
            
            if username in self.memory.conversations:
                user_queries = self.memory.conversations[username]
                stats['user_stats'] = {
                    'total': len(user_queries),
                    'successful': sum(1 for q in user_queries if q['success']),
                    'success_rate': f"{(sum(1 for q in user_queries if q['success']) / len(user_queries) * 100):.1f}%" if user_queries else "0%",
                    'feedback_given': len([q for q in user_queries if q.get('feedback')])
                }
        
        return stats
    
    def export_results(self, data: List[Dict] = None, filename: str = None) -> str:
        """Export query results to Excel"""
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
                
                # Metadata
                metadata = {
                    'Generated By': ['Enhanced CrewAI SQL Analysis System'],
                    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Records': [len(data)],
                    'LLM Provider': [type(self.llm).__name__ if self.llm else "No LLM"]
                }
                
                if self.last_execution_result:
                    metadata['SQL Query'] = [self.last_execution_result.get('sql_query', 'N/A')]
                    metadata['User Role'] = [self.last_execution_result.get('user_role', 'Unknown')]
                
                pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)
            
            return f"Results exported to: {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"

# Enhanced Application Interface
class CrewAIApp:
    def __init__(self):
        print("🏗️  Initializing Enhanced CrewAI Application...")
        
        # Create sample data FIRST
        self._create_sample_data()
        
        # Initialize the enhanced system
        db_path = os.path.abspath("sample.db")
        self.system = CrewAISQLSystem(db_path)
        
        # Verify database
        if not self.system._verify_database_connection():
            print("🔧 Database verification failed, attempting repair...")
            if self.system._create_fresh_database():
                print("✅ Database repair successful!")
            else:
                print("❌ Database repair failed!")
        
        self._show_llm_status()
    
    def _create_sample_data(self):
        """Create enhanced sample database and files"""
        print("📊 Creating enhanced sample database and files...")
        
        db_path = os.path.abspath("sample.db")
        
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"🗑️  Removed existing database")
            except Exception as e:
                print(f"⚠️  Could not remove existing database: {e}")
        
        try:
            conn = sqlite3.connect(db_path)
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
            
            print(f"✅ Enhanced database created successfully")
            
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
            print(f"✅ LLM Active: {llm_type}")
            print("🤖 Enhanced features enabled:")
            print("   • Role-based query generation")
            print("   • Memory-driven pattern learning")
            print("   • Multi-source query support")
            print("   • Human-in-the-loop feedback")
            
            if "Mock" in llm_type:
                print("⚠️  Mock LLM - Limited functionality")
            else:
                print("🚀 Full AI capabilities available")
        else:
            print("❌ No LLM configured - Limited functionality")
        
        print(f"{'='*60}")

def main():
    """Enhanced main function"""
    print("🚀 Initializing Enhanced CrewAI SQL Analysis System...")
    
    try:
        app = CrewAIApp()
        
        print("\n🎯 Enhanced System Features:")
        print("✅ Role-based access control (Admin/Analyst/Viewer)")
        print("✅ Memory-driven query learning")
        print("✅ Multi-source data queries (DB + Files)")
        print("✅ Human-in-the-loop feedback")
        print("✅ Enhanced security and privacy")
        
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
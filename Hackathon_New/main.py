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
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import tabulate with fallback
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers='keys', tablefmt='grid', showindex=False):
        if isinstance(data, pd.DataFrame):
            return data.to_string(index=showindex)
        return str(data)

# Simplified LLM Configuration
def get_llm(provider: str = "auto"):
    """Get LLM instance with simplified provider selection"""
    print(f"\n🔍 Initializing LLM provider: {provider}")
    
    if provider == "mock":
        return _create_mock_llm()
    
    # Try providers in order of preference
    providers = {
        "groq": _try_groq,
        "openai": _try_openai, 
        "anthropic": _try_anthropic,
        "ollama": _try_ollama
    }
    
    if provider in providers:
        return providers[provider]() or _create_mock_llm()
    
    # Auto mode - try all providers
    for provider_func in providers.values():
        llm = provider_func()
        if llm:
            return llm
    
    return _create_mock_llm()

def _try_groq():
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and groq_key.strip():
            from langchain_groq import ChatGroq
            llm = ChatGroq(groq_api_key=groq_key, model_name="mixtral-8x7b-32768", temperature=0.1)
            llm.invoke("Hello")  # Test connection
            print("✅ Using Groq LLM")
            return llm
    except Exception as e:
        print(f"❌ Groq failed: {e}")
    return None

def _try_openai():
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.strip():
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo", temperature=0.1)
            llm.invoke("Hello")
            print("✅ Using OpenAI LLM")
            return llm
    except Exception as e:
        print(f"❌ OpenAI failed: {e}")
    return None

def _try_anthropic():
    try:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key.strip():
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(anthropic_api_key=anthropic_key, model_name="claude-3-sonnet-20240229", temperature=0.1)
            llm.invoke("Hello")
            print("✅ Using Anthropic LLM")
            return llm
    except Exception as e:
        print(f"❌ Anthropic failed: {e}")
    return None

def _try_ollama():
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            from langchain_community.llms import Ollama
            llm = Ollama(model="llama2", temperature=0.1)
            print("✅ Using Ollama LLM")
            return llm
    except Exception as e:
        print(f"❌ Ollama failed: {e}")
    return None

def _create_mock_llm():
    try:
        from langchain_core.language_models.base import BaseLanguageModel
        from langchain_core.outputs import LLMResult, Generation
        
        class MockLLM(BaseLanguageModel):
            @property
            def _llm_type(self) -> str:
                return "mock"
            
            def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
                generations = []
                for prompt in prompts:
                    if "SELECT" in prompt or "sql" in prompt.lower():
                        if "count" in prompt.lower() and "department" in prompt.lower():
                            text = "SELECT department, COUNT(*) as count FROM employees GROUP BY department ORDER BY count DESC;"
                        elif "average" in prompt.lower() and "salary" in prompt.lower():
                            text = "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC;"
                        else:
                            text = "SELECT * FROM employees LIMIT 10;"
                    else:
                        text = "Query analysis completed successfully."
                    generations.append([Generation(text=text)])
                return LLMResult(generations=generations)
            
            def invoke(self, prompt, **kwargs):
                result = self._generate([prompt])
                return type('MockResponse', (), {'content': result.generations[0][0].text})()
        
        print("⚠️  Using Mock LLM - Limited functionality")
        return MockLLM()
    except Exception as e:
        logger.error(f"Failed to create mock LLM: {e}")
        return None

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
    accessible_tables: List[str]
    restricted_columns: Dict[str, List[str]]

class RoleManager:
    def __init__(self):
        self.users = {}
        self.role_permissions = {
            UserRole.ADMIN: {
                "permissions": ["read", "write", "execute", "delete", "manage_users", "register_files"],
                "accessible_tables": ["*"],
                "restricted_columns": {}
            },
            UserRole.ANALYST: {
                "permissions": ["read", "execute", "register_files"],
                "accessible_tables": ["employees", "departments", "products", "sales", "sample_products", "sample_sales"],
                "restricted_columns": {"employees": ["manager_id"]}
            },
            UserRole.VIEWER: {
                "permissions": ["read"],
                "accessible_tables": ["departments", "products", "sales", "sample_products", "sample_sales"],
                "restricted_columns": {
                    "departments": ["budget"],
                    "sales": ["customer_name"],
                    "sample_sales": ["customer_name"]
                }
            }
        }
    
    def add_user(self, username: str, role: UserRole):
        role_config = self.role_permissions.get(role)
        if role_config:
            self.users[username] = User(
                username=username, role=role,
                permissions=role_config["permissions"],
                accessible_tables=role_config["accessible_tables"],
                restricted_columns=role_config["restricted_columns"]
            )
    
    def check_permission(self, username: str, permission: str) -> bool:
        user = self.users.get(username)
        return user and permission in user.permissions
    
    def check_table_access(self, username: str, table_name: str) -> bool:
        user = self.users.get(username)
        if not user:
            return False
        return "*" in user.accessible_tables or table_name in user.accessible_tables
    
    def get_accessible_columns(self, username: str, table_name: str) -> List[str]:
        user = self.users.get(username)
        if not user:
            return []
        
        all_columns = self._get_table_columns(table_name)
        restricted = user.restricted_columns.get(table_name, [])
        return [col for col in all_columns if col not in restricted]
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        table_schemas = {
            "employees": ["id", "name", "department", "salary", "hire_date", "manager_id", "status"],
            "departments": ["id", "name", "budget", "location"],
            "products": ["product_id", "product_name", "category", "price", "stock", "supplier"],
            "sales": ["sale_id", "product_id", "customer_name", "quantity", "sale_date", "total_amount"],
            "sample_products": ["product_id", "product_name", "category", "price", "stock", "supplier"],
            "sample_sales": ["sale_id", "product_id", "customer_name", "quantity", "sale_date", "total_amount"]
        }
        return table_schemas.get(table_name, [])
    
    def get_user_role(self, username: str) -> Optional[UserRole]:
        user = self.users.get(username)
        return user.role if user else None

# Enhanced Memory system
class ConversationMemory:
    def __init__(self, memory_file: str = "conversation_memory.pkl"):
        self.memory_file = memory_file
        self.conversations = {}
        self.successful_patterns = {}
        self.feedback_history = {}
        self.load_memory()
    
    def save_memory(self):
        try:
            memory_data = {
                'conversations': self.conversations,
                'successful_patterns': self.successful_patterns,
                'feedback_history': self.feedback_history
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
                    self.successful_patterns = memory_data.get('successful_patterns', {})
                    self.feedback_history = memory_data.get('feedback_history', {})
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
    
    def add_conversation(self, username: str, request: str, sql_query: str, success: bool, 
                        feedback: str = None, user_role: str = None):
        if username not in self.conversations:
            self.conversations[username] = []
        
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'request': request, 'sql_query': sql_query,
            'success': success, 'feedback': feedback, 'user_role': user_role
        }
        
        self.conversations[username].append(conversation_entry)
        
        if success and user_role:
            pattern_key = self._extract_pattern(request)
            if pattern_key not in self.successful_patterns:
                self.successful_patterns[pattern_key] = []
            self.successful_patterns[pattern_key].append({
                'sql_query': sql_query, 'request': request,
                'timestamp': datetime.now().isoformat()
            })
        
        self.save_memory()
    
    def _extract_pattern(self, request: str) -> str:
        keywords = {
            'aggregation': ['count', 'sum', 'average', 'avg', 'max', 'min', 'total'],
            'filtering': ['where', 'filter', 'find', 'search'],
            'grouping': ['group', 'by department', 'by category'],
            'sorting': ['order', 'sort', 'top', 'highest', 'lowest']
        }
        
        request_lower = request.lower()
        found_categories = []
        for category, terms in keywords.items():
            if any(term in request_lower for term in terms):
                found_categories.append(category)
        
        return '_'.join(found_categories) if found_categories else 'general'
    
    def get_context(self, username: str, request: str) -> str:
        context = f"=== MEMORY CONTEXT ===\n"
        
        # Recent conversations
        if username in self.conversations:
            recent = self.conversations[username][-3:]
            context += f"Recent queries for {username}:\n"
            for entry in recent:
                context += f"- {entry['request']} -> {entry['sql_query']} (Success: {entry['success']})\n"
        
        # Similar patterns
        pattern = self._extract_pattern(request)
        if pattern in self.successful_patterns:
            examples = self.successful_patterns[pattern][-2:]
            context += f"\nSimilar successful patterns:\n"
            for example in examples:
                context += f"- {example['request']} -> {example['sql_query']}\n"
        
        return context

# Database Schema Inspector
class DatabaseSchemaInspector:
    def __init__(self, selected_databases: List[str]):
        self.selected_databases = selected_databases
        self.schema_cache = {}
        self.refresh_schema_cache()

    def refresh_schema_cache(self):
        self.schema_cache = {}
        for db_name in self.selected_databases:
            self.schema_cache[db_name] = self._inspect_database_schema(db_name)

    def _inspect_database_schema(self, db_name: str) -> Dict[str, Any]:
        try:
            db_path = os.path.abspath(db_name)
            if not os.path.exists(db_path):
                return {"tables": {}, "error": f"Database {db_name} not found"}
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            schema_info = {"tables": {}, "database_name": db_name}
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]
            
            for table_name in table_names:
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns_info = cursor.fetchall()
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                
                table_schema = {
                    "columns": {col[1]: {"type": col[2]} for col in columns_info},
                    "column_names": [col[1] for col in columns_info],
                    "row_count": row_count
                }
                schema_info["tables"][table_name] = table_schema
            
            conn.close()
            return schema_info
        except Exception as e:
            return {"tables": {}, "error": f"Schema inspection failed: {str(e)}"}

    def get_schema_context_for_llm(self, username: str = None, role_manager: 'RoleManager' = None) -> str:
        context = "=== DATABASE SCHEMA ===\n"
        for db_name, schema in self.schema_cache.items():
            if "error" in schema:
                context += f"❌ {db_name}: {schema['error']}\n"
                continue
            
            context += f"📊 DATABASE: {db_name}\n"
            for table_name, table_info in schema.get("tables", {}).items():
                if role_manager and username and not role_manager.check_table_access(username, table_name):
                    continue
                accessible_columns = (role_manager.get_accessible_columns(username, table_name) 
                                    if role_manager and username else table_info["column_names"])
                context += f"TABLE: {table_name} ({table_info['row_count']} rows)\n"
                context += f"COLUMNS: {', '.join(accessible_columns)}\n\n"
        
        context += "RULES:\n1. Only use tables/columns shown above\n2. Never add user_id filters\n3. Always include LIMIT clause\n"
        return context

    def validate_query_against_schema(self, sql_query: str) -> Dict[str, Any]:
        try:
            tables_re = re.findall(r'(?:FROM|JOIN)\s+([`"\']?\w+[`"\']?)', sql_query, re.IGNORECASE)
            columns_re = re.findall(r'(?:SELECT|WHERE|BY|AND|,)\s+([`"\']?\w+(?:\.\w+)?[`"\']?)', sql_query, re.IGNORECASE)
            
            referenced_tables = {re.sub(r'[`"\']', '', t) for t in tables_re}
            referenced_columns = {re.sub(r'[`"\']', '', c).split('.')[-1] for c in columns_re}
            referenced_columns -= {'distinct', 'as', 'on', 'limit', 'desc', 'asc'}
            
            # Get available tables and columns
            all_tables = set()
            all_columns = set()
            for schema in self.schema_cache.values():
                if "tables" in schema:
                    all_tables.update(schema["tables"].keys())
                    for table_info in schema["tables"].values():
                        all_columns.update(col.lower() for col in table_info["column_names"])
            
            issues = []
            for table in referenced_tables:
                if table not in all_tables:
                    issues.append(f"Table '{table}' does not exist")
            
            for col in referenced_columns:
                if col.lower() not in all_columns:
                    issues.append(f"Column '{col}' does not exist")
            
            return {"valid": len(issues) == 0, "issues": issues}
        except Exception as e:
            return {"valid": False, "issues": [f"Validation error: {str(e)}"]}

# File Data Manager
class FileDataManager:
    def __init__(self, selected_databases: List[str] = None):
        self.selected_databases = selected_databases or ["sample.db"]
        self.database_paths = {db: os.path.abspath(db) for db in self.selected_databases}
        self.temp_db_path = "FILE_DATA.db"
        self.data_sources = {}  # Track uploaded files
        
        # Ensure FILE_DATA.db is included in selected databases for schema inspection
        if self.temp_db_path not in self.selected_databases:
            self.selected_databases.append(self.temp_db_path)
            self.database_paths[self.temp_db_path] = os.path.abspath(self.temp_db_path)
        
        self.schema_inspector = DatabaseSchemaInspector(self.selected_databases)
        self._initialize_file_database()

    def _initialize_file_database(self):
        """Initialize the FILE_DATA.db if it doesn't exist"""
        try:
            if not os.path.exists(self.temp_db_path):
                conn = sqlite3.connect(self.temp_db_path)
                cursor = conn.cursor()
                # Create a metadata table to track uploaded files
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS _file_metadata (
                        table_name TEXT PRIMARY KEY,
                        original_filename TEXT,
                        upload_date TEXT,
                        rows_count INTEGER,
                        columns_list TEXT
                    )
                ''')
                conn.commit()
                conn.close()
                print(f"✅ Initialized FILE_DATA.db for file uploads")
        except Exception as e:
            print(f"❌ Failed to initialize FILE_DATA.db: {e}")

    def register_file(self, name: str, file_path: str, username: str = "admin") -> Dict[str, Any]:
        """Register and load a file into FILE_DATA.db"""
        try:
            if not os.path.exists(file_path):
                return {"success": False, "message": f"File not found: {file_path}"}
            
            print(f"📁 Processing file upload: {name} from {file_path}")
            
            # Read the file based on extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                return {"success": False, "message": f"Unsupported file type: {file_ext}"}
            
            if df.empty:
                return {"success": False, "message": "File is empty or could not be read"}
            
            # Clean table name (remove spaces, special chars)
            table_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower().strip())
            if not table_name or table_name[0].isdigit():
                table_name = f"table_{table_name}"
            
            # Connect to FILE_DATA.db and insert data
            conn = sqlite3.connect(self.temp_db_path)
            
            # Drop table if it already exists
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Insert data into new table
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # Update metadata table
            cursor.execute('''
                INSERT OR REPLACE INTO _file_metadata 
                (table_name, original_filename, upload_date, rows_count, columns_list) 
                VALUES (?, ?, ?, ?, ?)
            ''', (
                table_name,
                os.path.basename(file_path),
                datetime.now().isoformat(),
                len(df),
                ','.join(df.columns.tolist())
            ))
            
            conn.commit()
            conn.close()
            
            # Store in data sources tracking
            self.data_sources[name] = {
                'table_name': table_name,
                'original_filename': os.path.basename(file_path),
                'file_path': file_path,
                'rows': len(df),
                'columns': df.columns.tolist(),
                'upload_date': datetime.now().isoformat()
            }
            
            # Refresh schema cache to include new table
            self.refresh_schema_cache()
            
            print(f"✅ Successfully registered file '{name}' as table '{table_name}' with {len(df)} rows")
            
            return {
                "success": True,
                "message": f"File '{name}' uploaded successfully as table '{table_name}' with {len(df)} rows and {len(df.columns)} columns",
                "table_name": table_name,
                "rows": len(df),
                "columns": df.columns.tolist()
            }
            
        except Exception as e:
            error_msg = f"Failed to register file '{name}': {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "message": error_msg}

    def refresh_schema_cache(self):
        """Refresh schema cache including FILE_DATA.db"""
        if hasattr(self, 'schema_inspector'):
            self.schema_inspector.refresh_schema_cache()
            print("🔄 Schema cache refreshed including uploaded files")

    def get_uploaded_files_info(self) -> List[Dict[str, Any]]:
        """Get information about uploaded files"""
        try:
            if not os.path.exists(self.temp_db_path):
                return []
            
            conn = sqlite3.connect(self.temp_db_path)
            cursor = conn.cursor()
            
            # Check if metadata table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='_file_metadata';")
            if not cursor.fetchone():
                conn.close()
                return []
            
            cursor.execute("SELECT * FROM _file_metadata ORDER BY upload_date DESC")
            files_info = []
            
            for row in cursor.fetchall():
                files_info.append({
                    "table_name": row[0],
                    "original_filename": row[1],
                    "upload_date": row[2],
                    "rows_count": row[3],
                    "columns_list": row[4].split(',') if row[4] else []
                })
            
            conn.close()
            return files_info
            
        except Exception as e:
            print(f"❌ Error getting uploaded files info: {e}")
            return []

    def delete_uploaded_file(self, table_name: str) -> Dict[str, Any]:
        """Delete an uploaded file/table"""
        try:
            if not os.path.exists(self.temp_db_path):
                return {"success": False, "message": "No uploaded files database found"}
            
            conn = sqlite3.connect(self.temp_db_path)
            cursor = conn.cursor()
            
            # Remove from metadata
            cursor.execute("DELETE FROM _file_metadata WHERE table_name = ?", (table_name,))
            
            # Drop the actual table
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            conn.commit()
            conn.close()
            
            # Remove from data sources tracking
            for name, info in list(self.data_sources.items()):
                if info.get('table_name') == table_name:
                    del self.data_sources[name]
                    break
            
            # Refresh schema
            self.refresh_schema_cache()
            
            return {"success": True, "message": f"File table '{table_name}' deleted successfully"}
            
        except Exception as e:
            return {"success": False, "message": f"Failed to delete file: {str(e)}"}

    
    def execute_combined_query(self, sql_query: str, username: str, role_manager: 'RoleManager') -> pd.DataFrame:
        clean_query = self._clean_user_filters(sql_query)
        
        try:
            # First try FILE_DATA.db (uploaded files)
            if os.path.exists(self.temp_db_path):
                try:
                    conn = sqlite3.connect(self.temp_db_path)
                    df = pd.read_sql_query(clean_query, conn)
                    conn.close()
                    print(f"✅ Query executed successfully on FILE_DATA.db: {len(df)} rows")
                    return df
                except Exception as file_error:
                    print(f"⚠️ FILE_DATA.db query failed: {file_error}")
                    # Continue to try other databases
            
            # Try each configured database
            for db_name in self.selected_databases:
                try:
                    if db_name == self.temp_db_path:
                        continue  # Already tried above
                    
                    db_path = self.database_paths[db_name]
                    if not os.path.exists(db_path):
                        continue
                    
                    conn = sqlite3.connect(db_path)
                    df = pd.read_sql_query(clean_query, conn)
                    conn.close()
                    print(f"✅ Query executed successfully on {db_name}: {len(df)} rows")
                    return df
                except Exception as db_error:
                    print(f"⚠️ Database {db_name} query failed: {db_error}")
                    continue
            
            raise Exception("Query failed on all databases")
        except Exception as e:
            print(f"❌ Combined query execution failed: {e}")
            return pd.DataFrame([{"error_message": str(e)}])

    def _clean_user_filters(self, sql_query: str) -> str:
        cleaned = sql_query
        cleaned = re.sub(r'\s+WHERE\s+user_id\s*=\s*[\'"][^\'\"]*[\'"]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+AND\s+user_id\s*=\s*[\'"][^\'\"]*[\'"]', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def get_all_databases_tables(self) -> Dict[str, List[Dict[str, Any]]]:
        all_tables = {}
        for db_name in self.selected_databases:
            all_tables[db_name] = self._get_database_tables(db_name)
        return all_tables

    def _get_database_tables(self, database_name: str) -> List[Dict[str, Any]]:
        tables_info = []
        try:
            db_path = self.database_paths.get(database_name, os.path.abspath(database_name))
            if not os.path.exists(db_path):
                return tables_info
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]
            
            for table_name in table_names:
                # Skip metadata table for FILE_DATA.db
                if table_name == '_file_metadata' and database_name == self.temp_db_path:
                    continue
                
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns_info = cursor.fetchall()
                
                # For FILE_DATA.db tables, add upload info
                upload_info = ""
                if database_name == self.temp_db_path:
                    try:
                        cursor.execute("SELECT original_filename, upload_date FROM _file_metadata WHERE table_name = ?", (table_name,))
                        metadata = cursor.fetchone()
                        if metadata:
                            upload_info = f" (Uploaded: {metadata[0]})"
                    except:
                        pass
                
                table_info = {
                    "name": table_name,
                    "row_count": row_count,
                    "columns": [{"name": col[1], "type": col[2]} for col in columns_info],
                    "column_names": [col[1] for col in columns_info],
                    "upload_info": upload_info,
                    "is_uploaded": database_name == self.temp_db_path
                }
                tables_info.append(table_info)
            conn.close()
        except Exception as e:
            logger.error(f"Error getting tables for {database_name}: {e}")
        return tables_info

# SQL Converter Tool
class SQLConverterTool(BaseTool):
    name: str = "sql_converter"
    description: str = "Convert SQLite SQL queries to other database dialects (SQL Server, PostgreSQL, DB2)"
    
    def _run(self, sql_query: str, target_db: str = "postgresql") -> str:
        """Convert SQLite SQL to target database dialect"""
        try:
            query = sql_query.strip()
            if not query.upper().startswith('SELECT'):
                return f"Error: Only SELECT queries are supported for conversion"
            
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
        conversions = [
            (r'\bDATE\s*\(\s*([^)]+)\s*\)', r"TO_DATE(\1, 'YYYY-MM-DD')"),
            (r'\bDATETIME\s*\(\s*([^)]+)\s*\)', r"TO_TIMESTAMP(\1, 'YYYY-MM-DD HH24:MI:SS')"),
            (r'\bstrftime\s*\(\s*[\'"]%Y[\'"],\s*([^)]+)\s*\)', r"EXTRACT(YEAR FROM \1)"),
            (r'\bstrftime\s*\(\s*[\'"]%m[\'"],\s*([^)]+)\s*\)', r"EXTRACT(MONTH FROM \1)"),
            (r'\bstrftime\s*\(\s*[\'"]%d[\'"],\s*([^)]+)\s*\)', r"EXTRACT(DAY FROM \1)"),
            (r'\bSUBSTR\s*\(', r"SUBSTRING("),
            (r'\bINTEGER\b', r"INTEGER"), (r'\bTEXT\b', r"VARCHAR"), (r'\bREAL\b', r"DECIMAL"),
        ]
        for pattern, replacement in conversions:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        return f"-- PostgreSQL Query\n{converted}"
    
    def _convert_to_sqlserver(self, query: str) -> str:
        """Convert SQLite query to SQL Server"""
        converted = query
        conversions = [
            (r'\bDATE\s*\(\s*([^)]+)\s*\)', r"CAST(\1 AS DATE)"),
            (r'\bDATETIME\s*\(\s*([^)]+)\s*\)', r"CAST(\1 AS DATETIME)"),
            (r'\bstrftime\s*\(\s*[\'"]%Y[\'"],\s*([^)]+)\s*\)', r"YEAR(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%m[\'"],\s*([^)]+)\s*\)', r"MONTH(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%d[\'"],\s*([^)]+)\s*\)', r"DAY(\1)"),
            (r'\|\|', r"+"), (r'\bSUBSTR\s*\(', r"SUBSTRING("), (r'\bLENGTH\s*\(', r"LEN("),
            (r'\bINTEGER\b', r"INT"), (r'\bTEXT\b', r"NVARCHAR(MAX)"), (r'\bREAL\b', r"DECIMAL(18,2)"),
        ]
        for pattern, replacement in conversions:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        
        limit_match = re.search(r'\bLIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            limit_value = limit_match.group(1)
            converted = re.sub(r'\bSELECT\b', f"SELECT TOP {limit_value}", converted, count=1, flags=re.IGNORECASE)
            converted = re.sub(r'\bLIMIT\s+\d+\s*;?\s*$', '', converted, flags=re.IGNORECASE)
        return f"-- SQL Server Query\n{converted}"
    
    def _convert_to_db2(self, query: str) -> str:
        """Convert SQLite query to DB2"""
        converted = query
        conversions = [
            (r'\bDATE\s*\(\s*([^)]+)\s*\)', r"DATE(\1)"),
            (r'\bDATETIME\s*\(\s*([^)]+)\s*\)', r"TIMESTAMP(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%Y[\'"],\s*([^)]+)\s*\)', r"YEAR(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%m[\'"],\s*([^)]+)\s*\)', r"MONTH(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%d[\'"],\s*([^)]+)\s*\)', r"DAY(\1)"),
            (r'\bSUBSTR\s*\(', r"SUBSTR("), (r'\bLENGTH\s*\(', r"LENGTH("),
            (r'\bINTEGER\b', r"INTEGER"), (r'\bTEXT\b', r"VARCHAR(1000)"), (r'\bREAL\b', r"DECIMAL(15,2)"),
            (r'\bLIMIT\s+(\d+)\s*;?\s*$', r"FETCH FIRST \1 ROWS ONLY"),
        ]
        for pattern, replacement in conversions:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        return f"-- DB2 Query\n{converted}"

# CrewAI Tools
class SQLGeneratorTool(BaseTool):
    name: str = "sql_generator"
    description: str = "Generate SQL queries with schema awareness"

    def _run(self, query_description: str, username: str = "admin", schema_context: str = "", 
             memory_context: str = "") -> str:
        try:
            file_manager = getattr(self, '_file_manager', None)
            role_manager = getattr(self, '_role_manager', None)
            llm = getattr(self, '_llm', None)
            
            # Apply role-based schema filtering
            if role_manager and username:
                filtered_schema = self._filter_schema_by_role(schema_context, username, role_manager)
                schema_context = filtered_schema
            
            if llm and type(llm).__name__ != 'MockLLM':
                prompt = f"""
                Generate a SQLite query for: {query_description}
                
                {schema_context}
                {memory_context}
                
                Rules:
                1. Only use tables and columns from the schema above
                2. Never add user_id or username filters unless they exist as columns
                3. Return only the SQL query
                4. Always include LIMIT clause
                5. Respect user role restrictions shown in schema
                """
                response = llm.invoke(prompt)
                sql_query = response.content if hasattr(response, 'content') else str(response)
                sql_query = sql_query.strip().replace('```sql', '').replace('```', '').strip()
                if not sql_query.endswith(';'):
                    sql_query += ';'
                return sql_query
            else:
                # Enhanced fallback logic with role awareness
                return self._generate_role_aware_fallback(query_description, username, role_manager)
        except Exception as e:
            return f"SELECT 'Error: {str(e)}' AS error;"
    
    def _filter_schema_by_role(self, schema_context: str, username: str, role_manager: 'RoleManager') -> str:
        """Filter schema context based on user role"""
        if not role_manager:
            return schema_context
        
        user_role = role_manager.get_user_role(username)
        if not user_role or user_role == UserRole.ADMIN:
            return schema_context
        
        # For non-admin users, filter the schema context
        filtered_lines = []
        current_table = None
        
        for line in schema_context.split('\n'):
            if line.startswith('TABLE:'):
                table_name = line.split(':')[1].strip().split()[0]
                if role_manager.check_table_access(username, table_name):
                    current_table = table_name
                    filtered_lines.append(line)
                else:
                    current_table = None
            elif line.startswith('COLUMNS:') and current_table:
                # Filter columns based on role restrictions
                accessible_columns = role_manager.get_accessible_columns(username, current_table)
                filtered_lines.append(f"COLUMNS: {', '.join(accessible_columns)}")
            elif current_table:  # Only include other lines if table is accessible
                filtered_lines.append(line)
            else:
                # Include non-table specific lines
                if not line.startswith('TABLE:') and not line.startswith('COLUMNS:'):
                    filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _generate_role_aware_fallback(self, query_description: str, username: str, role_manager: 'RoleManager') -> str:
        """Generate fallback query with strict role awareness"""
        if not role_manager:
            return "SELECT * FROM employees LIMIT 10;"
        
        # Get accessible tables for user
        accessible_tables = []
        default_tables = ["employees", "departments", "products", "sales", "sample_products", "sample_sales"]
        
        for table in default_tables:
            if role_manager.check_table_access(username, table):
                accessible_tables.append(table)
        
        if not accessible_tables:
            user_role = role_manager.get_user_role(username)
            role_name = user_role.value if user_role else "unknown"
            return f"SELECT 'Access denied: Role {role_name} has no accessible tables' AS error_message;"
        
        # If user asks for employees data but doesn't have access, return error
        if ("employee" in query_description.lower() or "staff" in query_description.lower()) and "employees" not in accessible_tables:
            user_role = role_manager.get_user_role(username)
            role_name = user_role.value if user_role else "unknown"
            accessible_list = ", ".join(accessible_tables)
            return f"SELECT 'Access denied: Role {role_name} cannot access employee data. Available tables: {accessible_list}' AS error_message;"
        
        # Choose appropriate table based on query and access
        target_table = accessible_tables[0]  # Default to first accessible
        
        if "department" in query_description.lower() and "departments" in accessible_tables:
            target_table = "departments"
        elif "product" in query_description.lower() and "products" in accessible_tables:
            target_table = "products"
        elif "sales" in query_description.lower() and "sales" in accessible_tables:
            target_table = "sales"
        elif "employee" in query_description.lower() and "employees" in accessible_tables:
            target_table = "employees"
        
        # Generate appropriate query for accessible table
        if "count" in query_description.lower():
            if target_table == "employees":
                return "SELECT department, COUNT(*) as count FROM employees GROUP BY department LIMIT 10;"
            elif target_table == "departments":
                return "SELECT name, COUNT(*) as count FROM departments LIMIT 10;"
            else:
                return f"SELECT COUNT(*) as total_count FROM {target_table};"
        elif "average" in query_description.lower() and "salary" in query_description.lower():
            if target_table == "employees":
                return "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department LIMIT 10;"
            else:
                user_role = role_manager.get_user_role(username)
                role_name = user_role.value if user_role else "unknown"
                return f"SELECT 'Access denied: Role {role_name} cannot access salary data' AS error_message;"
        else:
            return f"SELECT * FROM {target_table} LIMIT 10;"

class SQLValidatorTool(BaseTool):
    name: str = "sql_validator"
    description: str = "Validate SQL queries against schema and role-based permissions"
    
    def _run(self, sql_query: str, username: str = "admin") -> str:
        try:
            schema_inspector = getattr(self, '_schema_inspector', None)
            role_manager = getattr(self, '_role_manager', None)
            
            # First check schema validation
            if schema_inspector:
                validation_result = schema_inspector.validate_query_against_schema(sql_query)
                if not validation_result["valid"]:
                    issues = '; '.join(validation_result["issues"])
                    return f"❌ Schema validation failed: {issues}"
            
            # Then check role-based table access
            if role_manager and username:
                table_access_result = self._validate_table_access(sql_query, username, role_manager)
                if not table_access_result["valid"]:
                    issues = '; '.join(table_access_result["issues"])
                    return f"❌ Access denied: {issues}"
            
            return f"✅ Query validated successfully: {sql_query}"
        except Exception as e:
            return f"❌ Validation error: {str(e)}"
    
    def _validate_table_access(self, sql_query: str, username: str, role_manager: 'RoleManager') -> Dict[str, Any]:
        """Validate that user has access to all tables in the query"""
        try:
            # Extract table names from SQL query
            tables_in_query = re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
            tables_in_query.extend(re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE))
            
            # Remove duplicates
            tables_in_query = list(set(tables_in_query))
            
            issues = []
            for table_name in tables_in_query:
                if not role_manager.check_table_access(username, table_name):
                    user_role = role_manager.get_user_role(username)
                    role_name = user_role.value if user_role else "unknown"
                    issues.append(f"Role '{role_name}' does not have access to table '{table_name}'")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "tables_checked": tables_in_query
            }
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Table access validation error: {str(e)}"],
                "tables_checked": []
            }

class SQLExecutorTool(BaseTool):
    name: str = "sql_executor"
    description: str = "Execute validated SQL queries with role-based access control"

    def _run(self, sql_query: str, username: str = "admin") -> str:
        try:
            file_manager = getattr(self, '_file_manager', None)
            role_manager = getattr(self, '_role_manager', None)
            
            if not file_manager:
                return "❌ File manager not available"
            
            # CRITICAL: Check table access permissions BEFORE execution
            if role_manager and username:
                table_access_check = self._check_table_access_permissions(sql_query, username, role_manager)
                if not table_access_check["allowed"]:
                    return f"❌ Access denied: {table_access_check['reason']}"
            
            df = file_manager.execute_combined_query(sql_query, username, role_manager)
            
            if "error_message" in df.columns:
                return f"❌ Execution failed: {df['error_message'].iloc[0]}"
            
            # Apply role-based column filtering (secondary protection)
            if role_manager and username:
                df = self._apply_role_based_filtering(df, username, role_manager, sql_query)
            
            # Store result for system
            if hasattr(self, '_system_ref'):
                self._system_ref.last_execution_result = {
                    "success": True, "data": df.to_dict('records'),
                    "columns": df.columns.tolist(), "row_count": len(df),
                    "sql_query": sql_query, "dataframe": df
                }
            
            table_str = tabulate(df.head(20), headers='keys', tablefmt='grid', showindex=False)
            if len(df) > 20:
                table_str += f"\n... and {len(df) - 20} more rows"
            
            return f"✅ Query executed successfully! {len(df)} rows returned.\n\n{table_str}"
        except Exception as e:
            return f"❌ Execution error: {str(e)}"
    
    def _check_table_access_permissions(self, sql_query: str, username: str, role_manager: 'RoleManager') -> Dict[str, Any]:
        """Check if user has permission to access all tables in the query"""
        try:
            # Extract table names from SQL query
            tables_in_query = re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
            tables_in_query.extend(re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE))
            
            # Remove duplicates
            tables_in_query = list(set(tables_in_query))
            
            user_role = role_manager.get_user_role(username)
            role_name = user_role.value if user_role else "unknown"
            
            for table_name in tables_in_query:
                if not role_manager.check_table_access(username, table_name):
                    return {
                        "allowed": False,
                        "reason": f"User role '{role_name}' does not have access to table '{table_name}'. Accessible tables for your role: {self._get_user_accessible_tables(username, role_manager)}",
                        "blocked_table": table_name
                    }
            
            return {
                "allowed": True,
                "reason": "All table access permissions verified",
                "tables_accessed": tables_in_query
            }
        except Exception as e:
            return {
                "allowed": False,
                "reason": f"Permission check error: {str(e)}",
                "blocked_table": "unknown"
            }
    
    def _get_user_accessible_tables(self, username: str, role_manager: 'RoleManager') -> str:
        """Get a formatted string of tables accessible to the user"""
        user = role_manager.users.get(username)
        if not user:
            return "none"
        
        if "*" in user.accessible_tables:
            return "all tables"
        
        return ", ".join(user.accessible_tables) if user.accessible_tables else "none"
    
    def _apply_role_based_filtering(self, df: pd.DataFrame, username: str, role_manager: 'RoleManager', sql_query: str) -> pd.DataFrame:
        """Apply role-based filtering to query results (secondary protection for columns)"""
        try:
            user = role_manager.users.get(username)
            if not user or user.role == UserRole.ADMIN:
                return df  # Admin has no restrictions
            
            # Extract table names from SQL query
            tables_in_query = re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
            tables_in_query.extend(re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE))
            
            # Get all restricted columns for tables in the query
            restricted_columns = set()
            for table_name in tables_in_query:
                if table_name in user.restricted_columns:
                    restricted_columns.update(user.restricted_columns[table_name])
            
            # Remove restricted columns that exist in the dataframe
            safe_columns = [col for col in df.columns if col not in restricted_columns]
            
            if len(safe_columns) < len(df.columns):
                removed_cols = [col for col in df.columns if col in restricted_columns]
                print(f"🔒 Removed restricted columns for {user.role.value}: {removed_cols}")
                return df[safe_columns]
            
            return df
        except Exception as e:
            print(f"⚠️ Role filtering error (using original data): {e}")
            return df

class ChartGeneratorTool(BaseTool):
    name: str = "chart_generator"
    description: str = "Generate charts from data"
    
    def _run(self, data: List[Dict], chart_type: str = "bar", title: str = None) -> str:
        try:
            if not data:
                return "❌ No data for chart generation"
            
            df = pd.DataFrame(data)
            print(f"📊 Chart data shape: {df.shape}")
            print(f"📊 Chart data columns: {df.columns.tolist()}")
            print(f"📊 Chart data types: {df.dtypes.to_dict()}")
            
            # Clear any existing plots
            plt.clf()
            plt.figure(figsize=(12, 8))
            
            if chart_type == "bar":
                # Get numeric and text columns
                numeric_cols = df.select_dtypes(include=['number', 'int64', 'float64']).columns.tolist()
                text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                
                print(f"📊 Numeric columns: {numeric_cols}")
                print(f"📊 Text columns: {text_cols}")
                
                if numeric_cols and text_cols:
                    x_col, y_col = text_cols[0], numeric_cols[0]
                    
                    # Handle data preparation
                    x_data = df[x_col].astype(str)
                    y_data = pd.to_numeric(df[y_col], errors='coerce').fillna(0)
                    
                    # Create bar chart
                    bars = plt.bar(x_data, y_data, color='steelblue', alpha=0.8, edgecolor='navy')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height, 
                               f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    
                elif numeric_cols:
                    # Only numeric data - use index as x-axis
                    y_col = numeric_cols[0]
                    y_data = pd.to_numeric(df[y_col], errors='coerce').fillna(0)
                    
                    plt.bar(range(len(df)), y_data, color='steelblue', alpha=0.8)
                    plt.xlabel('Records', fontsize=12)
                    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
                else:
                    return "❌ No suitable numeric data found for bar chart"
            
            elif chart_type == "pie":
                numeric_cols = df.select_dtypes(include=['number', 'int64', 'float64']).columns.tolist()
                text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                
                if numeric_cols and text_cols and len(df) <= 10:
                    # Prepare data for pie chart
                    labels = df[text_cols[0]].astype(str)
                    values = pd.to_numeric(df[numeric_cols[0]], errors='coerce').fillna(0)
                    
                    # Filter out zero values
                    non_zero_mask = values > 0
                    labels = labels[non_zero_mask]
                    values = values[non_zero_mask]
                    
                    if len(values) == 0:
                        return "❌ No non-zero values for pie chart"
                    
                    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')  # Equal aspect ratio ensures circular pie
                else:
                    return "❌ Pie chart requires categorical and numeric data with ≤10 categories"
            
            # Set title
            if not title:
                title = f'{chart_type.title()} Chart - Data Analysis'
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Improve layout
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = f"chart_{chart_type}_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Close the figure to free memory
            
            print(f"✅ Chart saved successfully: {chart_path}")
            return f"✅ Chart saved: {chart_path}"
            
        except Exception as e:
            plt.close()  # Close any open figures
            error_msg = f"❌ Chart generation failed: {str(e)}"
            print(error_msg)
            return error_msg

# Main CrewAI System using kickoff
class CrewAISQLSystem:
    def __init__(self, llm_provider: str = "auto", selected_databases: List[str] = None):
        self.selected_databases = selected_databases or ["sample.db"]
        self.llm_provider = llm_provider
        self.llm = get_llm(llm_provider)
        self.last_execution_result = None
        
        # Initialize components
        self.role_manager = RoleManager()
        self.memory = ConversationMemory()
        self.file_manager = FileDataManager(self.selected_databases)
        
        # Add default users
        self.role_manager.add_user("admin", UserRole.ADMIN)
        self.role_manager.add_user("analyst", UserRole.ANALYST)
        self.role_manager.add_user("viewer", UserRole.VIEWER)
        
        # Initialize tools with dependencies
        self.sql_generator = SQLGeneratorTool()
        self.sql_validator = SQLValidatorTool()
        self.sql_executor = SQLExecutorTool()
        self.chart_generator = ChartGeneratorTool()
        self.sql_converter = SQLConverterTool()  # Add back SQL converter
        
        # Set tool dependencies
        for tool in [self.sql_generator, self.sql_validator, self.sql_executor]:
            tool._file_manager = self.file_manager
            tool._role_manager = self.role_manager
            tool._schema_inspector = self.file_manager.schema_inspector
            tool._memory = self.memory
            tool._llm = self.llm
            tool._system_ref = self
        
        # Create agents
        self._create_agents()
        self._verify_database_connections()

    def _create_agents(self):
        """Create the 4 specialized agents"""
        self.sql_architect = Agent(
            role='Senior SQL Database Architect',
            goal='Generate perfect SQL queries using comprehensive schema knowledge and best practices',
            backstory=f"""You are a world-class database architect with deep knowledge of {len(self.selected_databases)} 
                        database(s): {', '.join(self.selected_databases)}. You create precise, efficient SQL queries 
                        that respect schema constraints and user permissions.""",
            tools=[self.sql_generator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        self.security_specialist = Agent(
            role='Database Security and Validation Specialist',
            goal='Ensure all SQL queries are secure, valid, and comply with role-based access controls',
            backstory="""You are a cybersecurity expert specializing in database security. You validate SQL queries 
                        against schemas, check for security vulnerabilities, and ensure role-based access compliance.""",
            tools=[self.sql_validator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        self.data_analyst = Agent(
            role='Senior Data Execution Engineer',
            goal='Execute validated SQL queries efficiently and return clean, formatted results',
            backstory=f"""You are a senior data engineer who executes queries across {len(self.selected_databases)} 
                        database(s) and file sources. You ensure reliable query execution and apply appropriate 
                        role-based filtering to results.""",
            tools=[self.sql_executor],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        self.visualization_expert = Agent(
            role='Data Visualization Specialist',
            goal='Create insightful, professional visualizations from query results',
            backstory="""You are a data visualization expert who transforms raw data into compelling charts and graphs. 
                        You understand which visualization types work best for different data patterns and user needs.""",
            tools=[self.chart_generator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def process_request(self, user_request: str, username: str = "admin", create_chart: bool = False, 
                       chart_type: str = "bar", **kwargs) -> Dict[str, Any]:
        """Process request using all 4 agents with CrewAI kickoff"""
        try:
            if not self.role_manager.check_permission(username, "read"):
                return {
                    "success": False,
                    "error": f"Permission denied: User '{username}' does not have read permissions",
                    "user": username,
                    "request": user_request,
                    "access_denied": True
                }
            
            # Get context for agents
            schema_context = self.file_manager.schema_inspector.get_schema_context_for_llm(username, self.role_manager)
            memory_context = self.memory.get_context(username, user_request)
            
            # Create tasks for each agent
            tasks = []
            
            # Task 1: SQL Generation
            sql_task = Task(
                description=f"""Generate a SQL query for this request: "{user_request}"
                
                Schema Context: {schema_context}
                Memory Context: {memory_context}
                Username: {username}
                
                Generate only the SQL query, nothing else.""",
                agent=self.sql_architect,
                expected_output="A valid SQL query string"
            )
            tasks.append(sql_task)
            
            # Task 2: SQL Validation
            validation_task = Task(
                description=f"""Validate the SQL query generated by the SQL Architect against the database schema.
                
                Username: {username}
                
                Check for:
                1. Table and column existence
                2. Role-based access permissions
                3. SQL syntax correctness
                
                Return validation status and any issues found.""",
                agent=self.security_specialist,
                expected_output="Validation result with status and any issues",
                context=[sql_task]
            )
            tasks.append(validation_task)
            
            # Task 3: SQL Execution
            execution_task = Task(
                description=f"""Execute the validated SQL query and return formatted results.
                
                Username: {username}
                
                Apply role-based filtering and return clean, tabulated results.""",
                agent=self.data_analyst,
                expected_output="Query execution results with data table",
                context=[sql_task, validation_task]
            )
            tasks.append(execution_task)
            
            # Task 4: Chart Generation (conditional)
            if create_chart:
                chart_task = Task(
                    description=f"""Generate a {chart_type} chart from the query execution results.
                    
                    Chart Type: {chart_type}
                    
                    Create a professional visualization that highlights key insights from the data.""",
                    agent=self.visualization_expert,
                    expected_output="Chart generation result with file path",
                    context=[execution_task]
                )
                tasks.append(chart_task)
            
            # Create and execute crew
            crew = Crew(
                agents=[self.sql_architect, self.security_specialist, self.data_analyst] + 
                       ([self.visualization_expert] if create_chart else []),
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            print(f"🚀 Starting CrewAI kickoff with {len(tasks)} tasks...")
            
            # Execute the crew workflow
            result = crew.kickoff()
            
            print(f"✅ CrewAI kickoff completed")
            
            # Check if the result contains access denied information
            execution_data = self.last_execution_result or {}
            sql_query = execution_data.get('sql_query', '')
            
            # Check for access denied in the data
            if execution_data and 'data' in execution_data:
                data = execution_data['data']
                if (data and len(data) > 0 and 
                    isinstance(data[0], dict) and 
                    'error_message' in data[0] and 
                    ('access denied' in data[0]['error_message'].lower() or 
                     'does not have access' in data[0]['error_message'].lower())):
                    
                    # This is an access denied scenario - return as "successful" so UI shows the message
                    return {
                        "success": True,  # Mark as successful so UI displays the message
                        "execution_data": execution_data,
                        "sql_query": sql_query,
                        "user": username,
                        "request": user_request,
                        "agents_used": len(tasks),
                        "llm_provider": self.llm_provider,
                        "access_denied": True
                    }
            
            # Store conversation in memory
            self.memory.add_conversation(
                username=username,
                request=user_request,
                sql_query=sql_query,
                success=bool(self.last_execution_result),
                user_role=self.role_manager.get_user_role(username).value if self.role_manager.get_user_role(username) else "admin"
            )
            
            # Return comprehensive result
            response = {
                "success": True,
                "crew_result": str(result),
                "execution_data": execution_data,
                "sql_query": sql_query,
                "user": username,
                "request": user_request,
                "agents_used": len(tasks),
                "llm_provider": self.llm_provider
            }
            
            # Add chart info if created
            if create_chart and self.last_execution_result:
                chart_files = [f for f in os.listdir('.') if f.startswith('chart_') and f.endswith('.png')]
                if chart_files:
                    latest_chart = sorted(chart_files, key=lambda x: os.path.getmtime(x), reverse=True)[0]
                    response["execution_data"]["chart_path"] = latest_chart
            
            return response
            
        except Exception as e:
            logger.error(f"CrewAI processing error: {e}")
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "user": username,
                "request": user_request
            }

    def _verify_database_connections(self):
        """Verify database connections"""
        print(f"\n🔍 Verifying {len(self.selected_databases)} database(s)...")
        for db_name in self.selected_databases:
            db_path = os.path.abspath(db_name)
            if not os.path.exists(db_path):
                if db_name == "sample.db":
                    self._create_sample_database(db_path)
                else:
                    print(f"❌ Database not found: {db_path}")
            else:
                print(f"✅ Database verified: {db_name}")

    def _create_sample_database(self, db_path: str):
        """Create sample database"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''CREATE TABLE employees (
                id INTEGER PRIMARY KEY, name TEXT, department TEXT, 
                salary REAL, hire_date DATE, manager_id INTEGER, status TEXT DEFAULT 'active'
            )''')
            
            cursor.execute('''CREATE TABLE departments (
                id INTEGER PRIMARY KEY, name TEXT, budget REAL, location TEXT
            )''')
            
            # Insert sample data
            employees = [
                (1, "John Doe", "Engineering", 75000, "2022-01-15", None, "active"),
                (2, "Jane Smith", "Marketing", 65000, "2021-03-20", None, "active"),
                (3, "Bob Johnson", "Engineering", 80000, "2020-07-10", 1, "active"),
                (4, "Alice Brown", "HR", 60000, "2023-02-01", None, "active"),
                (5, "Charlie Wilson", "Engineering", 72000, "2021-11-05", 1, "active")
            ]
            
            departments = [
                (1, "Engineering", 500000, "Building A"),
                (2, "Marketing", 300000, "Building B"),
                (3, "HR", 200000, "Building C")
            ]
            
            cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", employees)
            cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments)
            
            conn.commit()
            conn.close()
            print(f"✅ Created sample database: {db_path}")
        except Exception as e:
            print(f"❌ Failed to create database: {e}")

    # Additional utility methods
    def register_file(self, name: str, file_path: str, username: str = "admin") -> Dict[str, Any]:
        """Register a file with the system"""
        try:
            if not self.role_manager.check_permission(username, "register_files"):
                return {
                    "success": False,
                    "message": f"Permission denied: User '{username}' cannot upload files"
                }
            
            result = self.file_manager.register_file(name, file_path, username)
            
            if result["success"]:
                # Refresh schema inspector to include new table
                self.file_manager.refresh_schema_cache()
                print(f"✅ File '{name}' registered successfully by user '{username}'")
            
            return result
        except Exception as e:
            return {
                "success": False,
                "message": f"File registration failed: {str(e)}"
            }

    def get_uploaded_files_info(self, username: str = "admin") -> Dict[str, Any]:
        """Get information about uploaded files"""
        try:
            if not self.role_manager.check_permission(username, "read"):
                return {
                    "success": False,
                    "message": "Permission denied"
                }
            
            files_info = self.file_manager.get_uploaded_files_info()
            return {
                "success": True,
                "files": files_info,
                "count": len(files_info)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get files info: {str(e)}"
            }

    def delete_uploaded_file(self, table_name: str, username: str = "admin") -> Dict[str, Any]:
        """Delete an uploaded file"""
        try:
            if not self.role_manager.check_permission(username, "register_files"):
                return {
                    "success": False,
                    "message": "Permission denied"
                }
            
            result = self.file_manager.delete_uploaded_file(table_name)
            
            if result["success"]:
                # Refresh schema
                self.file_manager.refresh_schema_cache()
            
            return result
        except Exception as e:
            return {
                "success": False,
                "message": f"File deletion failed: {str(e)}"
            }

    def refresh_database_schema(self, username: str = "admin") -> Dict[str, Any]:
        """Refresh database schema including uploaded files"""
        try:
            if not self.role_manager.check_permission(username, "read"):
                return {
                    "success": False,
                    "message": "Permission denied"
                }
            
            # Refresh schema cache
            self.file_manager.refresh_schema_cache()
            
            # Get updated table info
            tables_result = self.get_database_tables_info(username)
            
            return {
                "success": True,
                "message": "Database schema refreshed successfully",
                "tables_info": tables_result
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Schema refresh failed: {str(e)}"
            }

    def convert_sql_to_target_db(self, sql_query: str, target_db: str) -> Dict[str, Any]:
        """Convert SQL query to target database dialect"""
        try:
            converted_query = self.sql_converter._run(sql_query, target_db)
            if converted_query.startswith("Error:") or converted_query.startswith("Conversion error:"):
                return {
                    "success": False,
                    "error": converted_query,
                    "original_query": sql_query,
                    "target_database": target_db
                }
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

    def export_results(self, data: List[Dict] = None, filename: str = None) -> str:
        """Export query results to Excel with enhanced metadata"""
        if not filename:
            filename = f"crewai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        try:
            # Use provided data or last execution result
            if not data and self.last_execution_result and 'data' in self.last_execution_result:
                data = self.last_execution_result['data']
            
            if not data:
                return "No data available to export"
            
            # Create Excel file with multiple sheets
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main data sheet
                pd.DataFrame(data).to_excel(writer, sheet_name='Query Results', index=False)
                
                # Metadata sheet
                metadata = {
                    'Generated By': ['Gainwell SQL Analysis System'],
                    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Records': [len(data)],
                    'LLM Provider': [self.llm_provider],
                    'Databases Used': [', '.join(self.selected_databases)]
                }
                
                if self.last_execution_result:
                    metadata['SQL Query'] = [self.last_execution_result.get('sql_query', 'N/A')]
                    metadata['Columns'] = [', '.join(self.last_execution_result.get('columns', []))]
                    metadata['Execution Success'] = [self.last_execution_result.get('success', False)]
                
                pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)
            
            return f"Results exported to: {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"

    def get_database_tables_info(self, username: str = "admin") -> Dict[str, Any]:
        """Get database tables information"""
        try:
            all_tables = self.file_manager.get_all_databases_tables()
            user_role = self.role_manager.get_user_role(username)
            
            filtered_tables = {}
            for db_name, tables in all_tables.items():
                accessible_tables = []
                for table_info in tables:
                    if self.role_manager.check_table_access(username, table_info["name"]):
                        accessible_tables.append(table_info)
                filtered_tables[db_name] = accessible_tables
            
            return {
                "success": True,
                "databases": filtered_tables,
                "user_role": user_role.value if user_role else "unknown"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# Application wrapper
class CrewAIApp:
    def __init__(self, llm_provider: str = "auto", selected_databases: List[str] = None):
        print(f"🏗️  Initializing CrewAI Application with LLM: {llm_provider}")
        self.system = CrewAISQLSystem(llm_provider, selected_databases or ["sample.db"])
        print("✅ CrewAI Application ready!")

def main():
    """Main function"""
    try:
        app = CrewAIApp(llm_provider="auto", selected_databases=["sample.db"])
        print("\n✅ Gainwell SQL System initialized successfully!")
        return app
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")

if __name__ == "__main__":
    main()
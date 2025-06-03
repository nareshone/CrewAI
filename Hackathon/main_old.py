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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("📝 Or set environment variables manually")

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Memory system for learning
class ConversationMemory:
    def __init__(self, memory_file: str = "conversation_memory.pkl"):
        self.memory_file = memory_file
        self.conversations = {}
        self.query_history = []
        self.successful_patterns = {}
        self.load_memory()
    
    def save_memory(self):
        try:
            memory_data = {
                'conversations': self.conversations,
                'query_history': self.query_history,
                'successful_patterns': self.successful_patterns
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
        
        # Learn successful patterns
        if success:
            pattern_key = self._extract_pattern(request)
            if pattern_key not in self.successful_patterns:
                self.successful_patterns[pattern_key] = []
            self.successful_patterns[pattern_key].append(sql_query)
        
        self.save_memory()
    
    def _extract_pattern(self, request: str) -> str:
        """Extract query pattern for learning"""
        keywords = ['count', 'average', 'sum', 'group', 'join', 'where', 'order']
        found = [kw for kw in keywords if kw in request.lower()]
        return '_'.join(found) if found else 'general'
    
    def get_context(self, username: str, request: str) -> str:
        """Get relevant context for LLM"""
        context = ""
        
        # Recent user queries
        if username in self.conversations:
            recent = self.conversations[username][-3:]
            if recent:
                context += "Recent queries by this user:\n"
                for entry in recent:
                    context += f"Request: {entry['request']}\nSQL: {entry['sql_query']}\nSuccess: {entry['success']}\n\n"
        
        # Similar successful patterns
        pattern = self._extract_pattern(request)
        if pattern in self.successful_patterns:
            examples = self.successful_patterns[pattern][-2:]  # Last 2 examples
            if examples:
                context += "Similar successful queries:\n"
                for sql in examples:
                    context += f"SQL: {sql}\n"
        
        return context

# File manager for flat files
class FileDataManager:
    def __init__(self, db_path: str = "sample.db"):
        self.data_sources = {}
        self.temp_db_path = "temp_file_data.db"
        self.main_db_path = os.path.abspath(db_path)  # Store main database path
    
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
    
    def get_schema_info(self) -> str:
        """Get comprehensive schema information for LLM"""
        schema_info = "=== AVAILABLE DATA SOURCES ===\n\n"
        
        # Database tables with detailed schema - use the correct path
        abs_db_path = self.main_db_path
        print(f"🔍 Schema check - looking for database at: {abs_db_path}")
        
        try:
            if not os.path.exists(abs_db_path):
                error_msg = f"❌ Database file not found: {abs_db_path}"
                schema_info += error_msg + "\n"
                print(error_msg)
                return schema_info
            
            conn = sqlite3.connect(abs_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"🔍 Schema check - found tables: {[t[0] for t in tables]}")
            
            if tables:
                schema_info += f"DATABASE TABLES (at {abs_db_path}):\n"
                for table in tables:
                    table_name = table[0]
                    print(f"📊 Reading schema for table: {table_name}")
                    
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                    sample_data = cursor.fetchall()
                    
                    schema_info += f"\n=== TABLE: {table_name} ===\n"
                    schema_info += "COLUMNS:\n"
                    for col in columns:
                        col_name, col_type = col[1], col[2]
                        pk_indicator = " (PRIMARY KEY)" if col[5] == 1 else ""
                        schema_info += f"  {col_name} ({col_type}){pk_indicator}\n"
                    
                    schema_info += f"\nSAMPLE DATA:\n"
                    if sample_data:
                        # Show column headers
                        col_names = [col[1] for col in columns]
                        schema_info += f"  Headers: {col_names}\n"
                        for i, row in enumerate(sample_data, 1):
                            schema_info += f"  Row {i}: {row}\n"
                    schema_info += "\n"
                    
                    print(f"✅ Schema loaded for {table_name}: {[col[1] for col in columns]}")
            else:
                error_msg = f"❌ No tables found in database: {abs_db_path}"
                schema_info += error_msg + "\n"
                print(error_msg)
            
            conn.close()
        except Exception as e:
            error_msg = f"❌ Error reading database schema: {e}"
            schema_info += error_msg + f"\nDatabase path: {abs_db_path}\n"
            print(error_msg)
        
        # File sources
        if self.data_sources:
            schema_info += "\nFILE DATA SOURCES:\n"
            for name, info in self.data_sources.items():
                schema_info += f"\n=== FILE: {name} ===\n"
                schema_info += f"SQL Table: {info['table_name']}\n"
                schema_info += f"Columns: {', '.join(info['columns'])}\n"
                schema_info += f"Data types: {info['dtypes']}\n"
                schema_info += f"Sample data:\n"
                for i, row in enumerate(info['sample_data'][:2], 1):
                    schema_info += f"  Row {i}: {row}\n"
                schema_info += "\n"
        
        # Add critical SQL generation rules
        schema_info += "\n" + "="*50 + "\n"
        schema_info += "CRITICAL SQL GENERATION RULES:\n"
        schema_info += "="*50 + "\n"
        schema_info += "1. ONLY use column names listed above - DO NOT INVENT NAMES\n"
        schema_info += "2. ONLY use table names listed above\n"
        schema_info += "3. For employees table, available columns are: id, name, department, salary, hire_date, manager_id, status\n"
        schema_info += "4. For departments table, available columns are: id, name, budget, location\n"
        schema_info += "5. DO NOT use: employee_name, department_name, employee_id, department_id\n"
        schema_info += "6. For joins: JOIN departments d ON e.department = d.name\n"
        schema_info += "7. Always include LIMIT clause\n"
        schema_info += "8. Use exact column names from schema above\n"
        schema_info += "="*50 + "\n"
        
        print(f"📋 Final schema info length: {len(schema_info)} characters")
        return schema_info

# CrewAI Tools with proper initialization

class SQLGeneratorTool(BaseTool):
    name: str = "sql_generator"
    description: str = "Generate SQL queries using LLM with schema awareness"
    
    def _run(self, query_description: str, username: str = "admin") -> str:
        """Generate SQL using LLM with full context"""
        
        # Access global instances (we'll set these up)
        try:
            file_manager = getattr(self, '_file_manager', None)
            memory = getattr(self, '_memory', None)
            llm = getattr(self, '_llm', None)
            
            print(f"🔍 SQL Generator - Starting query generation for: {query_description}")
            
            if file_manager and memory:
                # Get real-time schema info
                schema_info = file_manager.get_schema_info()
                print(f"📊 Schema info length: {len(schema_info)} characters")
                
                # Show first part of schema to verify it's working
                schema_preview = schema_info[:500] + "..." if len(schema_info) > 500 else schema_info
                print(f"📋 Schema preview: {schema_preview}")
                
                context = memory.get_context(username, query_description)
                
                if llm:
                    print(f"🧠 Using LLM for query generation")
                    result = self._generate_with_llm(query_description, schema_info, context, llm)
                else:
                    print(f"⚠️  No LLM available, using fallback")
                    result = self._generate_fallback(query_description)
            else:
                print(f"⚠️  File manager or memory not available, using fallback")
                result = self._generate_fallback(query_description)
            
            print(f"✅ Generated SQL: {result}")
            return result
                
        except Exception as e:
            print(f"❌ SQL generation error: {e}")
            logger.error(f"SQL generation error: {e}")
            return self._generate_fallback(query_description)
    
    def _generate_with_llm(self, query_description: str, schema_info: str, context: str, llm) -> str:
        """Generate SQL using LLM"""
        
        prompt = f"""You are an expert SQL developer. Generate a precise, syntactically correct SQL query.

{schema_info}

USER CONTEXT:
{context}

USER REQUEST: {query_description}

CRITICAL REQUIREMENTS:
1. Generate ONLY the SQL query, no explanations or markdown
2. Use EXACT table and column names from the schema above - DO NOT MODIFY THEM
3. The employees table has columns: id, name, department, salary, hire_date, manager_id, status
4. The departments table has columns: id, name, budget, location
5. DO NOT use column names like department_name, department_id, employee_name, etc.
6. Use the actual column names shown in the schema: 'name', 'department', 'salary', etc.
7. For joins between employees and departments, use: employees.department = departments.name
8. ALWAYS use LIMIT to prevent excessive results (LIMIT 50 or less)
9. Use proper SQL syntax with correct table and column references

CORRECT EXAMPLES:
- SELECT name, department, salary FROM employees LIMIT 20;
- SELECT department, COUNT(*) as count FROM employees GROUP BY department;
- SELECT AVG(salary) as avg_salary FROM employees;
- SELECT e.name, e.salary, d.budget FROM employees e JOIN departments d ON e.department = d.name LIMIT 20;

IMPORTANT: Only use the exact column names listed in the schema above. Do not assume or invent column names.

Generate the SQL query now:"""

        try:
            if hasattr(llm, 'invoke'):
                response = llm.invoke(prompt)
                sql_query = response.content if hasattr(response, 'content') else str(response)
            else:
                sql_query = llm(prompt)
            
            # Clean the response more aggressively
            sql_query = sql_query.strip()
            # Remove any markdown formatting
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            # Remove any explanatory text before/after the query
            lines = sql_query.split('\n')
            sql_lines = []
            for line in lines:
                line = line.strip()
                # Look for SQL statements
                if line and (line.upper().startswith('SELECT') or 
                           line.upper().startswith('WITH') or
                           (sql_lines and not line.endswith(';') and not line.startswith('Note:') and not line.startswith('This'))):
                    sql_lines.append(line)
                elif sql_lines and line.endswith(';'):
                    sql_lines.append(line)
                    break
                elif line.upper().startswith('SELECT'):
                    sql_lines = [line]  # Start fresh if we find a new SELECT
            
            if sql_lines:
                sql_query = ' '.join(sql_lines)
            
            # Final validation - ensure we have a proper SQL query
            sql_query = sql_query.strip()
            if not sql_query.upper().startswith('SELECT'):
                logger.warning(f"Generated query doesn't start with SELECT: {sql_query}")
                return self._generate_fallback(query_description)
            
            logger.info(f"LLM generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback(query_description)
    
    def _generate_fallback(self, query_description: str) -> str:
        """Fallback SQL generation without LLM"""
        desc = query_description.lower()
        
        if "count" in desc and "department" in desc:
            return "SELECT department, COUNT(*) as count FROM employees GROUP BY department ORDER BY count DESC LIMIT 10;"
        elif "count" in desc:
            return "SELECT COUNT(*) as total FROM employees;"
        elif "average" in desc and "salary" in desc:
            if "department" in desc:
                return "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC LIMIT 10;"
            else:
                return "SELECT AVG(salary) as avg_salary FROM employees;"
        elif "high" in desc and "salary" in desc:
            return "SELECT name, department, salary FROM employees WHERE salary > 70000 ORDER BY salary DESC LIMIT 20;"
        elif "top" in desc and "salary" in desc:
            return "SELECT name, department, salary FROM employees ORDER BY salary DESC LIMIT 10;"
        elif "all" in desc and "employees" in desc:
            return "SELECT name, department, salary FROM employees ORDER BY name LIMIT 50;"
        elif "department" in desc and ("budget" in desc or "total" in desc):
            return "SELECT name as department_name, budget FROM departments ORDER BY budget DESC LIMIT 10;"
        elif "join" in desc or ("employees" in desc and "departments" in desc):
            return "SELECT e.name, e.department, e.salary, d.budget FROM employees e JOIN departments d ON e.department = d.name LIMIT 20;"
        else:
            return "SELECT name, department, salary FROM employees LIMIT 10;"

class SQLExecutorTool(BaseTool):
    name: str = "sql_executor"
    description: str = "Execute SQL queries with validation and error handling"
    
    def _run(self, sql_query: str, username: str = "admin", db_path: str = None) -> Dict[str, Any]:
        """Execute SQL with comprehensive validation"""
        
        # Access role manager and database path
        role_manager = getattr(self, '_role_manager', None)
        stored_db_path = getattr(self, '_db_path', None)
        
        # Use the stored database path if available, otherwise use parameter
        if stored_db_path:
            actual_db_path = stored_db_path
        elif db_path:
            actual_db_path = db_path
        else:
            actual_db_path = os.path.abspath('sample.db')
        
        print(f"🔍 SQL Executor using database: {actual_db_path}")
        
        # Permission check
        if role_manager and not role_manager.check_permission(username, "execute"):
            return {"success": False, "error": "Permission denied"}
        
        # Verify database file exists
        if not os.path.exists(actual_db_path):
            print(f"❌ Database not found at: {actual_db_path}")
            print(f"📁 Current directory: {os.getcwd()}")
            print(f"📋 Available .db files: {[f for f in os.listdir('.') if f.endswith('.db')]}")
            return {"success": False, "error": f"Database file not found: {actual_db_path}"}
        
        # Basic SQL validation
        sql_lower = sql_query.lower().strip()
        dangerous_ops = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
        
        for op in dangerous_ops:
            if op in sql_lower:
                return {"success": False, "error": f"Dangerous operation '{op}' not allowed"}
        
        if not sql_lower.startswith('select'):
            return {"success": False, "error": "Only SELECT queries allowed"}
        
        try:
            # Execute query with absolute path
            print(f"🔍 Executing SQL on database: {actual_db_path}")
            
            # Verify database has tables before executing
            conn = sqlite3.connect(actual_db_path)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"📊 Available tables: {tables}")
            
            if not tables:
                conn.close()
                return {"success": False, "error": "Database exists but contains no tables"}
            
            # Execute the actual query
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            print(f"✅ Query executed successfully, returned {len(df)} rows")
            
            # Store the execution result globally for export functionality
            execution_result = {
                "success": True,
                "data": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "sql_query": sql_query,
                "database_path": actual_db_path
            }
            
            # Try to store in the system instance for export
            try:
                # This is a hack to get access to the system instance
                import inspect
                frame = inspect.currentframe()
                while frame:
                    if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'last_execution_result'):
                        frame.f_locals['self'].last_execution_result = execution_result
                        print(f"📦 Stored execution result for export")
                        break
                    frame = frame.f_back
            except:
                pass  # If we can't store it, that's okay
            
            return execution_result
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            print(f"❌ SQL execution error: {error_msg}")
            print(f"🔍 Query: {sql_query}")
            print(f"🔍 Database: {actual_db_path}")
            
            # Additional debugging
            try:
                if os.path.exists(actual_db_path):
                    conn = sqlite3.connect(actual_db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    print(f"🔍 Debug - Tables in database: {[t[0] for t in tables]}")
                    conn.close()
                else:
                    print(f"🔍 Debug - Database file does not exist: {actual_db_path}")
            except Exception as debug_e:
                print(f"🔍 Debug error: {debug_e}")
            
            return {"success": False, "error": error_msg}

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
    description: str = "Validate and automatically correct SQL queries to match actual database schema"
    
    def _run(self, sql_query: str) -> str:
        """Validate and auto-correct SQL query for schema compliance"""
        
        print(f"🔍 Query Validator - Input SQL: {sql_query}")
        
        corrected_query = sql_query.strip()
        corrections_made = []
        
        # Schema correction mappings
        corrections = {
            # Column name corrections
            'employee_id': 'id',
            'employee_name': 'name',
            'first_name': 'name',
            'last_name': 'name',
            'department_name': 'name',
            'department_id': 'id',
            
            # Table alias corrections
            'e.employee_id': 'e.id',
            'e.employee_name': 'e.name',
            'e.first_name': 'e.name',
            'e.last_name': 'e.name',
            'd.department_name': 'd.name',
            'd.department_id': 'd.id',
            
            # Full table references
            'employees.employee_id': 'employees.id',
            'employees.employee_name': 'employees.name',
            'employees.first_name': 'employees.name',
            'employees.last_name': 'employees.name',
            'departments.department_name': 'departments.name',
            'departments.department_id': 'departments.id'
        }
        
        # Apply corrections
        for wrong, correct in corrections.items():
            if wrong in corrected_query:
                corrected_query = corrected_query.replace(wrong, correct)
                corrections_made.append(f"{wrong} → {correct}")
        
        # Fix common JOIN patterns
        join_corrections = {
            'ON employees.department_id = departments.department_id': 'ON employees.department = departments.name',
            'ON employees.dept_id = departments.id': 'ON employees.department = departments.name',
            'ON e.department_id = d.department_id': 'ON e.department = d.name',
            'ON e.dept_id = d.id': 'ON e.department = d.name'
        }
        
        for wrong_join, correct_join in join_corrections.items():
            if wrong_join in corrected_query:
                corrected_query = corrected_query.replace(wrong_join, correct_join)
                corrections_made.append(f"JOIN: {wrong_join} → {correct_join}")
        
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

# Main CrewAI SQL Analysis System
class CrewAISQLSystem:
    def __init__(self, db_path: str = "sample.db"):
        self.db_path = os.path.abspath(db_path)  # Use absolute path
        self.llm = get_llm()
        
        # Initialize components with consistent database path
        self.role_manager = RoleManager()
        self.memory = ConversationMemory()
        self.file_manager = FileDataManager(self.db_path)  # Pass database path
        
        # Setup users
        self.role_manager.add_user("admin", UserRole.ADMIN)
        self.role_manager.add_user("analyst", UserRole.ANALYST)
        self.role_manager.add_user("viewer", UserRole.VIEWER)
        
        # Initialize tools without problematic constructors
        self.sql_generator = SQLGeneratorTool()
        self.sql_executor = SQLExecutorTool()
        self.chart_generator = ChartGeneratorTool()
        self.query_validator = QueryValidatorTool()
        
        # Set attributes on tools (workaround for Pydantic field validation)
        self.sql_generator._file_manager = self.file_manager
        self.sql_generator._memory = self.memory
        self.sql_generator._llm = self.llm
        
        self.sql_executor._role_manager = self.role_manager
        self.sql_executor._db_path = self.db_path  # Set the database path
        
        self.query_validator._llm = self.llm
        
        # Create CrewAI agents
        self._create_agents()
        
    def _verify_database_connection(self):
        """Verify database connection immediately after system initialization"""
        print(f"\n🔍 Verifying database connection...")
        print(f"📂 Expected database path: {self.db_path}")
        
        if not os.path.exists(self.db_path):
            print(f"❌ Database file not found at: {self.db_path}")
            print(f"📁 Current directory: {os.getcwd()}")
            print(f"📋 Files in current directory: {[f for f in os.listdir('.') if f.endswith('.db')]}")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                print(f"✅ Database connection verified!")
                print(f"📊 Found tables: {tables}")
                
                # Test a simple query
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
    
    def repair_database(self):
        """Repair or recreate database if needed"""
        print(f"\n🔧 Attempting to repair database...")
        
        # Remove corrupted database
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
                print(f"🗑️  Removed corrupted database")
            except Exception as e:
                print(f"⚠️  Could not remove database: {e}")
        
        # Recreate database using the same logic as CrewAIApp
        return self._create_fresh_database()
    
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
        """Create specialized CrewAI agents"""
        
        self.sql_architect = Agent(
            role='Senior SQL Database Architect',
            goal='Generate perfect, schema-compliant SQL queries that precisely match user requirements using only existing database columns',
            backstory="""You are a world-class database architect with 20+ years of experience. You NEVER invent 
                        column names and ALWAYS use the exact schema provided. You understand that using wrong column 
                        names breaks queries, so you strictly follow the actual database schema. You know that our 
                        employees table has: id, name, department, salary, hire_date, manager_id, status and our 
                        departments table has: id, name, budget, location.""",
            tools=[self.sql_generator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
        
        self.security_specialist = Agent(
            role='Database Security and Quality Assurance Specialist',
            goal='Ensure all SQL queries are secure, performant, and follow enterprise best practices',
            backstory="""You are a cybersecurity expert specializing in database security with deep expertise in 
                        SQL injection prevention, query optimization, and enterprise database standards. You have 
                        prevented numerous security breaches and optimized thousands of queries for peak performance.""",
            tools=[self.query_validator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
        
        self.data_analyst = Agent(
            role='Senior Data Analytics Engineer and SQL Execution Specialist',
            goal='Execute corrected SQL queries and extract meaningful business insights from the results',
            backstory="""You are a senior data engineer who specializes in executing SQL queries and analyzing results. 
                        You understand that security specialists provide corrected SQL queries that must be used instead 
                        of original queries. When you receive a validation response with "CORRECTED SQL:", you extract 
                        and execute only that corrected SQL. You never execute queries with wrong column names.""",
            tools=[self.sql_executor],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
        
        self.visualization_expert = Agent(
            role='Data Visualization and Business Intelligence Expert',
            goal='Create compelling, publication-ready visualizations that effectively communicate data insights',
            backstory="""You are a data visualization expert with a background in design and statistics. You have 
                        created dashboards and reports for C-level executives and understand how to present complex 
                        data in clear, actionable visual formats that drive business decisions.""",
            tools=[self.chart_generator],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
    
    def process_request(self, user_request: str, username: str = "admin", 
                       create_chart: bool = False, chart_type: str = "bar") -> Dict[str, Any]:
        """Process request using full CrewAI workflow"""
        
        if not self.role_manager.check_permission(username, "read"):
            return {"error": "Permission denied"}
        
        print(f"\n🚀 STARTING CREWAI WORKFLOW")
        print(f"📝 Request: {user_request}")
        print(f"👤 User: {username}")
        print(f"📊 Visualization: {create_chart} ({chart_type})")
        print("="*70)
        
        # Task 1: SQL Generation with Intelligence
        sql_task = Task(
            description=f"""
            Generate an advanced SQL query for: "{user_request}"
            
            Use your expertise to:
            - Analyze available database schema and file sources
            - Consider user's query history and successful patterns
            - Create syntactically perfect, optimized SQL
            - Handle complex business logic and edge cases
            - Apply appropriate joins, aggregations, and filters
            - Ensure query performance and scalability
            
            User context: {username}
            
            Return only the clean SQL query ready for execution.
            """,
            agent=self.sql_architect,
            expected_output="A production-ready, optimized SQL query"
        )
        
        # Task 2: Security and Quality Validation
        validation_task = Task(
            description=f"""
            Perform comprehensive security and quality analysis of the SQL query generated for: "{user_request}"
            
            Validate:
            - SQL injection attack vectors and security vulnerabilities
            - Query syntax correctness and SQL standard compliance
            - Performance optimization opportunities and indexing requirements
            - Business logic correctness and data integrity
            - Enterprise security policy compliance
            - Best practices adherence
            
            If issues are found, provide specific remediation steps.
            Approve the query only if it meets all security and quality standards.
            """,
            agent=self.security_specialist,
            expected_output="Security validation report with approved query or improvement recommendations",
            context=[sql_task]
        )
        
        # Task 3: Query Execution and Analysis
        execution_task = Task(
            description=f"""
            Execute the validated SQL query and perform comprehensive data analysis for user: {username}
            
            Execution protocol:
            - Verify user permissions and access rights
            - Execute query with proper error handling and monitoring
            - Analyze result set for data quality and completeness
            - Generate statistical summaries and key insights
            - Identify any anomalies or interesting patterns
            - Prepare data for potential visualization
            
            Provide detailed execution results with business insights.
            """,
            agent=self.data_analyst,
            expected_output="Query execution results with data analysis and business insights",
            context=[validation_task]
        )
        
        # Task 4: Visualization (conditional)
        tasks = [sql_task, validation_task, execution_task]
        
        if create_chart and self.role_manager.check_permission(username, "read"):
            viz_task = Task(
                description=f"""
                Create a professional {chart_type} visualization from the query results.
                
                Visualization requirements:
                - Analyze data structure and select optimal columns for visualization
                - Apply advanced styling and professional formatting
                - Ensure chart clearly communicates key insights and trends
                - Add informative titles, labels, and legends
                - Use appropriate colors and design principles
                - Create publication-ready, executive-level visualizations
                
                Deliver chart file path and interpretation of visual insights.
                """,
                agent=self.visualization_expert,
                expected_output=f"Professional {chart_type} chart with insight analysis",
                context=[execution_task]
            )
            tasks.append(viz_task)
        
        # Create and execute CrewAI workflow
        crew = Crew(
            agents=[self.sql_architect, self.security_specialist, self.data_analyst, self.visualization_expert],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            manager_llm=self.llm  # Explicitly set the LLM for CrewAI
        )
        
        try:
            print(f"\n🔄 EXECUTING MULTI-AGENT WORKFLOW...")
            print("Agents are collaborating to process your request...")
            
            # Execute the crew
            result = crew.kickoff()
            
            print(f"\n✅ CREWAI WORKFLOW COMPLETED SUCCESSFULLY")
            print("="*70)
            
            # Store in memory for learning
            self.memory.add_conversation(username, user_request, str(result), True)
            
            return {
                "success": True,
                "crew_result": str(result),
                "workflow_completed": True,
                "agents_used": len(tasks),
                "llm_provider": type(self.llm).__name__ if self.llm else "No LLM",
                "user": username,
                "request": user_request
            }
            
        except Exception as e:
            error_msg = f"CrewAI workflow failed: {str(e)}"
            logger.error(error_msg)
            
            # Store failed attempt
            self.memory.add_conversation(username, user_request, error_msg, False)
            
            return {
                "success": False,
                "error": error_msg,
                "llm_available": self.llm is not None,
                "fallback_recommended": True
            }
    
    def human_validation(self, result: Dict, username: str) -> bool:
        """Human validation with CrewAI context"""
        print(f"\n{'='*80}")
        print(f"🤖 CREWAI MULTI-AGENT ANALYSIS RESULTS")
        print(f"{'='*80}")
        print(f"User: {username}")
        print(f"LLM Provider: {result.get('llm_provider', 'Unknown')}")
        print(f"Agents Involved: {result.get('agents_used', 0)}")
        print(f"Workflow Status: {'✅ Completed' if result.get('success') else '❌ Failed'}")
        print(f"{'='*80}")
        
        if result.get('success'):
            print(f"🔍 ANALYSIS RESULTS:")
            print(f"{result.get('crew_result', 'No details available')[:500]}...")
        else:
            print(f"❌ ERROR: {result.get('error', 'Unknown error')}")
        
        print(f"{'='*80}")
        
        while True:
            print("\n🔍 VALIDATION OPTIONS:")
            print("1. ✅ Approve and proceed")
            print("2. 🔄 Request modifications")
            print("3. ❌ Reject and cancel")
            print("4. 📊 Show detailed breakdown")
            print("5. 🔄 Re-run with different approach")
            
            choice = input("Your choice (1-5): ").strip()
            
            if choice == "1":
                print("✅ Results approved!")
                return True
            elif choice == "2":
                feedback = input("What modifications would you like? ")
                print(f"📝 Feedback noted: {feedback}")
                print("In a production system, this would re-run the crew with your feedback")
                return True
            elif choice == "3":
                print("❌ Analysis rejected")
                return False
            elif choice == "4":
                self._show_detailed_breakdown(result)
                continue
            elif choice == "5":
                print("🔄 In production, this would re-run with different parameters")
                return True
            else:
                print("❌ Invalid choice. Please select 1-5.")
    
    def _show_detailed_breakdown(self, result: Dict):
        """Show detailed analysis breakdown"""
        print(f"\n{'='*60}")
        print("📊 DETAILED ANALYSIS BREAKDOWN")
        print(f"{'='*60}")
        print(f"Success: {result.get('success', 'Unknown')}")
        print(f"Workflow Completed: {result.get('workflow_completed', False)}")
        print(f"LLM Provider: {result.get('llm_provider', 'No LLM')}")
        print(f"Agents Used: {result.get('agents_used', 0)}")
        
        if result.get('crew_result'):
            print(f"\nFull Result:")
            print(f"{result['crew_result']}")
        
        if result.get('error'):
            print(f"\nError Details:")
            print(f"{result['error']}")
        
        print(f"{'='*60}")
    
    def register_file(self, name: str, file_path: str, username: str = "admin") -> Dict[str, Any]:
        """Register data file"""
        if not self.role_manager.check_permission(username, "register_files"):
            return {"error": "Permission denied"}
        
        success = self.file_manager.register_file(name, file_path)
        return {"success": success, "message": f"File {name} registered" if success else f"Failed to register {name}"}
    
    def get_stats(self, username: str = None) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'total_queries': len(self.memory.query_history),
            'successful_queries': sum(1 for q in self.memory.query_history if q['success']),
            'llm_available': self.llm is not None,
            'llm_provider': type(self.llm).__name__ if self.llm else "None",
            'registered_files': len(self.file_manager.data_sources)
        }
        
        if username and username in self.memory.conversations:
            user_queries = self.memory.conversations[username]
            stats['user_stats'] = {
                'total': len(user_queries),
                'successful': sum(1 for q in user_queries if q['success']),
                'success_rate': f"{(sum(1 for q in user_queries if q['success']) / len(user_queries) * 100):.1f}%" if user_queries else "0%"
            }
        
        return stats
    
    def export_results(self, data: List[Dict], filename: str = None) -> str:
        """Export query results to Excel"""
        if not filename:
            filename = f"crewai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main data
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name='Query Results', index=False)
                
                # Metadata
                metadata = {
                    'Generated By': ['CrewAI SQL Analysis System'],
                    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Records': [len(data)],
                    'LLM Provider': [type(self.llm).__name__ if self.llm else "No LLM"]
                }
                pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)
            
            return f"Results exported to: {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"

# Application Interface
class CrewAIApp:
    def __init__(self):
        print("🏗️  Initializing CrewAI Application...")
        
        # Create sample data FIRST, before initializing the system
        self._create_sample_data()
        
        # Then initialize the system with the correct database path
        db_path = os.path.abspath("sample.db")
        self.system = CrewAISQLSystem(db_path)
        
        # If database verification failed, try to repair
        if not self.system._verify_database_connection():
            print("🔧 Database verification failed, attempting repair...")
            if self.system.repair_database():
                print("✅ Database repair successful!")
            else:
                print("❌ Database repair failed!")
        
        self._show_llm_status()
    
    def _create_sample_data(self):
        """Create sample database and files"""
        print("📊 Creating sample database and files...")
        
        # Use absolute path for consistency
        db_path = os.path.abspath("sample.db")
        print(f"📂 Database will be created at: {db_path}")
        
        # Remove existing database to recreate with proper schema
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"🗑️  Removed existing database: {db_path}")
            except Exception as e:
                print(f"⚠️  Could not remove existing database: {e}")
        
        # Create fresh database
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            print("🏗️  Creating employees table...")
            # Create employees table with clear column names
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
            
            print("🏗️  Creating departments table...")
            # Create departments table
            cursor.execute("""
                CREATE TABLE departments (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    budget REAL,
                    location TEXT
                )
            """)
            
            print("📝 Inserting sample data...")
            # Insert sample employees data
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
            
            # Insert sample departments data
            departments = [
                (1, "Engineering", 500000, "Building A"),
                (2, "Marketing", 300000, "Building B"),
                (3, "HR", 200000, "Building C"),
                (4, "Finance", 250000, "Building A")
            ]
            
            cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", employees)
            cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments)
            
            conn.commit()
            
            # Verify data was inserted
            cursor.execute("SELECT COUNT(*) FROM employees")
            emp_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM departments")
            dept_count = cursor.fetchone()[0]
            
            print(f"✅ Database created successfully at: {db_path}")
            print(f"   📊 {emp_count} employees inserted")
            print(f"   🏢 {dept_count} departments inserted")
            
            # Show schema for verification
            print(f"\n📋 Database Schema:")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                print(f"   Table '{table_name}': {[col[1] for col in columns]}")
            
            conn.close()
            
            # Test database accessibility
            print(f"\n🔍 Testing database access...")
            test_conn = sqlite3.connect(db_path)
            test_cursor = test_conn.cursor()
            test_cursor.execute("SELECT COUNT(*) FROM employees")
            test_count = test_cursor.fetchone()[0]
            test_conn.close()
            print(f"✅ Database access test successful: {test_count} employees found")
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            import traceback
            traceback.print_exc()
            if 'conn' in locals():
                conn.close()
        
        # Create sample CSV files
        print("\n📁 Creating sample files...")
        
        try:
            # Create products CSV
            products_file = 'sample_products.csv'
            if not os.path.exists(products_file):
                products = {
                    'product_id': [1, 2, 3, 4, 5, 6, 7, 8],
                    'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam', 'Speakers', 'Headphones', 'Tablet'],
                    'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Accessories', 'Electronics'],
                    'price': [999.99, 25.99, 79.99, 299.99, 89.99, 149.99, 199.99, 599.99],
                    'stock': [50, 200, 150, 75, 120, 80, 60, 30],
                    'supplier': ['TechCorp', 'AccessoryInc', 'AccessoryInc', 'TechCorp', 'AccessoryInc', 'AudioCorp', 'AudioCorp', 'TechCorp']
                }
                pd.DataFrame(products).to_csv(products_file, index=False)
                print(f"✅ Created {products_file}")
            
            # Create sales Excel file
            sales_file = 'sample_sales.xlsx'
            if not os.path.exists(sales_file):
                sales = {
                    'sale_id': [1, 2, 3, 4, 5, 6, 7, 8],
                    'product_id': [1, 2, 3, 1, 4, 5, 6, 7],
                    'customer_name': ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Martinez', 'Frank Brown', 'Grace Lee', 'Henry Garcia'],
                    'quantity': [2, 5, 1, 1, 3, 2, 1, 1],
                    'sale_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20', '2024-01-21', '2024-01-22'],
                    'total_amount': [1999.98, 129.95, 79.99, 999.99, 899.97, 179.98, 199.99, 599.99]
                }
                pd.DataFrame(sales).to_excel(sales_file, index=False)
                print(f"✅ Created {sales_file}")
                
        except Exception as e:
            print(f"❌ Error creating sample files: {e}")
        
        print("🎯 Sample data creation completed!")
    
    def verify_database(self):
        """Verify database was created correctly"""
        print(f"\n🔍 VERIFYING DATABASE...")
        
        db_path = "sample.db"
        if not os.path.exists(db_path):
            print(f"❌ Database file does not exist: {db_path}")
            return False
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"📊 Tables found: {tables}")
            
            if 'employees' in tables:
                cursor.execute("SELECT COUNT(*) FROM employees")
                emp_count = cursor.fetchone()[0]
                print(f"👥 Employees: {emp_count} records")
                
                # Show sample data
                cursor.execute("SELECT * FROM employees LIMIT 3")
                sample_employees = cursor.fetchall()
                print(f"📋 Sample employees: {sample_employees}")
            
            if 'departments' in tables:
                cursor.execute("SELECT COUNT(*) FROM departments")
                dept_count = cursor.fetchone()[0]
                print(f"🏢 Departments: {dept_count} records")
                
                # Show sample data
                cursor.execute("SELECT * FROM departments LIMIT 3")
                sample_depts = cursor.fetchall()
                print(f"📋 Sample departments: {sample_depts}")
            
            conn.close()
            print("✅ Database verification successful!")
            return True
            
        except Exception as e:
            print(f"❌ Database verification failed: {e}")
            if 'conn' in locals():
                conn.close()
            return False
    
    def _show_llm_status(self):
        """Show LLM configuration status"""
        print(f"\n{'='*60}")
        print("🧠 LLM CONFIGURATION STATUS")
        print(f"{'='*60}")
        
        if self.system.llm:
            llm_type = type(self.system.llm).__name__
            print(f"✅ LLM Active: {llm_type}")
            
            if "Mock" in llm_type:
                print("⚠️  Mock LLM - Limited functionality, no API key required")
                print("🔧 For full features, set one of these environment variables:")
                print("   export GROQ_API_KEY='your_groq_key'")
                print("   export OPENAI_API_KEY='your_openai_key'")
                print("   export ANTHROPIC_API_KEY='your_claude_key'")
            else:
                print("🤖 CrewAI agents will use LLM for intelligent analysis")
        else:
            print("❌ No LLM configured")
            print("📝 To enable full AI features, set one of these:")
            print("   export GROQ_API_KEY='your_groq_key' (recommended - fast & free)")
            print("   export OPENAI_API_KEY='your_openai_key'")
            print("   export ANTHROPIC_API_KEY='your_claude_key'")
            print("   Or install: ollama pull llama2")
            print("⚠️  System will use basic fallback logic")
        
        print(f"{'='*60}")
    
    def run_interactive(self):
        """Run interactive CrewAI session"""
        
        # First verify database was created properly
        if not self.verify_database():
            print("⚠️  Database verification failed. Attempting to recreate...")
            self._create_sample_data()
            if not self.verify_database():
                print("❌ Could not create database. Some features may not work.")
        
        print(f"\n{'='*70}")
        print("🚀 CREWAI SQL ANALYSIS SYSTEM")
        print(f"{'='*70}")
        print("🤖 Multi-Agent AI System Features:")
        print("✅ Natural language to SQL with LLM intelligence")
        print("✅ Multi-agent collaboration (Architect + Security + Analyst + Viz)")
        print("✅ Advanced security validation and query optimization")
        print("✅ Professional data visualization")
        print("✅ Learning memory system")
        print("✅ Role-based access control")
        print("✅ File data source integration")
        print(f"{'='*70}")
        
        # User login
        username = input("\nEnter username (admin/analyst/viewer): ").strip() or "admin"
        if username not in self.system.role_manager.users:
            print("User not found. Using 'admin'")
            username = "admin"
        
        role = self.system.role_manager.get_user_role(username)
        print(f"✅ Logged in as: {username} ({role.value})")
        
        # Register sample files
        if self.system.role_manager.check_permission(username, "register_files"):
            print("\n📁 Registering sample data files...")
            self.system.register_file("products", "sample_products.csv", username)
            self.system.register_file("sales", "sample_sales.xlsx", username)
            print("✅ Sample files registered!")
        
        # Show system stats
        stats = self.system.get_stats(username)
        print(f"\n📊 SYSTEM STATUS:")
        print(f"🧠 LLM Provider: {stats['llm_provider']}")
        print(f"📈 Total Queries: {stats['total_queries']}")
        print(f"✅ Success Rate: {stats['successful_queries']}/{stats['total_queries']}")
        print(f"📁 Data Sources: {stats['registered_files']} files + database tables")
        
        if 'user_stats' in stats:
            print(f"👤 Your Stats: {stats['user_stats']['success_rate']} success rate")
        
        # Main interaction loop
        while True:
            print(f"\n{'='*70}")
            print("💬 What would you like to analyze?")
            print("🎯 EXAMPLE QUERIES:")
            print("   • 'Show me all employees with their departments'")
            print("   • 'What is the average salary by department?'")
            print("   • 'Count employees hired each year'")
            print("   • 'Show top 5 highest paid employees'")
            print("   • 'Which department has the highest budget?'")
            print("   • 'Show products by category with total stock'")
            print("💡 Type 'help' for more examples, 'verify' to check database, or 'quit' to exit")
            print(f"{'='*70}")
            
            request = input("🔍 Your analysis request: ").strip()
            
            if request.lower() == 'quit':
                print("👋 Thank you for using CrewAI SQL Analysis System!")
                break
            elif request.lower() == 'help':
                self._show_help()
                continue
            elif request.lower() == 'stats':
                self._show_detailed_stats(username)
                continue
            elif request.lower() == 'verify':
                self.verify_database()
                continue
            
            if not request:
                continue
            
            # Visualization options
            create_chart = False
            chart_type = "bar"
            if self.system.role_manager.check_permission(username, "read"):
                viz_choice = input("📊 Create visualization? (y/n): ").strip().lower()
                if viz_choice == 'y':
                    create_chart = True
                    print("Chart types: bar, pie")
                    chart_type = input("Chart type (bar/pie): ").strip() or "bar"
            
            # Process with CrewAI
            print(f"\n🚀 Initiating CrewAI multi-agent analysis...")
            result = self.system.process_request(request, username, create_chart, chart_type)
            
            # Human validation
            if result.get('success'):
                approved = self.system.human_validation(result, username)
                if approved:
                    print(f"\n✅ Analysis completed successfully!")
                    
                    # Export option
                    export = input("\n💾 Export results to Excel? (y/n): ").strip().lower()
                    if export == 'y':
                        filename = input("Filename (Enter for auto-generated): ").strip()
                        export_result = self.system.export_results(
                            [{"message": "CrewAI analysis completed", "status": "success"}],
                            filename if filename else None
                        )
                        print(f"📊 {export_result}")
                else:
                    print("❌ Analysis rejected by user")
            else:
                print(f"❌ CrewAI workflow failed: {result.get('error', 'Unknown error')}")
                if result.get('fallback_recommended'):
                    print("💡 Consider checking your LLM configuration or API keys")
                
                # Offer to verify database
                verify_choice = input("🔍 Verify database integrity? (y/n): ").strip().lower()
                if verify_choice == 'y':
                    self.verify_database()
    
    def _show_help(self):
        """Show comprehensive help"""
        print(f"\n{'='*70}")
        print("📚 CREWAI SQL ANALYSIS HELP")
        print(f"{'='*70}")
        print("\n🎯 QUERY EXAMPLES BY CATEGORY:")
        print("\n📊 Basic Analytics:")
        print("   • 'Show all employees'")
        print("   • 'Count total employees'")
        print("   • 'List all departments'")
        
        print("\n📈 Aggregation Queries:")
        print("   • 'Average salary by department'")
        print("   • 'Total budget per department'")
        print("   • 'Employee count by hire year'")
        
        print("\n🔍 Filtering & Sorting:")
        print("   • 'Employees with salary > 70000'")
        print("   • 'Top 10 highest paid employees'")
        print("   • 'Engineering department employees'")
        
        print("\n🔗 Complex Analysis:")
        print("   • 'Departments with more than 2 employees'")
        print("   • 'Employee salary compared to department average'")
        print("   • 'Products with low stock levels'")
        
        print("\n⚙️ SYSTEM COMMANDS:")
        print("   • 'stats' - Show detailed system statistics")
        print("   • 'help' - Show this help")
        print("   • 'quit' - Exit system")
        
        print(f"\n🤖 CREWAI AGENTS:")
        print("   🏗️  SQL Architect - Generates optimal queries")
        print("   🛡️  Security Specialist - Validates query safety")
        print("   📊 Data Analyst - Executes and analyzes results")
        print("   🎨 Visualization Expert - Creates professional charts")
        print(f"{'='*70}")
    
    def _show_detailed_stats(self, username: str):
        """Show comprehensive system statistics"""
        stats = self.system.get_stats(username)
        
        print(f"\n{'='*70}")
        print("📊 DETAILED SYSTEM STATISTICS")
        print(f"{'='*70}")
        print(f"🧠 LLM Provider: {stats['llm_provider']}")
        print(f"🤖 LLM Available: {'✅ Yes' if stats['llm_available'] else '❌ No'}")
        print(f"📈 Total Queries: {stats['total_queries']}")
        print(f"✅ Successful: {stats['successful_queries']}")
        print(f"❌ Failed: {stats['total_queries'] - stats['successful_queries']}")
        print(f"📁 Registered Files: {stats['registered_files']}")
        
        if 'user_stats' in stats:
            user_stats = stats['user_stats']
            print(f"\n👤 YOUR STATISTICS ({username}):")
            print(f"   Total Queries: {user_stats['total']}")
            print(f"   Successful: {user_stats['successful']}")
            print(f"   Success Rate: {user_stats['success_rate']}")
        
        # Show recent query patterns
        if username in self.system.memory.conversations:
            recent = self.system.memory.conversations[username][-5:]
            print(f"\n📝 RECENT QUERY HISTORY:")
            for i, query in enumerate(recent, 1):
                status = "✅" if query['success'] else "❌"
                print(f"   {i}. {status} {query['request'][:50]}...")
        
        print(f"{'='*70}")

def main():
    """Main application entry point"""
    print("🚀 Initializing CrewAI SQL Analysis System...")
    
    try:
        app = CrewAIApp()
        
        print("\n🎯 Choose mode:")
        print("1. 💬 Interactive Session (Full CrewAI Experience)")
        print("2. 🔧 API Demo (Programmatic Usage)")
        print("3. 📊 Quick Test (Single Query)")
        print("4. 🔧 Setup Guide (Configure LLM)")
        
        choice = input("Your choice (1-4): ").strip()
        
        if choice == "1":
            app.run_interactive()
        elif choice == "2":
            print("\n🔧 API DEMO - CrewAI Workflow:")
            result = app.system.process_request(
                "Show me average salary by department with employee counts",
                username="admin",
                create_chart=True,
                chart_type="bar"
            )
            print(f"📊 API Result: {result}")
        elif choice == "3":
            print("\n📊 QUICK TEST:")
            result = app.system.process_request("Count all employees", "admin")
            if result.get('success'):
                print("✅ CrewAI system working correctly!")
            else:
                print(f"❌ Test failed: {result.get('error')}")
        elif choice == "4":
            show_setup_guide()
        else:
            print("Invalid choice. Starting interactive mode...")
            app.run_interactive()
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        print("💡 Try option 4 for setup guidance")
        import traceback
        traceback.print_exc()

def show_setup_guide():
    """Show detailed setup guide"""
    print(f"\n{'='*70}")
    print("🔧 CREWAI SQL SYSTEM SETUP GUIDE")
    print(f"{'='*70}")
    
    print("\n📋 STEP 1: Install Dependencies")
    print("pip install crewai pandas matplotlib openpyxl")
    
    print("\n🧠 STEP 2: Choose an LLM Provider (pick one):")
    print("\n   🚀 OPTION A: Groq (Recommended - Fast & Free)")
    print("   1. Sign up at: https://console.groq.com/")
    print("   2. Get API key from dashboard")  
    print("   3. pip install langchain-groq")
    print("   4. export GROQ_API_KEY='your_groq_api_key'")
    
    print("\n   🤖 OPTION B: OpenAI")
    print("   1. Sign up at: https://platform.openai.com/")
    print("   2. Get API key from account settings")
    print("   3. pip install langchain-openai")
    print("   4. export OPENAI_API_KEY='your_openai_key'")
    
    print("\n   🏠 OPTION C: Local Ollama (Free, no API key)")
    print("   1. Install: https://ollama.ai/")
    print("   2. ollama pull llama2")
    print("   3. pip install langchain-community")
    
    print("\n   🧠 OPTION D: Anthropic Claude")
    print("   1. Sign up at: https://console.anthropic.com/")
    print("   2. Get API key")
    print("   3. pip install langchain-anthropic")
    print("   4. export ANTHROPIC_API_KEY='your_claude_key'")
    
    print("\n⚡ STEP 3: Test the System")
    print("python your_script.py")
    
    print("\n💡 TROUBLESHOOTING:")
    print("- System works without LLM but with limited features")
    print("- Mock LLM is used as fallback when no provider available")
    print("- Check your API key environment variables")
    print("- Ensure internet connection for cloud providers")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
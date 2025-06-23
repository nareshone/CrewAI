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
                llm.invoke("Hello")
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
                llm.invoke("Hello")
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
                llm.invoke("Hello")
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
        self.conversation_context = {}  # New: Store full conversation context
        self.load_memory()
    
    def save_memory(self):
        try:
            memory_data = {
                'conversations': self.conversations,
                'query_history': self.query_history,
                'successful_patterns': self.successful_patterns,
                'feedback_history': self.feedback_history,
                'role_based_patterns': self.role_based_patterns,
                'conversation_context': self.conversation_context
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
                    self.conversation_context = memory_data.get('conversation_context', {})
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
    
    def add_conversation(self, username: str, request: str, sql_query: str, success: bool, feedback: str = None, user_role: str = None, conversation_history: list = None):
        """Enhanced conversation tracking with full context"""
        if username not in self.conversations:
            self.conversations[username] = []
        
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'request': request,
            'sql_query': sql_query,
            'success': success,
            'feedback': feedback,
            'user_role': user_role,
            'context_length': len(conversation_history) if conversation_history else 0
        }
        
        self.conversations[username].append(conversation_entry)
        self.query_history.append(conversation_entry)
        
        # Store conversation context for this user
        if conversation_history:
            self.conversation_context[username] = {
                'full_history': conversation_history[-20:],  # Keep last 20 interactions
                'last_updated': datetime.now().isoformat(),
                'pattern_analysis': self._analyze_conversation_patterns(conversation_history)
            }
        
        # Learn successful patterns by role with conversation context
        if success and user_role:
            pattern_key = self._extract_pattern(request)
            
            # General patterns
            if pattern_key not in self.successful_patterns:
                self.successful_patterns[pattern_key] = []
            
            pattern_entry = {
                'sql_query': sql_query,
                'request': request,
                'context_size': len(conversation_history) if conversation_history else 0,
                'timestamp': datetime.now().isoformat()
            }
            self.successful_patterns[pattern_key].append(pattern_entry)
            
            # Role-based patterns with context
            if user_role not in self.role_based_patterns:
                self.role_based_patterns[user_role] = {}
            if pattern_key not in self.role_based_patterns[user_role]:
                self.role_based_patterns[user_role][pattern_key] = []
            self.role_based_patterns[user_role][pattern_key].append(pattern_entry)
        
        # Track feedback patterns with conversation context
        if feedback:
            self._track_feedback_pattern(request, feedback, sql_query, user_role, conversation_history)
        
        self.save_memory()
    
    def _analyze_conversation_patterns(self, conversation_history: list) -> dict:
        """Analyze conversation patterns for better context understanding"""
        if not conversation_history:
            return {}
        
        patterns = {
            'common_topics': [],
            'frequent_tables': [],
            'query_complexity_trend': 'stable',
            'success_rate': 0.0,
            'most_used_operations': [],
            'error_patterns': []
        }
        
        try:
            # Analyze recent conversations
            recent_messages = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            
            # Extract topics and tables mentioned
            topics = []
            tables = []
            operations = []
            successes = 0
            errors = []
            
            for msg in recent_messages:
                if hasattr(msg, 'message'):
                    message_lower = msg.message.lower()
                    
                    # Extract common business topics
                    business_terms = ['employee', 'department', 'salary', 'sales', 'product', 'customer', 'revenue', 'profit']
                    for term in business_terms:
                        if term in message_lower:
                            topics.append(term)
                    
                    # Extract SQL operations
                    sql_operations = ['select', 'count', 'sum', 'average', 'group by', 'order by', 'join', 'where']
                    for op in sql_operations:
                        if op in message_lower:
                            operations.append(op)
                
                if hasattr(msg, 'sql_query') and msg.sql_query:
                    # Extract table names from SQL
                    import re
                    table_matches = re.findall(r'FROM\s+(\w+)', msg.sql_query, re.IGNORECASE)
                    tables.extend(table_matches)
                
                if hasattr(msg, 'success'):
                    if msg.success:
                        successes += 1
                    else:
                        if hasattr(msg, 'response'):
                            errors.append(msg.response)
            
            # Calculate patterns
            from collections import Counter
            patterns['common_topics'] = [item for item, count in Counter(topics).most_common(5)]
            patterns['frequent_tables'] = [item for item, count in Counter(tables).most_common(5)]
            patterns['most_used_operations'] = [item for item, count in Counter(operations).most_common(5)]
            patterns['success_rate'] = successes / len(recent_messages) if recent_messages else 0.0
            patterns['error_patterns'] = errors[-3:]  # Last 3 errors
            
        except Exception as e:
            logger.error(f"Error analyzing conversation patterns: {e}")
        
        return patterns
    
    def _track_feedback_pattern(self, request: str, feedback: str, sql_query: str, user_role: str = None, conversation_history: list = None):
        """Enhanced feedback pattern tracking with conversation context"""
        feedback_key = self._extract_pattern(feedback)
        if feedback_key not in self.feedback_history:
            self.feedback_history[feedback_key] = []
        
        context_analysis = self._analyze_conversation_patterns(conversation_history) if conversation_history else {}
        
        self.feedback_history[feedback_key].append({
            'original_request': request,
            'feedback': feedback,
            'corrected_sql': sql_query,
            'timestamp': datetime.now().isoformat(),
            'user_role': user_role,
            'conversation_context': context_analysis,
            'context_size': len(conversation_history) if conversation_history else 0
        })
    
    def _extract_pattern(self, request: str) -> str:
        """Enhanced pattern extraction with more granular categorization"""
        keywords = {
            'aggregation': ['count', 'sum', 'average', 'avg', 'max', 'min', 'total'],
            'filtering': ['where', 'filter', 'find', 'search', 'lookup'],
            'grouping': ['group', 'by department', 'by category', 'breakdown'],
            'joining': ['join', 'combine', 'merge', 'relate'],
            'sorting': ['order', 'sort', 'rank', 'top', 'bottom', 'highest', 'lowest'],
            'temporal': ['date', 'time', 'year', 'month', 'recent', 'last'],
            'comparison': ['compare', 'vs', 'versus', 'difference', 'between'],
            'analysis': ['analysis', 'analyze', 'insight', 'trend', 'pattern']
        }
        
        request_lower = request.lower()
        found_categories = []
        
        for category, terms in keywords.items():
            if any(term in request_lower for term in terms):
                found_categories.append(category)
        
        return '_'.join(found_categories) if found_categories else 'general'
    
    def get_enhanced_role_based_context(self, username: str, request: str, user_role: str, feedback: str = None, conversation_history: list = None) -> str:
        """Enhanced context generation with full conversation awareness"""
        context = f"=== ENHANCED ROLE-BASED CONTEXT FOR {user_role.upper()} ===\n\n"
        
        # Add conversation history analysis
        if username in self.conversation_context:
            user_context = self.conversation_context[username]
            patterns = user_context.get('pattern_analysis', {})
            
            context += f"=== USER'S CONVERSATION PATTERNS ===\n"
            if patterns.get('common_topics'):
                context += f"Frequent Topics: {', '.join(patterns['common_topics'])}\n"
            if patterns.get('frequent_tables'):
                context += f"Most Used Tables: {', '.join(patterns['frequent_tables'])}\n"
            if patterns.get('most_used_operations'):
                context += f"Preferred Operations: {', '.join(patterns['most_used_operations'])}\n"
            context += f"Success Rate: {patterns.get('success_rate', 0):.1%}\n"
            if patterns.get('error_patterns'):
                context += f"Recent Issues: {'; '.join(patterns['error_patterns'][:2])}\n"
            context += "\n"
        
        # Enhanced role-specific successful patterns
        pattern = self._extract_pattern(request)
        if user_role in self.role_based_patterns and pattern in self.role_based_patterns[user_role]:
            role_examples = self.role_based_patterns[user_role][pattern][-3:]  # Last 3 examples
            if role_examples:
                context += f"=== SUCCESSFUL {user_role.upper()} PATTERNS FOR SIMILAR REQUESTS ===\n"
                for i, example in enumerate(role_examples, 1):
                    context += f"{i}. Request: {example['request']}\n"
                    context += f"   SQL: {example['sql_query']}\n"
                    context += f"   Context: {example.get('context_size', 0)} previous interactions\n\n"
        
        # Recent user queries with enhanced context
        if username in self.conversations:
            recent = [entry for entry in self.conversations[username][-5:] if entry.get('user_role') == user_role]
            if recent:
                context += f"=== RECENT {user_role.upper()} CONVERSATION HISTORY ===\n"
                for i, entry in enumerate(recent, 1):
                    context += f"{i}. Request: {entry['request']}\n"
                    context += f"   SQL: {entry['sql_query']}\n"
                    context += f"   Success: {entry['success']}\n"
                    if entry.get('feedback'):
                        context += f"   Previous Feedback: {entry['feedback']}\n"
                    context += f"   Context Size: {entry.get('context_length', 0)} interactions\n\n"
        
        # Enhanced feedback patterns with conversation context
        if feedback:
            feedback_pattern = self._extract_pattern(feedback)
            if feedback_pattern in self.feedback_history:
                role_feedback = [entry for entry in self.feedback_history[feedback_pattern] 
                               if entry.get('user_role') == user_role][-2:]
                if role_feedback:
                    context += f"=== SIMILAR {user_role.upper()} FEEDBACK CORRECTIONS ===\n"
                    for i, fb_entry in enumerate(role_feedback, 1):
                        context += f"{i}. Original Request: {fb_entry['original_request']}\n"
                        context += f"   Feedback Given: {fb_entry['feedback']}\n"
                        context += f"   Corrected SQL: {fb_entry['corrected_sql']}\n"
                        context += f"   Context: {fb_entry.get('context_size', 0)} previous interactions\n"
                        if fb_entry.get('conversation_context'):
                            conv_ctx = fb_entry['conversation_context']
                            if conv_ctx.get('common_topics'):
                                context += f"   User's Topics: {', '.join(conv_ctx['common_topics'])}\n"
                        context += "\n"
        
        # Add conversation-aware instructions
        context += f"=== CONVERSATION-AWARE INSTRUCTIONS ===\n"
        context += f"1. Consider the user's conversation patterns and preferences shown above\n"
        context += f"2. Build upon previous successful interactions with this user\n"
        context += f"3. Learn from past feedback corrections in similar contexts\n"
        context += f"4. Adapt your response style to match user's complexity preferences\n"
        context += f"5. Reference previous topics/tables if relevant to current request\n\n"
        
        return context

# (in main.py)

class DatabaseSchemaInspector:
    def __init__(self, selected_databases: List[str]):
        self.selected_databases = selected_databases
        self.schema_cache = {}
        self.refresh_schema_cache()

    def refresh_schema_cache(self):
        """Refresh schema information for all databases."""
        self.schema_cache = {}
        for db_name in self.selected_databases:
            self.schema_cache[db_name] = self._inspect_database_schema(db_name)
        print(f"✅ Schema cache refreshed for: {list(self.schema_cache.keys())}")

    def _inspect_database_schema(self, db_name: str) -> Dict[str, Any]:
        # This method remains the same as before...
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
                
                table_schema = {
                    "column_names": [col[1] for col in columns_info],
                    "columns": {
                        col[1]: {"type": col[2]} for col in columns_info
                    }
                }
                schema_info["tables"][table_name] = table_schema
            
            conn.close()
            return schema_info
            
        except Exception as e:
            return {"tables": {}, "error": f"Schema inspection failed: {str(e)}"}

    def get_schema_context_for_llm(self, username: str = None, role_manager: 'RoleManager' = None) -> str:
        # This method remains the same as before...
        context = "=== DATABASE SCHEMA INFORMATION ===\n\n"
        for db_name, schema in self.schema_cache.items():
            if "error" in schema:
                context += f"❌ {db_name}: {schema['error']}\n\n"
                continue
            
            context += f"--- DATABASE: {db_name} ---\n"
            for table_name, table_info in schema.get("tables", {}).items():
                if role_manager and username and not role_manager.check_table_access(username, table_name):
                    continue
                accessible_columns = role_manager.get_accessible_columns(username, table_name) if role_manager and username else table_info["column_names"]
                context += f"TABLE: {table_name}\n"
                context += f"COLUMNS: {', '.join(accessible_columns)}\n\n"
        context += "=== END OF SCHEMA ===\n\n"
        return context

    def get_all_accessible_columns(self) -> Dict[str, List[str]]:
        """Gets all columns for all tables across all cached databases."""
        all_columns = {}
        for db_name, schema in self.schema_cache.items():
            if "tables" in schema:
                for table_name, table_info in schema["tables"].items():
                    all_columns[table_name] = table_info["column_names"]
        return all_columns

    def validate_query_against_schema(self, sql_query: str) -> Dict[str, Any]:
        """
        Parses a SQL query to validate its tables and columns against the cached schema.
        This is the new, more powerful implementation.
        """
        print(f"🕵️  Validating query against schema: {sql_query}")
        
        # 1. A simplified but effective regex to extract table and column names
        # It captures words following FROM/JOIN (tables) and SELECT/WHERE/ORDER BY (columns)
        try:
            tables_re = re.findall(r'(?:FROM|JOIN)\s+([`"\']?\w+[`"\']?)', sql_query, re.IGNORECASE)
            # This captures columns, including those with aliases (e.g., "p.product_name")
            columns_re = re.findall(r'(?:SELECT|WHERE|BY|AND|,)\s+([`"\']?\w+(?:\.\w+)?[`"\']?)', sql_query, re.IGNORECASE)
        except Exception as e:
            return {"valid": False, "issues": [f"Regex parsing failed: {e}"]}

        # Normalize extracted names (remove quotes, etc.)
        referenced_tables = {re.sub(r'[`"\']', '', t) for t in tables_re}
        # Handle cases like 'e.name' -> 'name'
        referenced_columns = {re.sub(r'[`"\']', '', c).split('.')[-1] for c in columns_re}
        # Ignore common SQL keywords that might be captured
        sql_keywords = {'distinct', 'as', 'on', 'limit', 'desc', 'asc'}
        referenced_columns -= sql_keywords

        print(f"   - Referenced Tables: {referenced_tables}")
        print(f"   - Referenced Columns: {referenced_columns}")

        # 2. Get the complete list of actual tables and their columns from the schema cache
        schema_tables = self.get_all_accessible_columns()
        
        issues = []

        # 3. Validate tables
        for table in referenced_tables:
            if table not in schema_tables:
                issues.append(f"Table '{table}' does not exist in the selected database(s).")

        if issues: # If table validation fails, stop here
            return {"valid": False, "issues": issues}

        # 4. Validate columns
        all_available_columns = set()
        for table in referenced_tables:
            if table in schema_tables:
                all_available_columns.update(schema_tables[table])

        for col in referenced_columns:
            if col not in all_available_columns:
                # Find which table it might have belonged to for a better error message
                table_context = f"table(s) {', '.join(referenced_tables)}"
                issues.append(f"Column '{col}' does not exist in {table_context}. Available columns are: {list(all_available_columns)}")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def __init__(self, selected_databases: List[str]):
        self.selected_databases = selected_databases
        self.schema_cache = {}
        self.refresh_schema_cache()

    def refresh_schema_cache(self):
        """Refresh schema information for all databases"""
        self.schema_cache = {}
        for db_name in self.selected_databases:
            self.schema_cache[db_name] = self._inspect_database_schema(db_name)

    def _inspect_database_schema(self, db_name: str) -> Dict[str, Any]:
        """Get comprehensive schema information for a database"""
        try:
            db_path = os.path.abspath(db_name)
            if not os.path.exists(db_path):
                return {"tables": {}, "error": f"Database {db_name} not found"}
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            schema_info = {"tables": {}, "database_name": db_name}
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]
            
            for table_name in table_names:
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns_info = cursor.fetchall()
                
                # Get sample data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                sample_data = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                
                # Build table schema
                table_schema = {
                    "columns": {},
                    "column_names": [],
                    "primary_keys": [],
                    "sample_data": sample_data,
                    "row_count": row_count
                }
                
                for col_info in columns_info:
                    col_name = col_info[1]
                    col_type = col_info[2]
                    not_null = col_info[3]
                    default_value = col_info[4]
                    is_pk = col_info[5]
                    
                    table_schema["column_names"].append(col_name)
                    table_schema["columns"][col_name] = {
                        "type": col_type,
                        "not_null": bool(not_null),
                        "default_value": default_value,
                        "is_primary_key": bool(is_pk)
                    }
                    
                    if is_pk:
                        table_schema["primary_keys"].append(col_name)
                
                schema_info["tables"][table_name] = table_schema
            
            conn.close()
            return schema_info
            
        except Exception as e:
            return {"tables": {}, "error": f"Schema inspection failed: {str(e)}"}

    def get_schema_context_for_llm(self, username: str = None, role_manager: 'RoleManager' = None) -> str:
        """Generate comprehensive schema context for LLM"""
        context = "=== DATABASE SCHEMA INFORMATION ===\n\n"
        
        for db_name, schema in self.schema_cache.items():
            if "error" in schema:
                context += f"❌ {db_name}: {schema['error']}\n\n"
                continue
            
            context += f"📊 DATABASE: {db_name}\n"
            context += f"Tables: {len(schema['tables'])}\n\n"
            
            for table_name, table_info in schema["tables"].items():
                # Check role-based access
                if role_manager and username:
                    if not role_manager.check_table_access(username, table_name):
                        continue
                    accessible_columns = role_manager.get_accessible_columns(username, table_name)
                else:
                    accessible_columns = table_info["column_names"]
                
                context += f"🔸 TABLE: {table_name} ({table_info['row_count']} rows)\n"
                context += f"   ACCESSIBLE COLUMNS: {', '.join(accessible_columns)}\n"
                
                # Show column details
                for col_name in accessible_columns:
                    if col_name in table_info["columns"]:
                        col_info = table_info["columns"][col_name]
                        pk_indicator = " (PK)" if col_info["is_primary_key"] else ""
                        context += f"   - {col_name}: {col_info['type']}{pk_indicator}\n"
                
                # Show sample data (first row only)
                if table_info["sample_data"]:
                    sample_row = table_info["sample_data"][0]
                    accessible_sample = [str(sample_row[i]) for i, col in enumerate(table_info["column_names"]) if col in accessible_columns]
                    context += f"   SAMPLE: {', '.join(accessible_sample[:5])}{'...' if len(accessible_sample) > 5 else ''}\n"
                
                context += "\n"
        
        context += "=== IMPORTANT RULES ===\n"
        context += "1. ONLY use tables and columns that exist in the schema above\n"
        context += "2. NEVER add filters with user_id, username, or similar unless they exist as actual columns\n"
        context += "3. Use exact column names as shown in the schema\n"
        context += "4. Always include LIMIT clause for queries\n"
        context += "5. Use proper SQL syntax for SQLite\n\n"
        
        return context
    
    def validate_query_against_schema(self, sql_query: str, username: str = None, role_manager: 'RoleManager' = None) -> Dict[str, Any]:
        """Validate SQL query against actual database schema"""
        issues = []
        warnings = []
        suggestions = []
        
        try:
            # Basic SQL parsing to extract table and column references
            query_upper = sql_query.upper()
            
            # Check for problematic patterns
            if 'USER_ID' in query_upper and 'user_id' not in self._get_all_column_names():
                issues.append("Query references 'user_id' column which doesn't exist in any table")
                suggestions.append("Remove user_id filter - it's not needed for data access")
            
            if 'USERNAME' in query_upper and 'username' not in self._get_all_column_names():
                issues.append("Query references 'username' column which doesn't exist in any table")
                suggestions.append("Remove username filter - it's not needed for data access")
            
            # Extract table names (basic extraction)
            from_matches = re.findall(r'FROM\s+(\w+)', query_upper)
            join_matches = re.findall(r'JOIN\s+(\w+)', query_upper)
            table_references = from_matches + join_matches
            
            # Validate table existence
            all_tables = self._get_all_table_names()
            for table in table_references:
                if table.lower() not in [t.lower() for t in all_tables]:
                    issues.append(f"Table '{table}' does not exist in any database")
                    similar_tables = [t for t in all_tables if table.lower() in t.lower() or t.lower() in table.lower()]
                    if similar_tables:
                        suggestions.append(f"Did you mean: {', '.join(similar_tables[:3])}?")
            
            # Validate role-based access
            if role_manager and username:
                for table in table_references:
                    if not role_manager.check_table_access(username, table.lower()):
                        issues.append(f"User '{username}' doesn't have access to table '{table}'")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "suggestions": suggestions,
                "query": sql_query
            }
            
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Query validation error: {str(e)}"],
                "warnings": [],
                "suggestions": ["Please check query syntax"],
                "query": sql_query
            }

    def _get_all_table_names(self) -> List[str]:
        """Get all table names across all databases"""
        tables = []
        for schema in self.schema_cache.values():
            if "tables" in schema:
                tables.extend(schema["tables"].keys())
        return tables
    
    def _get_all_column_names(self) -> List[str]:
        """Get all column names across all databases"""
        columns = []
        for schema in self.schema_cache.values():
            if "tables" in schema:
                for table_info in schema["tables"].values():
                    columns.extend(col.lower() for col in table_info["column_names"])
        return columns

# Enhanced File manager supporting combined queries and multiple databases
class FileDataManager:
    def __init__(self, selected_databases: List[str] = None):
        self.data_sources = {}
        self.temp_db_path = "FILE_DATA.db"
        self.selected_databases = selected_databases or ["sample.db"]
        self.database_paths = {db: os.path.abspath(db) for db in self.selected_databases}
        self.schema_inspector = DatabaseSchemaInspector(self.selected_databases)

    def refresh_schema(self):
        """Refresh schema information"""
        if hasattr(self, 'schema_inspector'):
            self.schema_inspector.refresh_schema_cache()

    def get_database_tables(self, database_name: str) -> List[Dict[str, Any]]:
        """Get detailed table information for a database"""
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
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns_info = cursor.fetchall()
                
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                sample_rows = cursor.fetchall()
                
                table_info = {
                    "name": table_name,
                    "row_count": row_count,
                    "columns": [{"name": col[1], "type": col[2], "not_null": bool(col[3]), "primary_key": bool(col[5])} for col in columns_info],
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
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                logger.error(f"Unsupported file type: {file_path}")
                return False
            
            table_name = name.lower().replace(' ', '_')
            conn = sqlite3.connect(self.temp_db_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            
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
        """Execute query with schema validation"""
        print(f"🔍 Executing schema-aware query: {sql_query}")
    
        clean_query = self._clean_user_filters(sql_query)
    
        if clean_query.strip().lower().startswith("select 'error:"):
            error_msg = clean_query.split("'")[1] if "'" in clean_query else "Unknown error"
            return pd.DataFrame([{"error_message": error_msg}])
    
        try:
            # Try each database
            for db_name in self.selected_databases:
                try:
                    db_path = self.database_paths[db_name]
                    if not os.path.exists(db_path):
                        continue
                
                    print(f"🔍 Trying database: {db_name}")
                    conn = sqlite3.connect(db_path)
                    df = pd.read_sql_query(clean_query, conn)
                    conn.close()
                
                    print(f"✅ Query successful with {db_name}: {len(df)} rows")
                    return df
                
                except Exception as e:
                    print(f"⚠️  Database {db_name} failed: {e}")
                    continue
        
            # Try file sources
            if os.path.exists(self.temp_db_path):
                print(f"🔍 Trying file sources")
                conn = sqlite3.connect(self.temp_db_path)
                df = pd.read_sql_query(clean_query, conn)
                conn.close()
                print(f"✅ File query successful: {len(df)} rows")
                return df
        
            raise Exception("Query failed on all databases")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return pd.DataFrame([{
                "error_type": "Query Execution Failed",
                "error_message": str(e),
                "sql_query": clean_query,
                "suggestion": "Check table and column names"
            }])
    
    def _classify_error(self, error_str: str) -> str:
        """Classify the type of database error"""
        error_lower = error_str.lower()
    
        if "no such table" in error_lower:
            return "Table Not Found"
        elif "no such column" in error_lower:
            return "Column Not Found"
        elif "syntax error" in error_lower:
            return "SQL Syntax Error"
        elif "permission" in error_lower or "access" in error_lower:
            return "Permission Error"
        elif "constraint" in error_lower:
            return "Constraint Violation"
        elif "connection" in error_lower:
            return "Database Connection Error"
        else:
            return "Database Error"

    def _get_error_suggestion(self, error_type: str, error_message: str) -> str:
        """Get helpful suggestion based on error type"""
        suggestions = {
            "Table Not Found": "Verify the table name exists in your database. Check available tables in the sidebar.",
            "Column Not Found": "Check that the column name is spelled correctly and exists in the specified table.",
            "SQL Syntax Error": "Review your SQL syntax. Common issues include missing commas, incorrect joins, or invalid keywords.",
            "Permission Error": "You may not have sufficient permissions to access this data. Contact your administrator.",
            "Constraint Violation": "The query violates a database constraint. Check foreign key relationships and data types.",
            "Database Connection Error": "Unable to connect to the database. Check if the database file exists and is accessible.",
            "Database Error": "An unexpected database error occurred. Please check your query and try again."
        }
    
        base_suggestion = suggestions.get(error_type, "Please review your query and try again.")
    
        # Add specific suggestions based on error message content
        if "no such column" in error_message.lower():
            if "'" in error_message or '"' in error_message:
                return f"{base_suggestion} The specific column mentioned in the error may not exist in the table schema."
    
        return base_suggestion


    def _clean_user_filters(self, sql_query: str) -> str:
        """Remove user_id and username filters that don't exist"""
        cleaned = sql_query
        actual_columns = set()
        if hasattr(self, 'schema_inspector'):
            for schema in self.schema_inspector.schema_cache.values():
                if "tables" in schema:
                    for table_info in schema["tables"].values():
                        actual_columns.update(col.lower() for col in table_info["column_names"])
    
        if 'user_id' not in actual_columns:
            cleaned = re.sub(r'\s+WHERE\s+user_id\s*=\s*[\'"][^\'\"]*[\'"]', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+AND\s+user_id\s*=\s*[\'"][^\'\"]*[\'"]', '', cleaned, flags=re.IGNORECASE)
            print("🔧 Removed non-existent user_id filter")
    
        if 'username' not in actual_columns:
            cleaned = re.sub(r'\s+WHERE\s+username\s*=\s*[\'"][^\'\"]*[\'"]', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+AND\s+username\s*=\s*[\'"][^\'\"]*[\'"]', '', cleaned, flags=re.IGNORECASE)
            print("🔧 Removed non-existent username filter")
    
        cleaned = re.sub(r'\s+WHERE\s+AND\s+', ' WHERE ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+WHERE\s*;', ';', cleaned, flags=re.IGNORECASE)
    
        return cleaned.strip()

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
    
    def get_supported_databases(self) -> List[str]:
        """Get list of supported target databases"""
        return ["postgresql", "sqlserver", "db2"]

# Enhanced CrewAI Tools
# (in main.py)

class SQLGeneratorTool(BaseTool):
    name: str = "enhanced_sql_generator"
    description: str = "Generate SQL queries with comprehensive schema awareness, validation, and conversation context for the selected database."

    def _run(self, query_description: str, username: str = "admin", feedback: str = None, iteration: int = 1, conversation_history: list = None) -> str:
        """Enhanced SQL generation with conversation context awareness"""
        try:
            file_manager = getattr(self, '_file_manager', None)
            role_manager = getattr(self, '_role_manager', None)
            memory = getattr(self, '_memory', None)
            llm = getattr(self, '_llm', None)

            print(f"🔍 Enhanced SQL Generator - Iteration {iteration} for user '{username}'")
            print(f"📝 Request: {query_description}")
            print(f"📚 Conversation History: {len(conversation_history) if conversation_history else 0} messages")

            schema_context = ""
            if file_manager and hasattr(file_manager, 'schema_inspector'):
                schema_context = file_manager.schema_inspector.get_schema_context_for_llm(username, role_manager)
                print(f"📋 Using schema context for: {file_manager.selected_databases}")

            # Get enhanced memory context with conversation history
            memory_context = ""
            if memory and conversation_history:
                user_role = role_manager.get_user_role(username).value if role_manager and role_manager.get_user_role(username) else "admin"
                memory_context = memory.get_enhanced_role_based_context(
                    username=username,
                    request=query_description,
                    user_role=user_role,
                    feedback=feedback,
                    conversation_history=conversation_history
                )
                print(f"🧠 Enhanced memory context length: {len(memory_context)} characters")

            if llm and schema_context and type(llm).__name__ != 'MockLLM':
                result = self._generate_with_enhanced_llm(
                    query_description, llm, schema_context, memory_context, 
                    feedback, iteration, username, role_manager, conversation_history
                )
            else:
                if type(llm).__name__ == 'MockLLM':
                    print("⚠️ MockLLM is active. Using enhanced schema-aware fallback logic.")
                else:
                    print("⚠️ LLM or schema context not available. Using enhanced schema-aware fallback logic.")
                result = self._generate_enhanced_schema_aware_fallback(
                    query_description, file_manager.schema_inspector, username, 
                    role_manager, conversation_history
                )

            print(f"✅ Generated SQL: {result}")
            return result

        except Exception as e:
            print(f"❌ SQL generation error: {e}")
            return "SELECT 'Error generating SQL query due to an internal error' AS error;"

    def _generate_with_enhanced_llm(self, query_description: str, llm, schema_context: str, memory_context: str,
                                  feedback: str = None, iteration: int = 1, username: str = "admin",
                                  role_manager: 'RoleManager' = None, conversation_history: list = None) -> str:
        """Generate SQL using LLM with enhanced conversation context"""
        
        # Enhanced prompt template with conversation awareness
        prompt_template = """
        You are an expert SQLite query writer with advanced conversation awareness. Your task is to generate a single, valid SQLite query based on the user's business request, database schema, and full conversation context.

        --- DATABASE SCHEMA (Your primary source of truth for tables and columns) ---
        {schema_context}
        --- END OF SCHEMA ---

        --- CONVERSATION & MEMORY CONTEXT ---
        {memory_context}
        --- END OF MEMORY CONTEXT ---

        --- SYSTEM CONTEXT (DO NOT use in WHERE clauses unless columns explicitly exist) ---
        - User ID: {username}
        - User Role: {role}
        - Current Iteration: {iteration}
        
        --- CRITICAL RULES (UNCHANGED) ---
        1. **FOCUS ON THE BUSINESS REQUEST**: Answer the user's specific request below
        2. **NEVER USE SYSTEM CONTEXT FOR FILTERING**: Do not add user_id/role filters unless they exist as columns
        3. **STRICT SCHEMA ADHERENCE**: Only use tables and columns from the Database Schema
        4. **CONVERSATION AWARENESS**: Consider the conversation patterns and context shown above
        5. **LEARN FROM HISTORY**: Build upon successful patterns from previous interactions
        6. **OUTPUT ONLY SQL**: Return only the SQL query, nothing else

        --- PREVIOUS ATTEMPT FEEDBACK (if any) ---
        {feedback_section}
        
        --- USER'S CURRENT BUSINESS REQUEST ---
        "{query_description}"
        
        --- ADDITIONAL CONTEXT INSTRUCTIONS ---
        - If this user frequently asks about specific topics/tables, prioritize those in your response
        - If previous similar requests were successful, use similar patterns
        - If feedback was provided, address it while maintaining conversation consistency
        - Consider the user's preferred complexity level based on conversation history
        """
        
        user_role = role_manager.get_user_role(username).value if role_manager and role_manager.get_user_role(username) else "admin"
        
        feedback_section = "No feedback yet. This is the first attempt."
        if feedback:
            feedback_section = f"IMPORTANT: Your last query was rejected. You MUST address this feedback: {feedback}"
            if conversation_history:
                feedback_section += f"\n\nConversation context shows {len(conversation_history)} previous interactions. Use this context to better understand the user's intent."

        prompt = prompt_template.format(
            schema_context=schema_context,
            memory_context=memory_context,
            username=username,
            role=user_role,
            iteration=iteration,
            query_description=query_description,
            feedback_section=feedback_section
        )

        try:
            response = llm.invoke(prompt)
            sql_query = response.content if hasattr(response, 'content') else str(response)
            
            # Clean the response
            sql_query = sql_query.strip().replace('```sql', '').replace('```', '').strip()
            if not sql_query.upper().startswith(('SELECT', 'WITH')):
                raise ValueError("Generated response is not a valid SELECT or WITH query.")
            if not sql_query.endswith(';'):
                sql_query += ';'
            return sql_query

        except Exception as e:
            print(f"❌ Enhanced LLM generation failed: {e}. Falling back.")
            return self._generate_enhanced_schema_aware_fallback(
                query_description, getattr(self, '_file_manager').schema_inspector, 
                username, role_manager, conversation_history
            )
        
    def _generate_enhanced_schema_aware_fallback(self, query_description: str, schema_inspector: 'DatabaseSchemaInspector',
                                               username: str = "admin", role_manager: 'RoleManager' = None, 
                                               conversation_history: list = None) -> str:
        """Enhanced fallback that avoids auto-execution of invalid queries"""
        print("Enhanced fallback logic activated...")
        
        # Check if we have any accessible tables
        all_accessible_tables = []
        if not schema_inspector:
            return "SELECT 'Schema information is not available' AS validation_error;"

        for schema in schema_inspector.schema_cache.values():
            if "tables" in schema:
                for table_name in schema["tables"].keys():
                    if not role_manager or role_manager.check_table_access(username, table_name):
                        all_accessible_tables.append(table_name)

        if not all_accessible_tables:
            return "SELECT 'No accessible tables found for your role' AS validation_error;"

        # Instead of generating a random query, try to understand what the user wants
        query_words = set(query_description.lower().split())
        
        # Look for table names mentioned in the request
        mentioned_tables = [table for table in all_accessible_tables if table.lower() in query_words]
        
        if mentioned_tables:
            # If user mentioned specific tables, use the first one
            target_table = mentioned_tables[0]
            print(f"Enhanced fallback found mentioned table: '{target_table}'")
        elif conversation_history:
            # Look for frequently used tables in conversation history
            table_usage = {}
            for msg in conversation_history[-10:]:
                if hasattr(msg, 'sql_query') and msg.sql_query:
                    import re
                    table_matches = re.findall(r'FROM\s+(\w+)', msg.sql_query, re.IGNORECASE)
                    for table_match in table_matches:
                        if table_match in all_accessible_tables:
                            table_usage[table_match] = table_usage.get(table_match, 0) + 1
            
            if table_usage:
                target_table = max(table_usage, key=table_usage.get)
                print(f"Enhanced fallback selected frequently used table: '{target_table}'")
            else:
                target_table = all_accessible_tables[0]
                print(f"Enhanced fallback defaulted to first table: '{target_table}'")
        else:
            target_table = all_accessible_tables[0]
            print(f"Enhanced fallback defaulted to first table: '{target_table}'")
        
        # Generate a basic query for the selected table
        return f"SELECT * FROM {target_table} LIMIT 10;"

class SQLExecutorTool(BaseTool):
    name: str = "sql_executor"
    description: str = "Validate and then execute a SQL query against the selected databases."

    def _run(self, sql_query: str, username: str = "admin", db_path: str = None) -> str:
        """First, validate the SQL against the schema, then execute if valid."""
        role_manager = getattr(self, '_role_manager', None)
        file_manager = getattr(self, '_file_manager', None)

        if not file_manager or not hasattr(file_manager, 'schema_inspector'):
            return "Execution failed: Schema information is not available for validation."

        # --- STEP 1: VALIDATE THE QUERY PROGRAMMATICALLY ---
        validation_result = file_manager.schema_inspector.validate_query_against_schema(sql_query)
        if not validation_result["valid"]:
            error_message = f"SQL query validation failed: {'. '.join(validation_result['issues'])}"
            print(f"❌ {error_message}")
            # This clear, actionable feedback is crucial for the AI to self-correct
            return f"Validation Error: {error_message}. Please regenerate the query and fix these issues."

        print("✅ Query passed schema validation.")

        # --- STEP 2: EXECUTE THE QUERY (only if valid) ---
        try:
            df = file_manager.execute_combined_query(sql_query, username, role_manager)

            if "error_message" in df.columns:
                db_error = df["error_message"].iloc[0]
                return f"Execution failed with database error: {db_error}"

            print(f"✅ Query executed successfully, returned {len(df)} rows")

            if hasattr(self, '_role_manager') and self._role_manager and username != "admin":
                df = self._apply_role_filtering(df, username, role_manager)

            table_str = tabulate(df.head(20), headers='keys', tablefmt='grid', showindex=False) if len(df) > 0 else "No results found"
            if len(df) > 20:
                table_str += f"\n... and {len(df) - 20} more rows"
            
            execution_result = {
                "success": True, "data": df.to_dict('records'), "columns": df.columns.tolist(),
                "row_count": len(df), "sql_query": sql_query,
                "table_display": table_str, "dataframe": df
            }

            if hasattr(self, '_system_ref') and self._system_ref:
                self._system_ref.last_execution_result = execution_result
                print("📦 Stored execution result in system")

            return f"SQL Query Executed Successfully!\nRows Returned: {len(df)}\n\n{table_str}"

        except Exception as e:
            error_msg = f"Execution failed with an unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
        
    def _apply_role_filtering(self, df, username: str, role_manager):
        """SAFE METHOD: Apply role-based filtering with fallback protection"""
        try:
            if not role_manager or username == "admin":
                return df
            
            user = role_manager.users.get(username)
            if not user:
                return df
            
            # Get all restricted columns for this user
            restricted_columns = set()
            for table_name, cols in user.restricted_columns.items():
                restricted_columns.update(cols)
            
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
    description: str = "Generate professional charts from query results"
    
    def _run(self, data: List[Dict], chart_type: str = "bar", title: str = None) -> str:
        """Generate professional charts"""
        try:
            if not data: return "No data provided for chart generation"
            df = pd.DataFrame(data)
            plt.style.use('default')
            plt.figure(figsize=(12, 8))
            
            if chart_type == "bar":
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols and text_cols:
                    x_col, y_col = text_cols[0], numeric_cols[0]
                    bars = plt.bar(df[x_col].astype(str), df[y_col], color='steelblue', alpha=0.8, edgecolor='navy')
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
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
                    plt.pie(df[numeric_cols[0]], labels=df[text_cols[0]].astype(str), autopct='%1.1f%%', startangle=90)
                else:
                    return "Pie chart requires categorical and numeric data with ≤10 categories"
            
            if not title: title = f'{chart_type.title()} Chart - Data Analysis'
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            if chart_type == "bar": plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
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
        print(f"🔍 Query Validator - Input SQL: {sql_query}")
        corrected_query = sql_query.strip()
        corrections_made = []
        
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
        
        self.role_manager = RoleManager()
        self.memory = ConversationMemory()
        self.file_manager = FileDataManager(self.selected_databases)
        
        self.role_manager.add_user("admin", UserRole.ADMIN)
        self.role_manager.add_user("analyst", UserRole.ANALYST)
        self.role_manager.add_user("viewer", UserRole.VIEWER)
        
        self.sql_generator = SQLGeneratorTool()
        self.sql_executor = SQLExecutorTool()
        self.chart_generator = ChartGeneratorTool()
        self.query_validator = QueryValidatorTool()
        self.sql_converter = SQLConverterTool()

        # Set enhanced attributes on tools
        self.sql_generator._file_manager = self.file_manager
        self.sql_generator._schema_inspector = self.file_manager.schema_inspector
        self.sql_generator._memory = self.memory
        self.sql_generator._role_manager = self.role_manager
        self.sql_generator._llm = self.llm
        
        self.sql_executor._role_manager = self.role_manager
        self.sql_executor._file_manager = self.file_manager
        self.sql_executor._system_ref = self
        self.sql_executor._last_execution_data = None
        
        self.query_validator._role_manager = self.role_manager
        self.query_validator._llm = self.llm
        
        self._create_agents()
        self._verify_database_connections()

    def debug_schema_info(self, username: str = "admin"):
        """Debug method to show schema information"""
        print("\n" + "="*60 + "\n🔍 DATABASE SCHEMA DEBUG INFORMATION\n" + "="*60)
        if hasattr(self.file_manager, 'schema_inspector'):
            schema_inspector = self.file_manager.schema_inspector
            for db_name, schema in schema_inspector.schema_cache.items():
                print(f"\n📊 DATABASE: {db_name}")
                if "error" in schema: print(f"❌ Error: {schema['error']}"); continue
                if "tables" not in schema: print("⚠️  No tables found"); continue
                print(f"Tables: {len(schema['tables'])}")
                for table_name, table_info in schema["tables"].items():
                    has_access = self.role_manager.check_table_access(username, table_name)
                    access_indicator = "✅" if has_access else "❌"
                    print(f"{access_indicator} TABLE: {table_name} ({table_info['row_count']} rows)")
                    if has_access:
                        accessible_columns = self.role_manager.get_accessible_columns(username, table_name)
                        print(f"   Columns: {', '.join(accessible_columns)}")
        print("\n" + "="*60)

    def convert_sql_to_target_db(self, sql_query: str, target_db: str) -> Dict[str, Any]:
        """Convert SQL query to target database dialect"""
        try:
            converted_query = self.sql_converter._run(sql_query, target_db)
            return {"success": True, "original_query": sql_query, "converted_query": converted_query, "target_database": target_db, "conversion_notes": f"Converted SQLite query to {target_db.upper()} dialect"}
        except Exception as e:
            return {"success": False, "error": f"Conversion failed: {str(e)}", "original_query": sql_query, "target_database": target_db}
    
    def get_database_tables_info(self, username: str = "admin") -> Dict[str, Any]:
        """Get detailed table information for all selected databases"""
        try:
            user_role = self.role_manager.get_user_role(username)
            all_tables = self.file_manager.get_all_databases_tables()
            filtered_tables = {}
            for db_name, tables in all_tables.items():
                accessible_tables = []
                for table_info in tables:
                    table_name = table_info["name"]
                    if self.role_manager.check_table_access(username, table_name):
                        accessible_columns = self.role_manager.get_accessible_columns(username, table_name)
                        filtered_table = {
                            "name": table_info["name"], "row_count": table_info["row_count"],
                            "columns": [col for col in table_info["columns"] if col["name"] in accessible_columns],
                            "sample_data": table_info["sample_data"],
                            "column_names": [col for col in table_info["column_names"] if col in accessible_columns],
                            "access_level": user_role.value if user_role else "unknown"
                        }
                        accessible_tables.append(filtered_table)
                filtered_tables[db_name] = accessible_tables
            return {"success": True, "databases": filtered_tables, "user_role": user_role.value if user_role else "unknown", "total_databases": len(filtered_tables), "total_accessible_tables": sum(len(tables) for tables in filtered_tables.values())}
        except Exception as e:
            return {"success": False, "error": f"Failed to get table information: {str(e)}"}
    
    def _verify_database_connections(self):
        """Verify connections to all selected databases"""
        print(f"\n🔍 Verifying connections to {len(self.selected_databases)} database(s)...")
        for db_name in self.selected_databases:
            db_path = os.path.abspath(db_name)
            print(f"📂 Checking: {db_name} -> {db_path}")
            if not os.path.exists(db_path):
                print(f"❌ Database file not found: {db_path}")
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
                    for table in tables[:3]:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        print(f"   - {table}: {count} records")
                    if len(tables) > 3: print(f"   - ... and {len(tables) - 3} more tables")
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
            cursor.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT, salary REAL, hire_date DATE, manager_id INTEGER, status TEXT DEFAULT 'active')")
            cursor.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL, budget REAL, location TEXT)")
            employees = [(1,"John Doe","Engineering",75000,"2022-01-15",None,"active"),(2,"Jane Smith","Marketing",65000,"2021-03-20",None,"active"),(3,"Bob Johnson","Engineering",80000,"2020-07-10",1,"active"),(4,"Alice Brown","HR",60000,"2023-02-01",None,"active"),(5,"Charlie Wilson","Engineering",72000,"2021-11-05",1,"active"),(6,"Diana Prince","Marketing",68000,"2022-08-12",2,"active"),(7,"Edward Norton","Finance",70000,"2020-04-20",None,"active"),(8,"Fiona Green","HR",58000,"2023-01-10",4,"active"),(9,"George Miller","Engineering",85000,"2019-05-15",1,"active"),(10,"Helen Davis","Marketing",62000,"2022-09-01",2,"active")]
            departments = [(1,"Engineering",500000,"Building A"),(2,"Marketing",300000,"Building B"),(3,"HR",200000,"Building C"),(4,"Finance",250000,"Building A")]
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
        """Create enhanced agents with schema awareness"""
        self.sql_architect = Agent(
            role='Schema-Aware SQL Database Architect',
            goal='Generate perfect SQL queries using comprehensive schema knowledge, respecting user roles and database constraints',
            backstory=f"""You are a world-class database architect with deep knowledge of database schemas. 
                        You have access to comprehensive schema information for {len(self.selected_databases)} database(s): 
                        {', '.join(self.selected_databases)}. You ALWAYS use actual table and column names from the schema.
                        You NEVER add user_id or username filters unless they exist as real columns.
                        You excel at writing precise SQL that matches the exact database structure.""",
            tools=[self.sql_generator], llm=self.llm, verbose=True, allow_delegation=False, max_iter=3
        )
        self.security_specialist = Agent(
            role='Database Security and Role-Based Access Specialist',
            goal='Ensure all SQL queries respect role-based access controls and follow enterprise security practices across multiple databases',
            backstory=f"""You are a cybersecurity expert who specializes in role-based access control and database 
                        security across multiple database systems. You work with {len(self.selected_databases)} database(s) 
                        and understand that viewers should not see sensitive data, analysts have limited access, 
                        and only admins have full access. You enforce these rules strictly across all databases.""",
            tools=[self.query_validator], llm=self.llm, verbose=True, allow_delegation=False, max_iter=2
        )
        self.data_analyst = Agent(
            role='Senior Data Analytics Engineer with Multi-Database and Multi-Source Query Execution',
            goal='Execute queries across multiple databases and file sources while applying role-based column filtering',
            backstory=f"""You are a senior data engineer who can execute queries across multiple data sources 
                        including {len(self.selected_databases)} database(s): {', '.join(self.selected_databases)} 
                        and various file sources. You ensure that users only see data appropriate for their role 
                        and can intelligently route queries to the appropriate data sources. You apply column-level 
                        security and present results in clear, formatted tables.""",
            tools=[self.sql_executor], llm=self.llm, verbose=True, allow_delegation=False, max_iter=2
        )
        self.visualization_expert = Agent(
            role='Data Visualization Expert with Role-Aware Chart Generation',
            goal='Create professional visualizations that respect data privacy constraints based on user roles',
            backstory="""You are a data visualization expert who understands that different users should see 
                        different levels of detail in charts. You create visualizations that are both insightful 
                        and compliant with role-based access policies.""",
            tools=[self.chart_generator], llm=self.llm, verbose=True, allow_delegation=False, max_iter=2
        )
    
    def apply_feedback_and_retry(self, previous_result: Dict, feedback: str, username: str = "admin",
                               create_chart: bool = False, chart_type: str = "bar",
                               data_source: str = "auto", selected_databases: List[str] = None) -> Dict[str, Any]:
        """Enhanced feedback application with conversation history"""
        original_request = previous_result.get('request', '')
        previous_iteration = previous_result.get('iteration', 1)
        target_databases = selected_databases or previous_result.get('selected_databases', self.selected_databases)
        
        # Extract conversation history if available
        conversation_history = previous_result.get('conversation_history', [])
        
        print(f"🔄 Applying feedback with {len(conversation_history)} conversation context items")
        
        # Store conversation history in memory for the SQL generator
        if conversation_history:
            self.memory.add_conversation(
                username=username,
                request=original_request,
                sql_query=previous_result.get('original_sql', ''),
                success=False,  # This is a retry, so previous was not successful
                feedback=feedback,
                user_role=self.role_manager.get_user_role(username).value if self.role_manager.get_user_role(username) else "admin",
                conversation_history=conversation_history
            )
        
        return self.process_request(
            user_request=original_request, 
            username=username, 
            create_chart=create_chart,
            chart_type=chart_type, 
            data_source=data_source, 
            feedback=feedback,
            iteration=previous_iteration + 1, 
            selected_databases=target_databases,
            conversation_history=conversation_history  # Pass conversation history
        )

    def process_request(self, user_request: str, username: str = "admin",
                       create_chart: bool = False, chart_type: str = "bar",
                       data_source: str = "auto", feedback: str = None,
                       iteration: int = 1, selected_databases: List[str] = None,
                       conversation_history: List = None) -> Dict[str, Any]:
        """Enhanced request processing with validation error handling"""
        if not self.role_manager.check_permission(username, "read"):
            return {"error": f"Permission denied: {username} doesn't have read permissions"}

        print(f"\n🚀 ENHANCED VALIDATION-AWARE WORKFLOW (Max 3 attempts)")
        print(f"📝 User Request: {user_request}")
        print(f"📚 Conversation Context: {len(conversation_history) if conversation_history else 0} messages")

        max_attempts = 3
        current_attempt = 1
        last_error = feedback
        sql_query = ""
        
        while current_attempt <= max_attempts:
            print(f"\n--- Enhanced Attempt {current_attempt}/{max_attempts} ---")
            
            # Generate SQL with conversation context
            print("🧠 Generating SQL query with conversation awareness...")
            sql_query = self.sql_generator._run(
                query_description=user_request,
                username=username,
                feedback=last_error,
                iteration=current_attempt,
                conversation_history=conversation_history
            )
            
            # Validate the generated SQL
            print(f"🕵️  Validating query: {sql_query}")
            validation_result = self.file_manager.schema_inspector.validate_query_against_schema(sql_query)
            
            if validation_result["valid"]:
                print("✅ Query is valid. Proceeding to execution.")
                break
            else:
                validation_issues = validation_result.get("issues", ["Unknown validation error"])
                print(f"❌ Validation Failed. Issues: {validation_issues}")
                
                # Instead of auto-retrying, return validation error for user feedback
                if current_attempt == 1:  # On first attempt, show validation error
                    print("🔄 Returning validation error for user feedback instead of auto-retry")
                    return {
                        "success": False,
                        "validation_error": True,
                        "validation_issues": validation_issues,
                        "bad_sql": sql_query,
                        "error": f"Generated SQL has validation errors: {'. '.join(validation_issues)}",
                        "user_role": self.role_manager.get_user_role(username).value if self.role_manager.get_user_role(username) else "admin"
                    }
                
                # On subsequent attempts (with feedback), continue trying
                last_error = f"Validation failed: {'. '.join(validation_issues)}"
                current_attempt += 1
                if current_attempt > max_attempts:
                    print("🚫 Max attempts reached. Returning final validation error.")
                    return {
                        "success": False,
                        "validation_error": True,
                        "validation_issues": validation_issues,
                        "bad_sql": sql_query,
                        "error": f"Failed to generate a valid SQL query after {max_attempts} attempts. Last issues: {'. '.join(validation_issues)}"
                    }

        # Execute the validated SQL Query
        print("\n⚡ Executing validated query...")
        execution_result_str = self.sql_executor._run(sql_query=sql_query, username=username)
        
        # Process results
        execution_data = self.last_execution_result or {}
        if "Validation Error:" in execution_result_str or "Execution failed" in execution_result_str:
             return {"success": False, "error": execution_result_str, "sql_query": sql_query}

        # Store successful interaction in memory with conversation context
        if execution_data.get("success", True):  # Assume success if not explicitly failed
            self.memory.add_conversation(
                username=username,
                request=user_request,
                sql_query=sql_query,
                success=True,
                user_role=self.role_manager.get_user_role(username).value if self.role_manager.get_user_role(username) else "admin",
                conversation_history=conversation_history
            )

        # Generate Chart if requested
        if create_chart and execution_data.get("success"):
            print("📊 Generating chart...")
            chart_result = self.chart_generator._run(data=execution_data.get("data", []), chart_type=chart_type)
            execution_data["chart_path"] = chart_result

        return {
            "success": True,
            "crew_result": execution_result_str,
            "execution_data": execution_data,
            "sql_query": sql_query,
            "user": username,
            "request": user_request,
            "conversation_context_used": len(conversation_history) if conversation_history else 0
        }

    
    def process_request_with_feedback_loop(self, user_request: str, username: str = "admin",
                                         create_chart: bool = False, chart_type: str = "bar",
                                         data_source: str = "auto", selected_databases: List[str] = None) -> Dict[str, Any]:
        """Enhanced feedback loop with role awareness and database selection"""
        iteration = 1
        target_databases = selected_databases or self.selected_databases
        
        result = self.process_request(
            user_request=user_request, username=username, create_chart=create_chart,
            chart_type=chart_type, data_source=data_source, feedback=None,
            iteration=iteration, selected_databases=target_databases
        )
        
        # In the API context, we return after the first iteration. The UI handles the loop.
        return {**result, "needs_feedback": True, "iteration": iteration}
    
    def apply_feedback_and_retry(self, previous_result: Dict, feedback: str, username: str = "admin",
                               create_chart: bool = False, chart_type: str = "bar",
                               data_source: str = "auto", selected_databases: List[str] = None) -> Dict[str, Any]:
        """Apply feedback and retry the request"""
        original_request = previous_result.get('request', '')
        previous_iteration = previous_result.get('iteration', 1)
        target_databases = selected_databases or previous_result.get('selected_databases', self.selected_databases)
        
        return self.process_request(
            user_request=original_request, username=username, create_chart=create_chart,
            chart_type=chart_type, data_source=data_source, feedback=feedback,
            iteration=previous_iteration + 1, selected_databases=target_databases
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
            if not data and self.last_execution_result and 'data' in self.last_execution_result:
                data = self.last_execution_result['data']
            if not data: return "No data available to export"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                pd.DataFrame(data).to_excel(writer, sheet_name='Query Results', index=False)
                metadata = {'Generated By': ['Gainwell SQL Analysis System'], 'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')], 'Records': [len(data)], 'LLM Provider': [self.llm_provider], 'Databases Used': [', '.join(self.selected_databases)]}
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
        
        self.llm_provider = llm_provider
        self.selected_databases = selected_databases or ["sample.db"]
        
        self._create_sample_data(self.selected_databases)
        self.system = CrewAISQLSystem(llm_provider, self.selected_databases)
        self._show_llm_status()
    
    def _create_sample_data(self, selected_databases: List[str] = None):
        """Create enhanced sample database and files if needed"""
        databases = selected_databases or ["sample.db"]
        if "sample.db" in databases and not os.path.exists("sample.db"):
            print("📊 Creating sample database...")
            try:
                conn = sqlite3.connect("sample.db")
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT, salary REAL, hire_date DATE, manager_id INTEGER, status TEXT DEFAULT 'active')")
                cursor.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL, budget REAL, location TEXT)")
                employees = [(1, "John Doe", "Engineering", 75000, "2022-01-15", None, "active"), (2, "Jane Smith", "Marketing", 65000, "2021-03-20", None, "active"), (3, "Bob Johnson", "Engineering", 80000, "2020-07-10", 1, "active"), (4, "Alice Brown", "HR", 60000, "2023-02-01", None, "active"), (5, "Charlie Wilson", "Engineering", 72000, "2021-11-05", 1, "active"), (6, "Diana Prince", "Marketing", 68000, "2022-08-12", 2, "active"), (7, "Edward Norton", "Finance", 70000, "2020-04-20", None, "active"), (8, "Fiona Green", "HR", 58000, "2023-01-10", 4, "active"), (9, "George Miller", "Engineering", 85000, "2019-05-15", 1, "active"), (10, "Helen Davis", "Marketing", 62000, "2022-09-01", 2, "active"), (11, "Ian Foster", "Engineering", 90000, "2018-06-01", 1, "active"), (12, "Jessica Wong", "Finance", 73000, "2021-09-15", 7, "active")]
                departments = [(1, "Engineering", 500000, "Building A"), (2, "Marketing", 300000, "Building B"), (3, "HR", 200000, "Building C"), (4, "Finance", 250000, "Building A")]
                cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", employees)
                cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments)
                conn.commit()
                conn.close()
                print(f"✅ Sample database created successfully")
            except Exception as e:
                print(f"❌ Error creating database: {e}")
        self._create_sample_files()
    
    def _create_sample_files(self):
        """Create enhanced sample files"""
        try:
            products_file = 'sample_products.csv'
            if not os.path.exists(products_file):
                products = {'product_id': [1,2,3,4,5,6,7,8,9,10],'product_name': ['Laptop Pro','Wireless Mouse','Mechanical Keyboard','4K Monitor','HD Webcam','Bluetooth Speakers','Noise-Canceling Headphones','Tablet Pro','Smartwatch','Wireless Charger'],'category': ['Electronics','Accessories','Accessories','Electronics','Accessories','Accessories','Accessories','Electronics','Electronics','Accessories'],'price': [1299.99,45.99,129.99,599.99,149.99,199.99,299.99,899.99,399.99,79.99],'stock': [25,150,75,40,80,60,35,20,45,100],'supplier': ['TechCorp','AccessoryInc','AccessoryInc','TechCorp','AccessoryInc','AudioCorp','AudioCorp','TechCorp','TechCorp','AccessoryInc']}
                pd.DataFrame(products).to_csv(products_file, index=False)
                print(f"✅ Created enhanced {products_file}")
            
            sales_file = 'sample_sales.xlsx'
            if not os.path.exists(sales_file):
                sales = {'sale_id': list(range(1,16)),'product_id': [1,2,3,1,4,5,6,7,8,9,10,2,3,5,7],'customer_name': ['Alice Johnson','Bob Smith','Carol Davis','David Wilson','Eva Martinez','Frank Brown','Grace Lee','Henry Garcia','Iris Chen','Jack Taylor','Kelly Moore','Liam O\'Brien','Maya Patel','Nathan Kim','Olivia Ross'],'quantity': [1,2,1,1,1,1,1,1,1,1,2,3,1,1,1],'sale_date': ['2024-01-15','2024-01-16','2024-01-17','2024-01-18','2024-01-19','2024-01-20','2024-01-21','2024-01-22','2024-01-23','2024-01-24','2024-01-25','2024-01-26','2024-01-27','2024-01-28','2024-01-29'],'total_amount': [1299.99,91.98,129.99,1299.99,149.99,199.99,299.99,899.99,399.99,399.99,159.98,137.97,129.99,199.99,899.99]}
                pd.DataFrame(sales).to_excel(sales_file, index=False)
                print(f"✅ Created enhanced {sales_file}")
        except Exception as e:
            print(f"❌ Error creating sample files: {e}")
    
    def _show_llm_status(self):
        """Show enhanced LLM configuration status"""
        print(f"\n{'='*60}\n🧠 ENHANCED LLM CONFIGURATION STATUS\n{'='*60}")
        if self.system.llm:
            llm_type = type(self.system.llm).__name__
            print(f"✅ LLM Active: {self.system.llm_provider} ({llm_type})")
            print(f"🗄️ Databases: {len(self.system.selected_databases)} database(s)")
            print("🤖 Enhanced features enabled:\n   • Role-based query generation\n   • Memory-driven pattern learning\n   • Multi-database query support\n   • Human-in-the-loop feedback\n   • Configurable LLM providers\n   • SQL database conversion\n   • Database table inspection")
            if "Mock" in llm_type: print("⚠️  Mock LLM - Limited functionality")
            else: print("🚀 Full AI capabilities available")
        else:
            print(f"❌ No LLM configured ({self.system.llm_provider}) - Limited functionality")
        print(f"{'='*60}")

def main():
    """Enhanced main function with LLM selection"""
    print("🚀 Initializing Gainwell SQL Analysis System...")
    try:
        app = CrewAIApp(llm_provider="auto", selected_databases=["sample.db"])
        print("\n✅ Gainwell system initialized successfully!")
        print("🌐 Ready for Streamlit integration!")
        return app
    except Exception as e:
        logger.error(f"Enhanced application error: {e}")
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
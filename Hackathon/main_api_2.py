"""
Enhanced CrewAI SQL Assistant - API System
Filename: main_api.py

This file contains the FastAPI web server with enhanced features:
- WebSocket communication
- SQL conversion endpoints
- Database table inspection API
- File upload and export
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import sqlite3
import asyncio
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import base64
from decimal import Decimal

# Import your existing CrewAI system
from main import CrewAIApp, UserRole

# Pydantic models for API
class ChatMessage(BaseModel):
    id: str
    user_id: str
    message: str
    response: str
    timestamp: datetime
    user_role: str
    iteration: int = 1
    success: bool = True
    sql_query: Optional[str] = None
    chart_path: Optional[str] = None

class QueryRequest(BaseModel):
    message: str
    user_id: str = "admin"
    create_chart: bool = False
    chart_type: str = "bar"
    data_source: str = "auto"

class FeedbackRequest(BaseModel):
    message_id: str
    feedback: str
    user_id: str = "admin"

class SystemConfig(BaseModel):
    llm_provider: str = "auto"
    selected_databases: List[str] = ["sample.db"]
    user_role: str = "admin"

class SqlConversionRequest(BaseModel):
    sql_query: str
    target_database: str
    user_id: str = "admin"

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'chat_history': [],
                'config': SystemConfig(),
                'crewai_app': None
            }

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            try:
                # Use safe serialization for all messages
                safe_message = safe_json_serialize(message)
                await self.active_connections[user_id].send_text(json.dumps(safe_message))
            except Exception as e:
                print(f"❌ Failed to send message to {user_id}: {e}")
                # Remove problematic connection
                if user_id in self.active_connections:
                    del self.active_connections[user_id]

    async def broadcast_message(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_text(json.dumps(message))

# Initialize FastAPI app
app = FastAPI(title="Enhanced CrewAI SQL Assistant", description="Professional AI-powered SQL analysis with conversational interface and database conversion")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager
manager = ConnectionManager()

def safe_json_serialize(data):
    """Safely convert data to JSON-serializable format"""
    if data is None:
        return None
    elif isinstance(data, (str, int, float, bool)):
        # Handle NaN and infinity
        if isinstance(data, float):
            if pd.isna(data) or np.isnan(data):
                return None
            elif np.isinf(data):
                return str(data)
        return data
    elif isinstance(data, (np.integer, np.floating)):
        if pd.isna(data) or np.isnan(data):
            return None
        elif np.isinf(data):
            return str(data)
        return data.item()  # Convert numpy types to Python types
    elif isinstance(data, (np.ndarray,)):
        return [safe_json_serialize(item) for item in data.tolist()]
    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to list of dictionaries
        return data.fillna('').replace([np.inf, -np.inf], '').to_dict('records')
    elif isinstance(data, dict):
        return {key: safe_json_serialize(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [safe_json_serialize(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, Decimal):
        return float(data)
    elif pd.isna(data):
        return None
    else:
        # Try to convert to string as fallback
        try:
            return str(data)
        except:
            return None

async def safe_send_message(user_id: str, message: dict):
    """Safely send message with JSON serialization"""
    try:
        # Serialize the message to ensure it's JSON compatible
        serialized_message = safe_json_serialize(message)
        await manager.send_message(user_id, serialized_message)
        return True
    except Exception as e:
        print(f"❌ Failed to send message to {user_id}: {e}")
        # Try to send a simple error message instead
        try:
            error_message = {
                "type": "error",
                "message": f"Failed to send response: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            await manager.send_message(user_id, error_message)
        except:
            print(f"🚨 Critical: Could not send any message to {user_id}")
        return False

# Store for active CrewAI apps by user
user_apps: Dict[str, CrewAIApp] = {}

# Serve static files (updated path)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main chat interface"""
    # Try to serve the enhanced HTML file
    html_files = ["index.html", "static/index.html"]
    
    for html_file in html_files:
        if os.path.exists(html_file):
            return FileResponse(html_file)
    
    # Fallback HTML if no file found
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced CrewAI SQL Assistant</title>
    </head>
    <body>
        <h1>Enhanced CrewAI SQL Assistant</h1>
        <p>The interface file is missing. Please ensure index.html exists.</p>
    </body>
    </html>
    """)

def get_available_databases():
    """Get list of available database files in current directory"""
    try:
        current_dir = "."
        db_files = []
        
        print(f"🔍 Scanning directory: {os.path.abspath(current_dir)}")
        
        # Look for .db files in current directory
        for file in os.listdir(current_dir):
            if file.endswith(".db"):
                db_path = os.path.join(current_dir, file)
                if os.path.exists(db_path):
                    db_files.append(file)
                    print(f"   ✅ Found database: {file}")
        
        # If no databases found, create sample.db
        if not db_files:
            print("📊 No database files found, creating sample.db...")
            if create_sample_database("sample.db"):
                db_files.append("sample.db")
        
        print(f"📋 Total databases available: {len(db_files)} -> {db_files}")
        return sorted(db_files)
    except Exception as e:
        print(f"❌ Error scanning for databases: {e}")
        return ["sample.db"]  # Fallback

def create_sample_database(db_name: str):
    """Create a sample database if none exists"""
    try:
        conn = sqlite3.connect(db_name)
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
            (5, "Charlie Wilson", "Engineering", 72000, "2021-11-05", 1, "active")
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
        print(f"✅ Created sample database: {db_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample database: {e}")
        return False

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    
    # Initialize user session with default config if not exists
    if user_id not in manager.user_sessions:
        available_dbs = get_available_databases()
        manager.user_sessions[user_id] = {
            'chat_history': [],
            'config': SystemConfig(
                llm_provider="auto",
                selected_databases=available_dbs[:1],  # Default to first available DB
                user_role="admin"  # Default role
            ),
            'crewai_app': None
        }
    
    # Create and configure CrewAI app for the user
    try:
        config = manager.user_sessions[user_id]['config']
        user_apps[user_id] = CrewAIApp(
            llm_provider=config.llm_provider,
            selected_databases=config.selected_databases
        )
        print(f"✅ CrewAI app initialized for user {user_id} with role {config.user_role}")
    except Exception as e:
        print(f"❌ Failed to initialize CrewAI app for {user_id}: {e}")
    
    # Send initial system status with available databases
    available_dbs = get_available_databases()
    await safe_send_message(user_id, {
        "type": "system_status",
        "status": "connected",
        "user_id": user_id,
        "available_databases": available_dbs,
        "current_config": manager.user_sessions[user_id]['config'].dict(),
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "chat_message":
                await handle_chat_message(user_id, message_data)
            elif message_data["type"] == "feedback":
                await handle_feedback(user_id, message_data)
            elif message_data["type"] == "config_update":
                await handle_config_update(user_id, message_data)
            elif message_data["type"] == "export_request":
                await handle_export_request(user_id, message_data)
            elif message_data["type"] == "convert_sql":
                await handle_sql_conversion(user_id, message_data)
            elif message_data["type"] == "get_database_tables":
                await handle_get_database_tables(user_id, message_data)
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)

async def handle_chat_message(user_id: str, message_data: dict):
    """Handle incoming chat messages"""
    try:
        # Get or create CrewAI app for user
        if user_id not in user_apps:
            config = manager.user_sessions[user_id]['config']
            user_apps[user_id] = CrewAIApp(
                llm_provider=config.llm_provider,
                selected_databases=config.selected_databases
            )
        
        app_instance = user_apps[user_id]
        config = manager.user_sessions[user_id]['config']
        
        # Ensure user is registered with the correct role in the CrewAI system
        user_role_str = config.user_role
        if user_role_str == "admin":
            user_role_enum = UserRole.ADMIN
        elif user_role_str == "analyst":
            user_role_enum = UserRole.ANALYST
        elif user_role_str == "viewer":
            user_role_enum = UserRole.VIEWER
        else:
            user_role_enum = UserRole.ADMIN  # Default fallback
        
        # Make sure user is registered in the role manager
        app_instance.system.role_manager.add_user(user_id, user_role_enum)
        
        print(f"🔍 Processing chat message for user {user_id} with role {user_role_str}")
        print(f"📝 Message: {message_data['message']}")
        
        # Send typing indicator
        await safe_send_message(user_id, {
            "type": "typing",
            "status": "processing"
        })
        
        # Generate unique message ID
        message_id = str(uuid.uuid4())
        
        try:
            # Process the query
            print(f"🤖 Starting CrewAI processing...")
            result = app_instance.system.process_request_with_feedback_loop(
                user_request=message_data["message"],
                username=user_id,
                create_chart=message_data.get("create_chart", False),
                chart_type=message_data.get("chart_type", "bar"),
                data_source=message_data.get("data_source", "auto"),
                selected_databases=config.selected_databases
            )
            print(f"✅ CrewAI processing completed, success: {result.get('success', False)}")
            
        except Exception as processing_error:
            print(f"❌ CrewAI processing failed: {processing_error}")
            result = {
                "success": False,
                "error": f"Processing failed: {str(processing_error)}",
                "user_role": user_role_str
            }
        
        # Prepare response - ALWAYS send a response
        response_data = {
            "type": "chat_response",
            "message_id": message_id,
            "user_message": message_data["message"],
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat(),
            "user_role": result.get("user_role", user_role_str)
        }
        
        if result.get("success"):
            try:
                execution_data = result.get("execution_data", {})
                print(f"📊 Execution data keys: {list(execution_data.keys()) if execution_data else 'None'}")
                
                # Extract data more carefully with safe serialization
                data_rows = []
                columns = []
                row_count = 0
                sql_query = ""
                
                if execution_data:
                    # Try different ways to get the data
                    if "data" in execution_data:
                        raw_data = execution_data["data"]
                        if isinstance(raw_data, pd.DataFrame):
                            data_rows = safe_json_serialize(raw_data)
                        elif isinstance(raw_data, list):
                            data_rows = safe_json_serialize(raw_data)
                        else:
                            data_rows = []
                    elif "dataframe" in execution_data:
                        df = execution_data["dataframe"]
                        if df is not None and not df.empty:
                            data_rows = safe_json_serialize(df)
                            columns = df.columns.tolist()
                    
                    row_count = execution_data.get("row_count", len(data_rows) if data_rows else 0)
                    columns = execution_data.get("columns", columns)
                    sql_query = execution_data.get("sql_query", "")
                
                # Also check system-level last execution result
                if not data_rows and hasattr(app_instance.system, 'last_execution_result') and app_instance.system.last_execution_result:
                    last_result = app_instance.system.last_execution_result
                    print(f"📦 Found system last execution result with keys: {list(last_result.keys())}")
                    
                    if "data" in last_result:
                        raw_data = last_result["data"]
                        data_rows = safe_json_serialize(raw_data) if raw_data else []
                    if "columns" in last_result:
                        columns = last_result["columns"] or []
                    if "sql_query" in last_result:
                        sql_query = last_result["sql_query"] or ""
                    row_count = last_result.get("row_count", len(data_rows) if data_rows else 0)
                
                print(f"📈 Final data summary: {len(data_rows) if data_rows else 0} rows, {len(columns) if columns else 0} columns")
                
                # Ensure all data is JSON serializable
                safe_data = safe_json_serialize(data_rows[:100] if data_rows else [])  # Limit for UI performance
                safe_columns = safe_json_serialize(columns) if columns else []
                safe_sql_query = safe_json_serialize(sql_query) if sql_query else ""
                
                response_data.update({
                    "response": f"✅ Query executed successfully! Found {len(safe_data) if safe_data else 0} rows.",
                    "sql_query": safe_sql_query,
                    "row_count": len(safe_data) if safe_data else 0,
                    "columns": safe_columns,
                    "data": safe_data,
                    "agents_used": result.get("agents_used", 0),
                    "llm_provider": result.get("llm_provider", config.llm_provider),
                    "databases_used": result.get("selected_databases", config.selected_databases)
                })
                
                # Check for chart
                if message_data.get("create_chart"):
                    chart_files = [f for f in os.listdir('.') if f.startswith('chart_') and f.endswith('.png')]
                    if chart_files:
                        latest_chart = sorted(chart_files, key=lambda x: os.path.getmtime(x), reverse=True)[0]
                        response_data["chart_path"] = latest_chart
                        print(f"📊 Chart generated: {latest_chart}")
                
            except Exception as data_error:
                print(f"⚠️  Data extraction error: {data_error}")
                response_data.update({
                    "response": f"Query completed but data extraction failed: {str(data_error)}",
                    "sql_query": "",
                    "row_count": 0,
                    "columns": [],
                    "data": []
                })
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            response_data.update({
                "response": f"❌ Query failed: {error_msg}",
                "error": error_msg,
                "sql_query": "",
                "row_count": 0,
                "columns": [],
                "data": []
            })
            print(f"❌ Query failed for user {user_id}: {error_msg}")
        
        # Store in chat history
        chat_message = ChatMessage(
            id=message_id,
            user_id=user_id,
            message=message_data["message"],
            response=response_data["response"],
            timestamp=datetime.now(),
            user_role=response_data["user_role"],
            success=result.get("success", False),
            sql_query=response_data.get("sql_query"),
            chart_path=response_data.get("chart_path")
        )
        
        manager.user_sessions[user_id]['chat_history'].append(chat_message)
        
        # ALWAYS send response using safe serialization
        print(f"📤 Sending response to user {user_id}")
        success = await safe_send_message(user_id, response_data)
        if success:
            print(f"✅ Response sent successfully")
        else:
            print(f"❌ Failed to send response")
        
    except Exception as e:
        error_msg = f"Critical error processing request: {str(e)}"
        print(f"🚨 CRITICAL ERROR for user {user_id}: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Always try to send an error response using safe serialization
        try:
            error_response = {
                "type": "chat_response",
                "message_id": str(uuid.uuid4()),
                "user_message": message_data.get("message", "Unknown"),
                "success": False,
                "response": f"❌ System error: {error_msg}",
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
                "user_role": "unknown",
                "sql_query": "",
                "row_count": 0,
                "columns": [],
                "data": []
            }
            await safe_send_message(user_id, error_response)
        except Exception as send_error:
            print(f"🚨 Failed to send error response: {send_error}")

async def handle_sql_conversion(user_id: str, message_data: dict):
    """Handle SQL conversion requests"""
    try:
        sql_query = message_data.get("sql_query", "")
        target_database = message_data.get("target_database", "postgresql")
        
        if not sql_query:
            await safe_send_message(user_id, {
                "type": "sql_conversion",
                "success": False,
                "error": "No SQL query provided for conversion",
                "target_database": target_database
            })
            return
        
        # Get CrewAI app for user
        app_instance = user_apps.get(user_id)
        if not app_instance:
            await safe_send_message(user_id, {
                "type": "sql_conversion", 
                "success": False,
                "error": "No active session found",
                "target_database": target_database
            })
            return
        
        print(f"🔄 Converting SQL for user {user_id} to {target_database}")
        print(f"📝 Original SQL: {sql_query}")
        
        # Use the SQL converter from the CrewAI system
        conversion_result = app_instance.system.convert_sql_to_target_db(sql_query, target_database)
        
        response_data = {
            "type": "sql_conversion",
            "success": conversion_result.get("success", False),
            "target_database": target_database,
            "original_query": sql_query
        }
        
        if conversion_result.get("success"):
            response_data.update({
                "converted_query": conversion_result["converted_query"],
                "conversion_notes": conversion_result.get("conversion_notes", "")
            })
            print(f"✅ SQL conversion successful for {target_database}")
        else:
            response_data.update({
                "error": conversion_result.get("error", "Conversion failed")
            })
            print(f"❌ SQL conversion failed: {conversion_result.get('error')}")
        
        await safe_send_message(user_id, response_data)
        
    except Exception as e:
        error_msg = f"SQL conversion error: {str(e)}"
        print(f"❌ {error_msg}")
        
        await safe_send_message(user_id, {
            "type": "sql_conversion",
            "success": False,
            "error": error_msg,
            "target_database": message_data.get("target_database", "unknown")
        })

async def handle_get_database_tables(user_id: str, message_data: dict):
    """Handle database tables listing requests"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            await safe_send_message(user_id, {
                "type": "database_tables",
                "success": False,
                "error": "No active session found"
            })
            return
        
        print(f"📊 Getting database tables for user {user_id}")
        
        # Get database tables info from the CrewAI system
        tables_result = app_instance.system.get_database_tables_info(user_id)
        
        response_data = {
            "type": "database_tables",
            "success": tables_result.get("success", False),
            "user_id": user_id
        }
        
        if tables_result.get("success"):
            response_data.update({
                "databases": tables_result["databases"],
                "user_role": tables_result.get("user_role", "unknown"),
                "total_databases": tables_result.get("total_databases", 0),
                "total_accessible_tables": tables_result.get("total_accessible_tables", 0)
            })
            print(f"✅ Database tables retrieved: {tables_result.get('total_databases')} databases, {tables_result.get('total_accessible_tables')} tables")
        else:
            response_data.update({
                "error": tables_result.get("error", "Failed to get database tables")
            })
            print(f"❌ Failed to get database tables: {tables_result.get('error')}")
        
        await safe_send_message(user_id, response_data)
        
    except Exception as e:
        error_msg = f"Database tables error: {str(e)}"
        print(f"❌ {error_msg}")
        
        await safe_send_message(user_id, {
            "type": "database_tables",
            "success": False,
            "error": error_msg
        })

async def handle_feedback(user_id: str, message_data: dict):
    """Handle feedback on previous responses"""
    try:
        message_id = message_data["message_id"]
        feedback = message_data["feedback"]
        
        # Find the original message
        chat_history = manager.user_sessions[user_id]['chat_history']
        original_message = None
        for msg in chat_history:
            if msg.id == message_id:
                original_message = msg
                break
        
        if not original_message:
            await safe_send_message(user_id, {
                "type": "error",
                "message": "Original message not found for feedback"
            })
            return
        
        app_instance = user_apps[user_id]
        
        # Send processing indicator
        await safe_send_message(user_id, {
            "type": "typing",
            "status": "processing_feedback"
        })
        
        # Apply feedback
        previous_result = {
            "request": original_message.message,
            "iteration": original_message.iteration,
            "selected_databases": manager.user_sessions[user_id]['config'].selected_databases
        }
        
        result = app_instance.system.apply_feedback_and_retry(
            previous_result=previous_result,
            feedback=feedback,
            username=user_id,
            create_chart=message_data.get("create_chart", False),
            chart_type=message_data.get("chart_type", "bar")
        )
        
        # Generate new message ID for feedback response
        new_message_id = str(uuid.uuid4())
        
        response_data = {
            "type": "feedback_response",
            "message_id": new_message_id,
            "original_message_id": message_id,
            "feedback": feedback,
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat(),
            "iteration": original_message.iteration + 1
        }
        
        if result.get("success"):
            execution_data = result.get("execution_data", {})
            response_data.update({
                "response": "Feedback applied successfully! 🔄",
                "sql_query": execution_data.get("sql_query", ""),
                "row_count": execution_data.get("row_count", 0),
                "data": safe_json_serialize(execution_data.get("data", [])[:100])
            })
        else:
            response_data.update({
                "response": f"Feedback application failed: {result.get('error', 'Unknown error')}",
                "error": result.get('error', 'Unknown error')
            })
        
        await safe_send_message(user_id, response_data)
        
    except Exception as e:
        await safe_send_message(user_id, {
            "type": "error",
            "message": f"Error processing feedback: {str(e)}"
        })

async def handle_config_update(user_id: str, message_data: dict):
    """Handle system configuration updates"""
    try:
        config_data = message_data["config"]
        
        # Validate databases exist
        requested_databases = config_data.get("selected_databases", ["sample.db"])
        available_databases = get_available_databases()
        valid_databases = [db for db in requested_databases if db in available_databases]
        
        if not valid_databases:
            valid_databases = [available_databases[0]] if available_databases else ["sample.db"]
        
        # Update user session config
        manager.user_sessions[user_id]['config'] = SystemConfig(
            llm_provider=config_data.get("llm_provider", "auto"),
            selected_databases=valid_databases,
            user_role=config_data.get("user_role", "admin")
        )
        
        # Recreate CrewAI app with new config
        config = manager.user_sessions[user_id]['config']
        user_apps[user_id] = CrewAIApp(
            llm_provider=config.llm_provider,
            selected_databases=config.selected_databases
        )
        
        # Register user with correct role
        user_role_str = config.user_role
        if user_role_str == "admin":
            user_role_enum = UserRole.ADMIN
        elif user_role_str == "analyst":
            user_role_enum = UserRole.ANALYST
        elif user_role_str == "viewer":
            user_role_enum = UserRole.VIEWER
        else:
            user_role_enum = UserRole.ADMIN
        
        user_apps[user_id].system.role_manager.add_user(user_id, user_role_enum)
        
        print(f"✅ Updated configuration for user {user_id}:")
        print(f"   - LLM Provider: {config.llm_provider}")
        print(f"   - Databases: {config.selected_databases}")
        print(f"   - User Role: {config.user_role}")
        
        await safe_send_message(user_id, {
            "type": "config_updated",
            "status": "success",
            "config": {
                "llm_provider": config.llm_provider,
                "selected_databases": config.selected_databases,
                "user_role": config.user_role
            },
            "message": f"Configuration updated successfully! Using {len(config.selected_databases)} database(s) with {config.llm_provider} LLM as {config.user_role}.",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Config update error for user {user_id}: {e}")
        await safe_send_message(user_id, {
            "type": "error",
            "message": f"Error updating configuration: {str(e)}"
        })

async def handle_export_request(user_id: str, message_data: dict):
    """Handle data export requests"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            await safe_send_message(user_id, {
                "type": "error",
                "message": "No active session found for export"
            })
            return
        
        # Export last execution result
        filename = f"export_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        result = app_instance.system.export_results(filename=filename)
        
        await safe_send_message(user_id, {
            "type": "export_ready",
            "filename": filename,
            "message": result,
            "download_url": f"/download/{filename}",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        await safe_send_message(user_id, {
            "type": "error",
            "message": f"Export failed: {str(e)}"
        })

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...), name: str = Form(...)):
    """Handle file uploads"""
    try:
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Register with CrewAI system
        app_instance = user_apps.get(user_id)
        if app_instance:
            result = app_instance.system.register_file(name, file_path, user_id)
            return {"success": result["success"], "message": result.get("message", "")}
        else:
            return {"success": False, "message": "No active session"}
            
    except Exception as e:
        return {"success": False, "message": f"Upload failed: {str(e)}"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download exported files"""
    if os.path.exists(filename):
        return FileResponse(filename, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/chart/{filename}")
async def get_chart(filename: str):
    """Serve chart images"""
    if os.path.exists(filename) and filename.endswith('.png'):
        return FileResponse(filename)
    else:
        raise HTTPException(status_code=404, detail="Chart not found")

@app.get("/api/system_info")
async def get_system_info():
    """Get system configuration information"""
    try:
        # Get actual available databases from current directory
        available_dbs = get_available_databases()
        
        return {
            "available_llm_providers": {
                "Auto (Best Available)": "auto",
                "Groq (Fast)": "groq",
                "OpenAI (GPT-3.5)": "openai", 
                "Anthropic (Claude)": "anthropic",
                "Ollama (Local)": "ollama",
                "Mock LLM (No API Key)": "mock"
            },
            "available_databases": available_dbs,
            "user_roles": ["admin", "analyst", "viewer"],
            "chart_types": ["bar", "pie"],
            "supported_db_conversions": ["postgresql", "sqlserver", "db2"],
            "version": "2.1.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat_history/{user_id}")
async def get_chat_history(user_id: str):
    """Get chat history for a user"""
    try:
        if user_id in manager.user_sessions:
            history = manager.user_sessions[user_id]['chat_history']
            return [msg.dict() for msg in history]
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat_history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a user"""
    try:
        if user_id in manager.user_sessions:
            manager.user_sessions[user_id]['chat_history'] = []
        return {"success": True, "message": "Chat history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/convert_sql")
async def convert_sql_query(request: SqlConversionRequest):
    """Convert SQL query to different database dialects"""
    try:
        app_instance = user_apps.get(request.user_id)
        if not app_instance:
            raise HTTPException(status_code=404, detail="No active session found")
        
        conversion_result = app_instance.system.convert_sql_to_target_db(
            request.sql_query, 
            request.target_database
        )
        
        return conversion_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database_tables/{user_id}")
async def get_database_tables(user_id: str):
    """Get database tables information for a user"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            raise HTTPException(status_code=404, detail="No active session found")
        
        tables_result = app_instance.system.get_database_tables_info(user_id)
        return tables_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0",
        "features": [
            "SQL Generation",
            "Database Conversion", 
            "Table Inspection",
            "Role-Based Access",
            "Multi-Database Support"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced CrewAI SQL Assistant API...")
    print("🔧 New Features:")
    print("   • SQL dialect conversion (PostgreSQL, SQL Server, DB2)")
    print("   • Database table inspection and listing")
    print("   • Enhanced role-based access control")
    print("   • Copy to clipboard functionality")
    uvicorn.run(app, host="0.0.0.0", port=8000)
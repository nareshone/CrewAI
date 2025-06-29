"""
Enhanced Gainwell SQL Assistant - API System
Refactored for clean CrewAI kickoff workflow
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import asyncio
import uuid
from datetime import datetime
import pandas as pd
import numpy as np

# Import the refactored CrewAI system
from main import CrewAIApp, UserRole

# Pydantic models
class ChatMessage(BaseModel):
    id: str
    user_id: str
    message: str
    response: str
    timestamp: datetime
    user_role: str
    success: bool = True
    sql_query: Optional[str] = None

class SystemConfig(BaseModel):
    llm_provider: str = "auto"
    selected_databases: List[str] = ["sample.db"]
    user_role: str = "admin"

class SqlConversionRequest(BaseModel):
    sql_query: str
    target_database: str
    user_id: str = "admin"

# Simplified Connection Manager
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
                'config': SystemConfig()
            }

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            try:
                safe_message = safe_json_serialize(message)
                await self.active_connections[user_id].send_text(json.dumps(safe_message))
            except Exception as e:
                print(f"❌ Failed to send message to {user_id}: {e}")
                if user_id in self.active_connections:
                    del self.active_connections[user_id]

# Safe JSON serialization
def safe_json_serialize(data):
    """Convert data to JSON-serializable format"""
    if data is None:
        return None
    elif isinstance(data, (str, int, float, bool)):
        if isinstance(data, float) and (pd.isna(data) or np.isnan(data) or np.isinf(data)):
            return None if pd.isna(data) or np.isnan(data) else str(data)
        return data
    elif isinstance(data, (np.integer, np.floating)):
        if pd.isna(data) or np.isnan(data) or np.isinf(data):
            return None if pd.isna(data) or np.isnan(data) else str(data)
        return data.item()
    elif isinstance(data, pd.DataFrame):
        return data.fillna('').replace([np.inf, -np.inf], '').to_dict('records')
    elif isinstance(data, dict):
        return {key: safe_json_serialize(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [safe_json_serialize(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        try:
            return str(data)
        except:
            return None

# Initialize FastAPI
app = FastAPI(title="Enhanced CrewAI SQL Assistant", description="AI-powered SQL analysis with CrewAI workflow")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global managers
manager = ConnectionManager()
user_apps: Dict[str, CrewAIApp] = {}

# Serve static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main interface"""
    html_files = ["index.html", "static/index.html"]
    for html_file in html_files:
        if os.path.exists(html_file):
            return FileResponse(html_file)
    
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>Enhanced CrewAI SQL Assistant</title></head>
    <body>
        <h1>Enhanced CrewAI SQL Assistant</h1>
        <p>Interface file missing. Please ensure index.html exists.</p>
    </body>
    </html>
    """)

def get_available_databases():
    """Get available database files"""
    try:
        db_files = [f for f in os.listdir(".") if f.endswith(".db")]
        if not db_files:
            create_sample_database("sample.db")
            db_files.append("sample.db")
        return sorted(db_files)
    except Exception as e:
        print(f"❌ Error scanning databases: {e}")
        return ["sample.db"]

def create_sample_database(db_name: str):
    """Create sample database if needed"""
    try:
        import sqlite3
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute('''CREATE TABLE employees (
            id INTEGER PRIMARY KEY, name TEXT, department TEXT, 
            salary REAL, hire_date DATE, status TEXT DEFAULT 'active'
        )''')
        
        cursor.execute('''CREATE TABLE departments (
            id INTEGER PRIMARY KEY, name TEXT, budget REAL, location TEXT
        )''')
        
        employees = [
            (1, "John Doe", "Engineering", 75000, "2022-01-15", "active"),
            (2, "Jane Smith", "Marketing", 65000, "2021-03-20", "active"),
            (3, "Bob Johnson", "Engineering", 80000, "2020-07-10", "active"),
            (4, "Alice Brown", "HR", 60000, "2023-02-01", "active")
        ]
        
        departments = [
            (1, "Engineering", 500000, "Building A"),
            (2, "Marketing", 300000, "Building B"),
            (3, "HR", 200000, "Building C")
        ]
        
        cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)", employees)
        cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments)
        
        conn.commit()
        conn.close()
        print(f"✅ Created sample database: {db_name}")
    except Exception as e:
        print(f"❌ Failed to create sample database: {e}")

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    
    # Initialize session
    if user_id not in manager.user_sessions:
        available_dbs = get_available_databases()
        manager.user_sessions[user_id] = {
            'chat_history': [],
            'config': SystemConfig(selected_databases=[available_dbs[0]] if available_dbs else ["sample.db"])
        }
    
    # Create CrewAI app
    try:
        config = manager.user_sessions[user_id]['config']
        user_apps[user_id] = CrewAIApp(
            llm_provider=config.llm_provider,
            selected_databases=config.selected_databases
        )
        print(f"✅ CrewAI app initialized for user {user_id}")
    except Exception as e:
        print(f"❌ Failed to initialize CrewAI app: {e}")
    
    # Send initial status
    await manager.send_message(user_id, {
        "type": "system_status",
        "status": "connected",
        "available_databases": get_available_databases(),
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
            elif message_data["type"] == "get_database_tables":
                await handle_get_database_tables(user_id, message_data)
            elif message_data["type"] == "convert_sql":
                await handle_sql_conversion(user_id, message_data)
            elif message_data["type"] == "export_request":
                await handle_export_request(user_id, message_data)
            elif message_data["type"] == "refresh_schema":
                await handle_refresh_schema(user_id, message_data)
            elif message_data["type"] == "get_uploaded_files":
                await handle_get_uploaded_files(user_id, message_data)
            elif message_data["type"] == "delete_uploaded_file":
                await handle_delete_uploaded_file(user_id, message_data)
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)

async def handle_chat_message(user_id: str, message_data: dict):
    """Handle chat messages using CrewAI kickoff workflow"""
    try:
        # Get CrewAI app
        if user_id not in user_apps:
            config = manager.user_sessions[user_id]['config']
            user_apps[user_id] = CrewAIApp(
                llm_provider=config.llm_provider,
                selected_databases=config.selected_databases
            )
        
        app_instance = user_apps[user_id]
        config = manager.user_sessions[user_id]['config']
        
        # Register user with correct role
        user_role = getattr(UserRole, config.user_role.upper(), UserRole.ADMIN)
        app_instance.system.role_manager.add_user(user_id, user_role)
        
        print(f"🚀 Processing request with CrewAI kickoff for {user_id}")
        print(f"📝 Message: {message_data['message']}")
        
        # Send typing indicator
        await manager.send_message(user_id, {
            "type": "typing",
            "status": "processing"
        })
        
        message_id = str(uuid.uuid4())
        
        try:
            # Use the new CrewAI kickoff workflow
            result = app_instance.system.process_request(
                user_request=message_data["message"],
                username=user_id,
                create_chart=message_data.get("create_chart", False),
                chart_type=message_data.get("chart_type", "bar")
            )
            
            print(f"✅ CrewAI kickoff completed, success: {result.get('success', False)}")
            
        except Exception as processing_error:
            print(f"❌ CrewAI processing failed: {processing_error}")
            result = {
                "success": False,
                "error": f"Processing failed: {str(processing_error)}"
            }
        
        # Prepare response
        response_data = {
            "type": "chat_response",
            "message_id": message_id,
            "user_message": message_data["message"],
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat(),
            "user_role": config.user_role
        }
        
        if result.get("success"):
            execution_data = result.get("execution_data", {})
            
            # Extract data safely
            data_rows = safe_json_serialize(execution_data.get("data", []))
            columns = execution_data.get("columns", [])
            sql_query = execution_data.get("sql_query", "")
            row_count = execution_data.get("row_count", 0)
            
            # Check if this is an access denied response (error_message column in data)
            if (data_rows and len(data_rows) > 0 and 
                isinstance(data_rows[0], dict) and 
                "error_message" in data_rows[0]):
                # This is actually an access denied error
                error_msg = data_rows[0]["error_message"]
                response_data.update({
                    "success": False,
                    "response": f"❌ {error_msg}",
                    "error": error_msg,
                    "access_denied": True,
                    "sql_query": sql_query,
                    "row_count": 0,
                    "columns": [],
                    "data": []
                })
            else:
                # Normal successful response
                response_data.update({
                    "response": f"✅ Query executed successfully using CrewAI workflow! Found {row_count} rows.",
                    "sql_query": sql_query,
                    "row_count": row_count,
                    "columns": columns,
                    "data": data_rows[:100] if data_rows else [],  # Limit for UI performance
                    "agents_used": result.get("agents_used", 4),
                    "llm_provider": result.get("llm_provider", config.llm_provider)
                })
            
            # Add chart if created
            if message_data.get("create_chart") and "chart_path" in execution_data:
                response_data["chart_path"] = execution_data["chart_path"]
                
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            
            # Check if this is an access denied error
            if "access denied" in error_msg.lower() or "does not have access" in error_msg.lower():
                response_data.update({
                    "response": f"🔒 {error_msg}",
                    "error": error_msg,
                    "access_denied": True,
                    "sql_query": "",
                    "row_count": 0,
                    "columns": [],
                    "data": []
                })
            else:
                response_data.update({
                    "response": f"❌ Query failed: {error_msg}",
                    "error": error_msg,
                    "sql_query": "",
                    "row_count": 0,
                    "columns": [],
                    "data": []
                })
        
        # Store in chat history
        chat_message = ChatMessage(
            id=message_id,
            user_id=user_id,
            message=message_data["message"],
            response=response_data["response"],
            timestamp=datetime.now(),
            user_role=config.user_role,
            success=result.get("success", False),
            sql_query=response_data.get("sql_query", "")
        )
        
        manager.user_sessions[user_id]['chat_history'].append(chat_message)
        response_data["message_id"] = message_id
        
        # Send response
        await manager.send_message(user_id, response_data)
        print(f"✅ Response sent to user {user_id}")
        
    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        print(f"🚨 CRITICAL ERROR for {user_id}: {error_msg}")
        
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
        await manager.send_message(user_id, error_response)

async def handle_feedback(user_id: str, message_data: dict):
    """Handle feedback with enhanced context"""
    try:
        message_id = message_data["message_id"]
        feedback = message_data["feedback"]
        
        # Find original message
        chat_history = manager.user_sessions[user_id]['chat_history']
        original_message = None
        for msg in chat_history:
            if msg.id == message_id:
                original_message = msg
                break
        
        if not original_message:
            await manager.send_message(user_id, {
                "type": "error",
                "message": "Original message not found"
            })
            return
        
        app_instance = user_apps[user_id]
        
        await manager.send_message(user_id, {
            "type": "typing",
            "status": "processing_feedback"
        })
        
        print(f"🔄 Processing feedback for user {user_id}")
        
        # Process with CrewAI kickoff
        result = app_instance.system.process_request(
            user_request=f"FEEDBACK: {feedback} | ORIGINAL REQUEST: {original_message.message}",
            username=user_id,
            create_chart=message_data.get("create_chart", False),
            chart_type=message_data.get("chart_type", "bar")
        )
        
        new_message_id = str(uuid.uuid4())
        
        response_data = {
            "type": "feedback_response",
            "message_id": new_message_id,
            "original_message_id": message_id,
            "feedback": feedback,
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat()
        }
        
        if result.get("success"):
            execution_data = result.get("execution_data", {})
            response_data.update({
                "response": "🔄 Feedback applied successfully using CrewAI workflow!",
                "sql_query": execution_data.get("sql_query", ""),
                "row_count": execution_data.get("row_count", 0),
                "columns": execution_data.get("columns", []),
                "data": safe_json_serialize(execution_data.get("data", []))[:100]
            })
            
            # Store improved response
            improved_message = ChatMessage(
                id=new_message_id,
                user_id=user_id,
                message=f"[FEEDBACK] {original_message.message}",
                response=response_data["response"],
                timestamp=datetime.now(),
                user_role=original_message.user_role,
                success=True,
                sql_query=execution_data.get("sql_query", "")
            )
            manager.user_sessions[user_id]['chat_history'].append(improved_message)
        else:
            response_data.update({
                "response": f"❌ Feedback application failed: {result.get('error', 'Unknown error')}",
                "error": result.get('error', 'Unknown error')
            })
        
        await manager.send_message(user_id, response_data)
        print(f"✅ Feedback processed for user {user_id}")
        
    except Exception as e:
        await manager.send_message(user_id, {
            "type": "error",
            "message": f"Feedback processing error: {str(e)}"
        })

async def handle_config_update(user_id: str, message_data: dict):
    """Handle configuration updates"""
    try:
        config_data = message_data["config"]
        
        # Validate databases
        requested_databases = config_data.get("selected_databases", ["sample.db"])
        available_databases = get_available_databases()
        valid_databases = [db for db in requested_databases if db in available_databases]
        
        if not valid_databases:
            valid_databases = [available_databases[0]] if available_databases else ["sample.db"]

        # Update config
        current_config = manager.user_sessions[user_id]['config']
        current_config.llm_provider = config_data.get("llm_provider", "auto")
        current_config.selected_databases = valid_databases
        current_config.user_role = config_data.get("user_role", "admin")
        
        # Recreate CrewAI app
        print(f"✅ Recreating CrewAI app for {user_id} with {len(valid_databases)} database(s)")
        user_apps[user_id] = CrewAIApp(
            llm_provider=current_config.llm_provider,
            selected_databases=current_config.selected_databases
        )
        
        # Register user
        user_role = getattr(UserRole, current_config.user_role.upper(), UserRole.ADMIN)
        user_apps[user_id].system.role_manager.add_user(user_id, user_role)
        
        await manager.send_message(user_id, {
            "type": "config_updated",
            "status": "success",
            "config": current_config.dict(),
            "message": f"Configuration updated. Using {len(valid_databases)} database(s) as {current_config.user_role}.",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        await manager.send_message(user_id, {
            "type": "error",
            "message": f"Configuration update error: {str(e)}"
        })

async def handle_get_database_tables(user_id: str, message_data: dict):
    """Handle database tables requests"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            await manager.send_message(user_id, {
                "type": "database_tables",
                "success": False,
                "error": "No active session"
            })
            return
        
        tables_result = app_instance.system.get_database_tables_info(user_id)
        
        response_data = {
            "type": "database_tables",
            "success": tables_result.get("success", False),
            "user_id": user_id
        }
        
        if tables_result.get("success"):
            response_data.update({
                "databases": tables_result["databases"],
                "user_role": tables_result.get("user_role", "unknown")
            })
        else:
            response_data["error"] = tables_result.get("error", "Failed to get tables")
        
        await manager.send_message(user_id, response_data)
        
    except Exception as e:
        await manager.send_message(user_id, {
            "type": "database_tables",
            "success": False,
            "error": str(e)
        })

async def handle_sql_conversion(user_id: str, message_data: dict):
    """Handle SQL conversion requests"""
    try:
        sql_query = message_data.get("sql_query", "")
        target_database = message_data.get("target_database", "postgresql")
        
        if not sql_query:
            await manager.send_message(user_id, {
                "type": "sql_conversion",
                "success": False,
                "error": "No SQL query provided for conversion",
                "target_database": target_database
            })
            return
        
        # Get CrewAI app for user
        app_instance = user_apps.get(user_id)
        if not app_instance:
            await manager.send_message(user_id, {
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
        
        await manager.send_message(user_id, response_data)
        
    except Exception as e:
        error_msg = f"SQL conversion error: {str(e)}"
        print(f"❌ {error_msg}")
        
        await manager.send_message(user_id, {
            "type": "sql_conversion",
            "success": False,
            "error": error_msg,
            "target_database": message_data.get("target_database", "unknown")
        })

async def handle_export_request(user_id: str, message_data: dict):
    """Handle data export requests"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            await manager.send_message(user_id, {
                "type": "error",
                "message": "No active session found for export"
            })
            return
        
        # Export last execution result
        filename = f"export_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        result = app_instance.system.export_results(filename=filename)
        
        if result.startswith("Results exported"):
            await manager.send_message(user_id, {
                "type": "export_ready",
                "filename": filename,
                "message": result,
                "download_url": f"/download/{filename}",
                "timestamp": datetime.now().isoformat()
            })
        else:
            await manager.send_message(user_id, {
                "type": "error",
                "message": result
            })
        
    except Exception as e:
        await manager.send_message(user_id, {
            "type": "error",
            "message": f"Export failed: {str(e)}"
        })

async def handle_refresh_schema(user_id: str, message_data: dict):
    """Handle schema refresh requests"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            await manager.send_message(user_id, {
                "type": "refresh_schema",
                "success": False,
                "error": "No active session found"
            })
            return
        
        print(f"🔄 Refreshing database schema for user {user_id}")
        
        # Refresh schema and get updated tables
        refresh_result = app_instance.system.refresh_database_schema(user_id)
        
        response_data = {
            "type": "refresh_schema",
            "success": refresh_result.get("success", False),
            "user_id": user_id,
            "message": refresh_result.get("message", "")
        }
        
        if refresh_result.get("success"):
            # Also send updated database tables
            tables_result = app_instance.system.get_database_tables_info(user_id)
            if tables_result.get("success"):
                response_data.update({
                    "databases": tables_result["databases"],
                    "user_role": tables_result.get("user_role", "unknown")
                })
            print(f"✅ Schema refreshed successfully for user {user_id}")
        else:
            response_data["error"] = refresh_result.get("message", "Schema refresh failed")
            print(f"❌ Schema refresh failed for user {user_id}")
        
        await manager.send_message(user_id, response_data)
        
    except Exception as e:
        error_msg = f"Schema refresh error: {str(e)}"
        print(f"❌ {error_msg}")
        
        await manager.send_message(user_id, {
            "type": "refresh_schema",
            "success": False,
            "error": error_msg
        })

async def handle_get_uploaded_files(user_id: str, message_data: dict):
    """Handle get uploaded files requests"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            await manager.send_message(user_id, {
                "type": "uploaded_files",
                "success": False,
                "error": "No active session found"
            })
            return
        
        files_result = app_instance.system.get_uploaded_files_info(user_id)
        
        response_data = {
            "type": "uploaded_files",
            "success": files_result.get("success", False),
            "user_id": user_id
        }
        
        if files_result.get("success"):
            response_data.update({
                "files": files_result.get("files", []),
                "count": files_result.get("count", 0)
            })
        else:
            response_data["error"] = files_result.get("message", "Failed to get uploaded files")
        
        await manager.send_message(user_id, response_data)
        
    except Exception as e:
        await manager.send_message(user_id, {
            "type": "uploaded_files",
            "success": False,
            "error": str(e)
        })

async def handle_delete_uploaded_file(user_id: str, message_data: dict):
    """Handle delete uploaded file requests"""
    try:
        table_name = message_data.get("table_name", "")
        if not table_name:
            await manager.send_message(user_id, {
                "type": "delete_file",
                "success": False,
                "error": "No table name provided"
            })
            return
        
        app_instance = user_apps.get(user_id)
        if not app_instance:
            await manager.send_message(user_id, {
                "type": "delete_file",
                "success": False,
                "error": "No active session found"
            })
            return
        
        delete_result = app_instance.system.delete_uploaded_file(table_name, user_id)
        
        response_data = {
            "type": "delete_file",
            "success": delete_result.get("success", False),
            "table_name": table_name,
            "message": delete_result.get("message", "")
        }
        
        if delete_result.get("success"):
            # Also send updated database tables
            tables_result = app_instance.system.get_database_tables_info(user_id)
            if tables_result.get("success"):
                response_data.update({
                    "databases": tables_result["databases"]
                })
        else:
            response_data["error"] = delete_result.get("message", "File deletion failed")
        
        await manager.send_message(user_id, response_data)
        
    except Exception as e:
        await manager.send_message(user_id, {
            "type": "delete_file",
            "success": False,
            "error": str(e)
        })

# REST API endpoints
@app.get("/api/uploaded_files/{user_id}")
async def get_uploaded_files(user_id: str):
    """Get list of uploaded files for a user"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            raise HTTPException(status_code=404, detail="No active session found")
        
        files_result = app_instance.system.get_uploaded_files_info(user_id)
        return files_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/uploaded_files/{user_id}/{table_name}")
async def delete_uploaded_file_api(user_id: str, table_name: str):
    """Delete an uploaded file via REST API"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            raise HTTPException(status_code=404, detail="No active session found")
        
        delete_result = app_instance.system.delete_uploaded_file(table_name, user_id)
        return delete_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh_schema/{user_id}")
async def refresh_schema_api(user_id: str):
    """Refresh database schema via REST API"""
    try:
        app_instance = user_apps.get(user_id)
        if not app_instance:
            raise HTTPException(status_code=404, detail="No active session found")
        
        refresh_result = app_instance.system.refresh_database_schema(user_id)
        return refresh_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system_info")
async def get_system_info():
    """Get system information"""
    return {
        "available_llm_providers": {
            "Auto (Best Available)": "auto",
            "Groq (Fast)": "groq", 
            "OpenAI (GPT-3.5)": "openai",
            "Anthropic (Claude)": "anthropic",
            "Ollama (Local)": "ollama",
            "Mock LLM (Testing)": "mock"
        },
        "available_databases": get_available_databases(),
        "user_roles": ["admin", "analyst", "viewer"],
        "chart_types": ["bar", "pie"],
        "supported_db_conversions": ["postgresql", "sqlserver", "db2"],
        "supported_file_types": [".csv", ".xlsx", ".xls", ".json"],
        "file_upload_enabled": True,
        "version": "3.1.0 - Full File Upload Edition"
    }

@app.get("/api/chat_history/{user_id}")
async def get_chat_history(user_id: str):
    """Get chat history"""
    try:
        if user_id in manager.user_sessions:
            history = manager.user_sessions[user_id]['chat_history']
            return [msg.dict() for msg in history]
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat_history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history"""
    try:
        if user_id in manager.user_sessions:
            manager.user_sessions[user_id]['chat_history'] = []
        return {"success": True, "message": "Chat history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert_sql")
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

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download exported files"""
    if os.path.exists(filename):
        return FileResponse(filename, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...), name: str = Form(...)):
    """Handle file uploads and register with CrewAI system"""
    try:
        print(f"📁 Receiving file upload: {file.filename} from user {user_id} as '{name}'")
        
        # Validate file type
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.json'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return {
                "success": False, 
                "message": f"Unsupported file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}"
            }
        
        # Create upload directory
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(upload_dir, safe_filename)
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"📁 File saved to: {file_path}")
        
        # Get CrewAI app instance for the user
        app_instance = user_apps.get(user_id)
        if not app_instance:
            # Clean up file if no app instance
            os.remove(file_path)
            return {
                "success": False,
                "message": "No active session found. Please refresh the page and try again."
            }
        
        # Register file with the CrewAI system
        registration_result = app_instance.system.register_file(name, file_path, user_id)
        
        # Clean up temporary file after processing
        try:
            os.remove(file_path)
            print(f"🗑️ Cleaned up temporary file: {file_path}")
        except:
            pass  # Don't fail if cleanup fails
        
        if registration_result["success"]:
            print(f"✅ File '{name}' registered successfully with CrewAI system")
            
            # Get updated table information
            #tables_result = app_instance.system.get_database_tables_info(user_id)
            try:
                tables_result = app_instance.system.get_database_tables_info(user_id)
                if tables_result.get("success"):
                    response["updated_databases"] = tables_result["databases"]
            except Exception as table_error:
                print(f"⚠️ Could not get updated tables (upload still successful): {table_error}")
            
            response = {
                "success": True,
                "message": registration_result["message"],
                "table_name": registration_result.get("table_name", ""),
                "rows": registration_result.get("rows", 0),
                "columns": registration_result.get("columns", [])
            }
            
            # Include updated database tables if available
            if tables_result.get("success"):
                response["updated_databases"] = tables_result["databases"]
            
            return response
        else:
            print(f"❌ File registration failed: {registration_result['message']}")
            return {
                "success": False,
                "message": registration_result["message"]
            }
        
    except Exception as e:
        error_msg = f"File upload failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Clean up file if something went wrong
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
        
        return {
            "success": False,
            "message": error_msg
        }

@app.get("/chart/{filename}")
async def get_chart(filename: str):
    """Serve chart images"""
    if os.path.exists(filename) and filename.endswith('.png'):
        return FileResponse(filename)
    else:
        raise HTTPException(status_code=404, detail="Chart not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.1.0 - Full File Upload Edition",
        "features": [
            "CrewAI Kickoff Workflow",
            "4-Agent Collaboration", 
            "SQL Generation & Validation",
            "Query Execution with Role-Based Access",
            "Chart Generation (Fixed)",
            "SQL Database Conversion (PostgreSQL, SQL Server, DB2)",
            "Export Results to Excel",
            "Role-Based Access Control (Secured)",
            "Multi-Database Support",
            "File Upload & Processing (CSV, Excel, JSON)",
            "Dynamic Schema Updates",
            "Uploaded File Management"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced CrewAI SQL Assistant API (Full File Upload Edition)...")
    print("🤖 Features:")
    print("   • CrewAI kickoff workflow with 4 specialized agents")
    print("   • Role-based access control (SECURED)")
    print("   • SQL database conversion (PostgreSQL, SQL Server, DB2)")
    print("   • Export results to Excel")
    print("   • Enhanced chart generation")
    print("   • File upload & processing (CSV, Excel, JSON)")
    print("   • Dynamic schema updates with FILE_DATA.db")
    print("   • Uploaded file management")
    print("   • Multi-database support")
    print("   • Professional SQL analysis pipeline")
    uvicorn.run(app, host="0.0.0.0", port=8000)
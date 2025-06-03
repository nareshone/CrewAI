"""
FastAPI Application for CrewAI SQL Analysis System
Run with: uvicorn fastapi_app:app --reload
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
from datetime import datetime
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Import the main CrewAI system
from main import CrewAISQLSystem, CrewAIApp

# Create FastAPI app
app = FastAPI(
    title="CrewAI SQL Analysis API",
    description="Multi-Agent AI System for SQL Analysis with Natural Language",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the CrewAI system
crew_app = None
crew_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize CrewAI system on startup"""
    global crew_app, crew_system
    print("🚀 Initializing CrewAI SQL Analysis System...")
    crew_app = CrewAIApp()
    crew_system = crew_app.system
    print("✅ CrewAI system initialized")

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    username: str = "admin"
    create_chart: bool = False
    chart_type: str = "bar"
    data_source: str = "database"  # database, files, or both
    feedback: Optional[str] = None

class QueryResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[Dict]] = None
    sql_query: Optional[str] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    chart_path: Optional[str] = None
    execution_id: Optional[str] = None

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    rows: Optional[int] = None
    columns: Optional[List[str]] = None

class ExportRequest(BaseModel):
    execution_id: str
    filename: Optional[str] = None

class DirectQueryRequest(BaseModel):
    sql_query: str
    username: str = "admin"

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CrewAI SQL Analysis System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, textarea, select {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            textarea {
                height: 100px;
                resize: vertical;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-right: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            .results {
                margin-top: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                border: 1px solid #e0e0e0;
            }
            .error {
                color: #d32f2f;
                padding: 10px;
                background-color: #ffebee;
                border-radius: 5px;
                margin-top: 10px;
            }
            .success {
                color: #388e3c;
                padding: 10px;
                background-color: #e8f5e9;
                border-radius: 5px;
                margin-top: 10px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #4CAF50;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .tabs {
                display: flex;
                margin-bottom: 20px;
                border-bottom: 2px solid #ddd;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-bottom: none;
                margin-right: 5px;
            }
            .tab.active {
                background-color: white;
                font-weight: bold;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .checkbox-group {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            .checkbox-group input {
                width: auto;
                margin-right: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 CrewAI SQL Analysis System</h1>
            <p style="text-align: center; color: #666;">
                Multi-Agent AI System with Natural Language to SQL
            </p>
            
            <div class="tabs">
                <div class="tab active" onclick="showTab('query')">Natural Language Query</div>
                <div class="tab" onclick="showTab('direct')">Direct SQL</div>
                <div class="tab" onclick="showTab('upload')">Upload Data</div>
                <div class="tab" onclick="showTab('stats')">Statistics</div>
            </div>
            
            <!-- Natural Language Query Tab -->
            <div id="query-tab" class="tab-content active">
                <form id="queryForm">
                    <div class="form-group">
                        <label for="query">Natural Language Query:</label>
                        <textarea id="query" name="query" placeholder="e.g., Show me average salary by department" required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="username">Username:</label>
                        <select id="username" name="username">
                            <option value="admin">Admin</option>
                            <option value="analyst">Analyst</option>
                            <option value="viewer">Viewer</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="dataSource">Data Source:</label>
                        <select id="dataSource" name="dataSource">
                            <option value="database">Database (Employees, Departments)</option>
                            <option value="files">Files (Products, Sales)</option>
                            <option value="both">Both</option>
                        </select>
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="createChart" name="createChart">
                        <label for="createChart">Create Visualization</label>
                    </div>
                    
                    <div class="form-group" id="chartTypeGroup" style="display: none;">
                        <label for="chartType">Chart Type:</label>
                        <select id="chartType" name="chartType">
                            <option value="bar">Bar Chart</option>
                            <option value="pie">Pie Chart</option>
                        </select>
                    </div>
                    
                    <button type="submit">🚀 Analyze Query</button>
                    <button type="button" onclick="clearResults()">🗑️ Clear</button>
                </form>
            </div>
            
            <!-- Direct SQL Tab -->
            <div id="direct-tab" class="tab-content">
                <form id="directForm">
                    <div class="form-group">
                        <label for="sqlQuery">SQL Query:</label>
                        <textarea id="sqlQuery" name="sqlQuery" placeholder="SELECT * FROM employees LIMIT 10" required></textarea>
                    </div>
                    
                    <button type="submit">▶️ Execute SQL</button>
                    <button type="button" onclick="clearResults()">🗑️ Clear</button>
                </form>
            </div>
            
            <!-- Upload Data Tab -->
            <div id="upload-tab" class="tab-content">
                <form id="uploadForm">
                    <div class="form-group">
                        <label for="file">Upload CSV or Excel File:</label>
                        <input type="file" id="file" name="file" accept=".csv,.xlsx,.xls" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="tableName">Table Name:</label>
                        <input type="text" id="tableName" name="tableName" placeholder="e.g., sales_data" required>
                    </div>
                    
                    <button type="submit">📤 Upload File</button>
                </form>
            </div>
            
            <!-- Statistics Tab -->
            <div id="stats-tab" class="tab-content">
                <button onclick="loadStats()">📊 Load Statistics</button>
                <div id="statsResults"></div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing your request...</p>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            // Show/hide chart type based on checkbox
            document.getElementById('createChart').addEventListener('change', function() {
                document.getElementById('chartTypeGroup').style.display = this.checked ? 'block' : 'none';
            });
            
            // Tab switching
            function showTab(tabName) {
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                event.target.classList.add('active');
                document.getElementById(tabName + '-tab').classList.add('active');
            }
            
            // Natural language query form
            document.getElementById('queryForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const requestData = {
                    query: formData.get('query'),
                    username: formData.get('username'),
                    create_chart: formData.get('createChart') === 'on',
                    chart_type: formData.get('chartType') || 'bar',
                    data_source: formData.get('dataSource')
                };
                
                showLoading(true);
                
                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(requestData)
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    displayError('Error: ' + error.message);
                } finally {
                    showLoading(false);
                }
            });
            
            // Direct SQL form
            document.getElementById('directForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const requestData = {
                    sql_query: formData.get('sqlQuery'),
                    username: 'admin'
                };
                
                showLoading(true);
                
                try {
                    const response = await fetch('/api/direct-query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(requestData)
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    displayError('Error: ' + error.message);
                } finally {
                    showLoading(false);
                }
            });
            
            // File upload form
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('file', document.getElementById('file').files[0]);
                formData.append('table_name', document.getElementById('tableName').value);
                formData.append('username', 'admin');
                
                showLoading(true);
                
                try {
                    const response = await fetch('/api/upload-file', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    displayUploadResult(result);
                } catch (error) {
                    displayError('Error: ' + error.message);
                } finally {
                    showLoading(false);
                }
            });
            
            // Display functions
            function showLoading(show) {
                document.getElementById('loading').style.display = show ? 'block' : 'none';
            }
            
            function displayResults(result) {
                const resultsDiv = document.getElementById('results');
                
                if (result.success) {
                    let html = '<div class="success">✅ ' + result.message + '</div>';
                    
                    if (result.sql_query) {
                        html += '<h3>SQL Query:</h3><pre>' + result.sql_query + '</pre>';
                    }
                    
                    if (result.data && result.data.length > 0) {
                        html += '<h3>Results (' + result.row_count + ' rows):</h3>';
                        html += '<div style="overflow-x: auto;">';
                        html += '<table>';
                        
                        // Header
                        html += '<tr>';
                        Object.keys(result.data[0]).forEach(col => {
                            html += '<th>' + col + '</th>';
                        });
                        html += '</tr>';
                        
                        // Data rows (limit to 20 for display)
                        result.data.slice(0, 20).forEach(row => {
                            html += '<tr>';
                            Object.values(row).forEach(val => {
                                html += '<td>' + (val !== null ? val : '') + '</td>';
                            });
                            html += '</tr>';
                        });
                        
                        html += '</table>';
                        html += '</div>';
                        
                        if (result.row_count > 20) {
                            html += '<p>... and ' + (result.row_count - 20) + ' more rows</p>';
                        }
                        
                        // Export button
                        if (result.execution_id) {
                            html += '<button onclick="exportResults(\'' + result.execution_id + '\')">💾 Export to Excel</button>';
                        }
                    }
                    
                    if (result.chart_path) {
                        html += '<h3>Visualization:</h3>';
                        html += '<img src="/api/chart/' + result.chart_path + '" style="max-width: 100%;">';
                    }
                    
                    resultsDiv.innerHTML = html;
                } else {
                    displayError(result.message);
                }
            }
            
            function displayUploadResult(result) {
                const resultsDiv = document.getElementById('results');
                
                if (result.success) {
                    let html = '<div class="success">✅ ' + result.message + '</div>';
                    html += '<p>Filename: ' + result.filename + '</p>';
                    if (result.rows) {
                        html += '<p>Rows: ' + result.rows + '</p>';
                    }
                    if (result.columns) {
                        html += '<p>Columns: ' + result.columns.join(', ') + '</p>';
                    }
                    resultsDiv.innerHTML = html;
                } else {
                    displayError(result.message);
                }
            }
            
            function displayError(message) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<div class="error">❌ ' + message + '</div>';
            }
            
            function clearResults() {
                document.getElementById('results').innerHTML = '';
            }
            
            async function exportResults(executionId) {
                try {
                    const response = await fetch('/api/export', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({execution_id: executionId})
                    });
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'query_results_' + new Date().toISOString().slice(0, 10) + '.xlsx';
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                        window.URL.revokeObjectURL(url);
                    } else {
                        const error = await response.json();
                        displayError(error.detail);
                    }
                } catch (error) {
                    displayError('Export failed: ' + error.message);
                }
            }
            
            async function loadStats() {
                showLoading(true);
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    
                    let html = '<h3>System Statistics</h3>';
                    html += '<p>🧠 LLM Provider: ' + stats.llm_provider + '</p>';
                    html += '<p>📈 Total Queries: ' + stats.total_queries + '</p>';
                    html += '<p>✅ Successful Queries: ' + stats.successful_queries + '</p>';
                    html += '<p>📁 Registered Files: ' + stats.registered_files + '</p>';
                    
                    document.getElementById('statsResults').innerHTML = html;
                } catch (error) {
                    document.getElementById('statsResults').innerHTML = 
                        '<div class="error">Failed to load statistics</div>';
                } finally {
                    showLoading(false);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """Execute a natural language query using CrewAI"""
    try:
        # Process the request
        result = crew_system.process_request(
            user_request=request.query,
            username=request.username,
            create_chart=request.create_chart,
            chart_type=request.chart_type,
            data_source=request.data_source,
            feedback=request.feedback
        )
        
        if result.get('success'):
            execution_data = result.get('execution_data', {})
            
            # Generate execution ID for export
            execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store execution data for later export
            if execution_data:
                crew_system.last_execution_result = execution_data
                
                # Also store in a temporary cache for the web UI
                if not hasattr(app, 'execution_cache'):
                    app.execution_cache = {}
                app.execution_cache[execution_id] = execution_data
            
            return QueryResponse(
                success=True,
                message="Query executed successfully",
                data=execution_data.get('data', [])[:100],  # Limit to 100 rows for web display
                sql_query=execution_data.get('sql_query'),
                row_count=execution_data.get('row_count', 0),
                columns=execution_data.get('columns', []),
                execution_id=execution_id
            )
        else:
            return QueryResponse(
                success=False,
                message=result.get('error', 'Query execution failed')
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/direct-query", response_model=QueryResponse)
async def execute_direct_query(request: DirectQueryRequest):
    """Execute a direct SQL query"""
    try:
        result = crew_system.direct_query(request.sql_query)
        
        if result.get('success'):
            return QueryResponse(
                success=True,
                message="Query executed successfully",
                data=result.get('data', []),
                sql_query=request.sql_query,
                row_count=result.get('row_count', 0),
                columns=result.get('columns', [])
            )
        else:
            return QueryResponse(
                success=False,
                message=result.get('error', 'Query execution failed')
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-file", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    table_name: str = Form(...),
    username: str = Form("admin")
):
    """Upload a CSV or Excel file"""
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Register the file
        result = crew_system.register_file(table_name, temp_path, username)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        if result.get('success'):
            file_info = crew_system.file_manager.data_sources.get(table_name, {})
            return FileUploadResponse(
                success=True,
                message=f"File '{file.filename}' uploaded successfully as table '{table_name}'",
                filename=file.filename,
                rows=file_info.get('rows'),
                columns=file_info.get('columns', [])
            )
        else:
            return FileUploadResponse(
                success=False,
                message=result.get('message', 'File upload failed'),
                filename=file.filename
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export")
async def export_results(request: ExportRequest):
    """Export query results to Excel"""
    try:
        # Generate filename
        filename = request.filename or f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Check if we have execution data to export
        if crew_system.last_execution_result and 'data' in crew_system.last_execution_result:
            # Use the stored execution result
            exec_data = crew_system.last_execution_result
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Query results
                df = pd.DataFrame(exec_data['data'])
                df.to_excel(writer, sheet_name='Query Results', index=False)
                
                # Metadata
                metadata = pd.DataFrame({
                    'Generated By': ['CrewAI SQL Analysis System'],
                    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'SQL Query': [exec_data.get('sql_query', 'N/A')],
                    'Total Rows': [exec_data.get('row_count', len(df))],
                    'Columns': [', '.join(exec_data.get('columns', df.columns.tolist()))],
                    'LLM Provider': [type(crew_system.llm).__name__ if crew_system.llm else "No LLM"]
                })
                metadata.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Summary statistics for numeric columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    summary = df[numeric_cols].describe()
                    summary.to_excel(writer, sheet_name='Summary Statistics')
            
            # Return the file
            return FileResponse(
                path=filename,
                filename=filename,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            # Try to export using the standard method
            export_result = crew_system.export_results(None, filename)
            
            if "exported to:" in export_result and os.path.exists(filename):
                return FileResponse(
                    path=filename,
                    filename=filename,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                raise HTTPException(status_code=400, detail="No data available to export")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = crew_system.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_available": crew_system.llm is not None if crew_system else False
    }

# Static file serving for charts
@app.get("/api/chart/{filename}")
async def get_chart(filename: str):
    """Serve generated chart images"""
    chart_path = os.path.join(".", filename)
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Chart not found")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    print("📍 Access the UI at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
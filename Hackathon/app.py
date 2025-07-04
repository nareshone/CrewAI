import streamlit as st
import pandas as pd
import os
import sqlite3
import matplotlib.pyplot as plt
from main import CrewAIApp, UserRole
import time
from datetime import datetime

# ✅ This must come first
st.set_page_config(
    page_title="🧠 Enhanced CrewAI SQL Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Header Card */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
    }
    
    /* Configuration Cards */
    .config-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e1e5e9;
    }
    
    .config-card h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    /* Role Badges */
    .role-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .admin-badge { 
        background: linear-gradient(135deg, #e74c3c, #c0392b); 
        color: white; 
        box-shadow: 0 2px 10px rgba(231, 76, 60, 0.3);
    }
    .analyst-badge { 
        background: linear-gradient(135deg, #f39c12, #e67e22); 
        color: white; 
        box-shadow: 0 2px 10px rgba(243, 156, 18, 0.3);
    }
    .viewer-badge { 
        background: linear-gradient(135deg, #27ae60, #229954); 
        color: white; 
        box-shadow: 0 2px 10px rgba(39, 174, 96, 0.3);
    }
    
    /* Status Indicators */
    .llm-status {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .database-info {
        background: linear-gradient(135deg, #cce5ff, #b3d9ff);
        border: 1px solid #80bdff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.95rem;
        box-shadow: 0 2px 10px rgba(0,123,255,0.1);
    }
    
    /* Content Cards */
    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    .content-card h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1.5rem;
        font-size: 1.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Query Interface Card */
    .query-interface {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Primary Button Override */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    
    div[data-testid="stButton"] button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
    }
    
    /* Example Buttons */
    .example-btn {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        font-weight: 500;
        width: 100%;
        text-align: left;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(108, 117, 125, 0.2);
    }
    
    .example-btn:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3);
    }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Results Container */
    .results-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    /* SQL Code Block */
    .stCodeBlock {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Feedback Box */
    .feedback-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1);
    }
    
    .feedback-box h4 {
        color: #1565c0;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Iteration Badge */
    .iteration-badge {
        background: linear-gradient(135deg, #17a2b8, #138496);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(23, 162, 184, 0.3);
    }
    
    /* Statistics Cards */
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border: 1px solid #e8eaed;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    /* Text Area and Inputs */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* SelectBox Styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
    }
    
    /* Multiselect Styling */
    .stMultiSelect > div > div {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
    }
    
    /* Alert Styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 -5px 20px rgba(0,0,0,0.1);
    }
    
    /* Sidebar Headers */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Custom spacing */
    .section-divider {
        height: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 2px;
        margin: 2rem 0;
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize app with dynamic LLM selection
@st.cache_resource
def load_system(_llm_provider, _selected_databases):
    return CrewAIApp(llm_provider=_llm_provider, selected_databases=_selected_databases)

# Function to get available databases
@st.cache_data
def get_available_databases():
    """Get list of available database files"""
    db_folder = "."
    available_dbs = []
    
    # Look for .db files
    for file in os.listdir(db_folder):
        if file.endswith(".db"):
            db_path = os.path.join(db_folder, file)
            if os.path.exists(db_path):
                available_dbs.append(file)
    
    # Add sample database if it doesn't exist
    if "sample.db" not in available_dbs:
        available_dbs.append("sample.db")
    
    return sorted(available_dbs)

# Function to get tables from a database
@st.cache_data
def get_database_tables(db_name):
    """Get tables from a specific database"""
    try:
        if not os.path.exists(db_name):
            return []
        
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception as e:
        st.error(f"Error reading database {db_name}: {e}")
        return []

# Function to get available LLM providers
def get_available_llm_providers():
    """Get list of available LLM providers based on API keys"""
    providers = {
        "Auto (Best Available)": "auto",
        "Groq (Fast)": "groq", 
        "OpenAI (GPT-3.5)": "openai",
        "Anthropic (Claude)": "anthropic",
        "Ollama (Local)": "ollama",
        "Mock LLM (No API Key)": "mock"
    }
    
    # Check which providers have API keys
    available = {}
    for name, key in providers.items():
        if key == "auto" or key == "mock":
            available[name] = key
        elif key == "groq" and os.getenv('GROQ_API_KEY'):
            available[f"{name} ✅"] = key
        elif key == "openai" and os.getenv('OPENAI_API_KEY'):
            available[f"{name} ✅"] = key
        elif key == "anthropic" and os.getenv('ANTHROPIC_API_KEY'):
            available[f"{name} ✅"] = key
        elif key == "ollama":
            # Check if Ollama is running
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    available[f"{name} ✅"] = key
                else:
                    available[f"{name} ⚠️"] = key
            except:
                available[f"{name} ❌"] = key
        else:
            available[f"{name} ❌"] = key
    
    return available

# Initialize session state
def init_session_state():
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'feedback_mode' not in st.session_state:
        st.session_state.feedback_mode = False
    if 'iteration_count' not in st.session_state:
        st.session_state.iteration_count = 0
    if 'query_history' not in st.session_state:
        st.session_state.query_history = {}
    if 'selected_llm' not in st.session_state:
        st.session_state.selected_llm = "auto"
    if 'selected_databases' not in st.session_state:
        st.session_state.selected_databases = ["All"]
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False

init_session_state()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧠 Enhanced CrewAI SQL Assistant</h1>
    <p>Intelligent Multi-Agent Data Analysis with Role-Based Access & Human-in-the-Loop Feedback</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    # System Configuration Header
    st.markdown("""
    <div class="sidebar-header">
        ⚙️ System Configuration
    </div>
    """, unsafe_allow_html=True)
    
    # LLM Provider Selection Card
    st.markdown("""
    <div class="config-card">
        <h3>🧠 LLM Provider</h3>
    </div>
    """, unsafe_allow_html=True)
    
    available_llms = get_available_llm_providers()
    
    selected_llm_display = st.selectbox(
        "Choose AI Model Provider",
        options=list(available_llms.keys()),
        index=0,
        help="Select the AI model provider for intelligent query generation",
        label_visibility="collapsed"
    )
    
    selected_llm = available_llms[selected_llm_display]
    
    # Show LLM status with better styling
    if "✅" in selected_llm_display:
        st.markdown('<div class="llm-status">✅ Provider Ready & Available</div>', unsafe_allow_html=True)
    elif "⚠️" in selected_llm_display:
        st.markdown('<div class="llm-status" style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); border-color: #ffc107;">⚠️ Provider Partially Available</div>', unsafe_allow_html=True)
    elif "❌" in selected_llm_display:
        st.markdown('<div class="llm-status" style="background: linear-gradient(135deg, #f8d7da, #f5c6cb); border-color: #dc3545;">❌ Provider Not Available</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="llm-status" style="background: linear-gradient(135deg, #d1ecf1, #bee5eb); border-color: #17a2b8;">🔧 Auto-Detection Mode</div>', unsafe_allow_html=True)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Database Selection Card
    st.markdown("""
    <div class="config-card">
        <h3>🗄️ Database Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    available_dbs = get_available_databases()
    db_options = ["All"] + available_dbs
    
    selected_dbs = st.multiselect(
        "Select Target Databases",
        options=db_options,
        default=["All"],
        help="Choose specific databases or 'All' for cross-database analysis",
        label_visibility="collapsed"
    )
    
    # If "All" is selected, use all databases
    if "All" in selected_dbs:
        selected_databases = available_dbs
        st.markdown("""
        <div class="database-info">
            📊 <strong>All Databases Selected</strong><br>
            Cross-database queries enabled
        </div>
        """, unsafe_allow_html=True)
    else:
        selected_databases = [db for db in selected_dbs if db != "All"]
    
    # Show database information in cards
    if selected_databases:
        st.markdown("**📋 Database Overview:**")
        total_tables = 0
        for db in selected_databases:
            tables = get_database_tables(db)
            total_tables += len(tables)
            
            st.markdown(f"""
            <div class="stat-card">
                <strong>{db}</strong><br>
                📊 {len(tables)} tables available
            </div>
            """, unsafe_allow_html=True)
            
            if len(tables) > 0 and len(selected_databases) == 1:  # Show details for single DB
                with st.expander(f"📋 Tables in {db}", expanded=False):
                    for table in tables:
                        st.markdown(f"• {table}")
        
        st.markdown(f"""
        <div class="database-info">
            <strong>Total Tables Available: {total_tables}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # User Configuration Card
    st.markdown("""
    <div class="sidebar-header">
        👤 User Profile
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="config-card">
        <h3>🔐 Role & Permissions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    username = st.selectbox(
        "Select your user role", 
        options=["admin", "analyst", "viewer"], 
        index=0,
        help="Each role has different data access levels and permissions",
        label_visibility="collapsed"
    )
    
    # Initialize or update app when configuration changes
    if (st.session_state.selected_llm != selected_llm or 
        st.session_state.selected_databases != selected_databases or
        not st.session_state.app_initialized):
        
        st.session_state.selected_llm = selected_llm
        st.session_state.selected_databases = selected_databases
        
        with st.spinner("🔄 Initializing system with selected configuration..."):
            try:
                # Clear cache and reinitialize
                load_system.clear()
                st.session_state.app = load_system(selected_llm, selected_databases)
                st.session_state.app_initialized = True
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ System initialization failed: {e}")
                st.session_state.app_initialized = False

# Only proceed if app is initialized
if st.session_state.app_initialized and 'app' in st.session_state:
    app = st.session_state.app
    
    # Get user role and permissions
    user_role = app.system.role_manager.get_user_role(username)
    accessible_tables = app.system.get_user_accessible_tables(username, selected_databases)
    
    with st.sidebar:
        # Role display with enhanced styling
        if user_role:
            role_class = f"{user_role.value}-badge"
            st.markdown(f"""
            <div class="role-badge {role_class}">
                {user_role.value.upper()} USER
            </div>
            """, unsafe_allow_html=True)
            
            # Show accessible tables in a clean format
            st.markdown("**🔓 Your Access Level:**")
            if accessible_tables:
                # Limit display to prevent overcrowding
                display_tables = accessible_tables[:8]
                for table in display_tables:
                    st.markdown(f"• {table}")
                if len(accessible_tables) > 8:
                    st.markdown(f"• *... and {len(accessible_tables) - 8} more tables*")
            else:
                st.markdown("• *No accessible tables*")
        
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # System Statistics Card
        st.markdown("""
        <div class="sidebar-header">
            📊 System Statistics
        </div>
        """, unsafe_allow_html=True)
        
        stats = app.system.get_stats(username)
        
        # Create cleaner metrics display
        st.markdown(f"""
        <div class="stat-card">
            <strong>📈 Query Performance</strong><br>
            Total: {stats['total_queries']} | Success: {(stats['successful_queries']/max(stats['total_queries'],1)*100):.0f}%
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card">
            <strong>🔄 System Activity</strong><br>
            Feedback Sessions: {stats['feedback_sessions']}<br>
            Registered Files: {stats['registered_files']}
        </div>
        """, unsafe_allow_html=True)
        
        if 'user_stats' in stats:
            st.markdown(f"""
            <div class="stat-card">
                <strong>👤 Your Activity</strong><br>
                Queries: {stats['user_stats']['total']}<br>
                Success Rate: {stats['user_stats']['success_rate']}<br>
                Feedback Given: {stats['user_stats']['feedback_given']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # File Upload Section (role-based)
        if app.system.role_manager.check_permission(username, "register_files"):
            st.markdown("""
            <div class="config-card">
                <h3>📁 File Upload</h3>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload Data Files", 
                type=['csv', 'xlsx', 'xls', 'json'],
                help="Upload files to query alongside database tables",
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                file_name_input = st.text_input(
                    "Registration name", 
                    value=uploaded_file.name.split('.')[0],
                    help="Name to use when querying this file"
                )
                
                if st.button("📥 Register File", type="primary", use_container_width=True):
                    # Save uploaded file
                    upload_dir = "uploads"
                    os.makedirs(upload_dir, exist_ok=True)
                    file_path = os.path.join(upload_dir, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    # Register with system
                    result = app.system.register_file(file_name_input, file_path, username)
                    
                    if result["success"]:
                        st.success(f"✅ File '{file_name_input}' registered successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to register: {result.get('message', 'Unknown error')}")
        else:
            st.markdown("""
            <div class="llm-status" style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); border-color: #ffc107;">
                🔒 File upload requires analyst or admin role
            </div>
            """, unsafe_allow_html=True)

    # Main content area with improved layout
    col1, col2 = st.columns([2.5, 1.5], gap="large")

    with col1:
        # Query Interface Card
        st.markdown("""
        <div class="query-interface">
            <h2>💬 Query Interface</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Current configuration display
        st.markdown(f"""
        <div class="database-info">
            🧠 <strong>AI Model:</strong> {selected_llm_display}<br>
            🗄️ <strong>Target Databases:</strong> {len(selected_databases)} database(s)<br>
            📊 <strong>Your Access:</strong> {len(accessible_tables)} table(s) available
        </div>
        """, unsafe_allow_html=True)
        
        # Query input with better styling
        user_query = st.text_area(
            "🔍 **Enter your data query in natural language**",
            value="Show average salary by department" if user_role != UserRole.VIEWER else "Show departments and their locations",
            height=120,
            help="Ask questions about your data in plain English. The AI will generate appropriate SQL queries based on your role and selected databases.",
            placeholder="e.g., 'Show me the top 10 products by sales' or 'What are the average salaries by department?'"
        )
        
        # Query configuration in organized layout
        st.markdown("**⚙️ Query Configuration**")
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            if len(selected_databases) == 1:
                data_source = selected_databases[0]
                st.info(f"🗄️ **Target:** {data_source}")
            else:
                data_source = st.selectbox(
                    "🗄️ Primary database", 
                    options=selected_databases + ["Auto-detect"], 
                    index=len(selected_databases),  # Default to Auto-detect
                    help="Primary database for the query"
                )
        
        with col_config2:
            create_chart = st.checkbox(
                "📊 Generate visualization", 
                value=True,
                help="Create a professional chart from query results"
            )
        
        with col_config3:
            chart_type = st.selectbox(
                "📈 Chart style", 
                options=["bar", "pie"], 
                index=0,
                disabled=not create_chart,
                help="Choose the type of visualization"
            )

    with col2:
        # Query Examples Card
        st.markdown("""
        <div class="content-card">
            <h2>🎯 Smart Query Examples</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Database-specific and role-based example queries
        examples = []
        
        if len(selected_databases) == 1:
            # Single database examples
            db_name = selected_databases[0]
            tables = get_database_tables(db_name)
            
            if user_role == UserRole.ADMIN:
                if "employees" in tables:
                    examples.extend([
                        "Show all employees with salaries > 70000",
                        "Average salary by department",
                        "Employee count by hire year"
                    ])
                if "departments" in tables:
                    examples.append("Departments with highest budgets")
                if "products" in tables or "sales" in tables:
                    examples.extend([
                        "Top selling products this month",
                        "Sales trends by product"
                    ])
            elif user_role == UserRole.ANALYST:
                if "employees" in tables:
                    examples.extend([
                        "Count employees by department",
                        "Department employee counts"
                    ])
                if "departments" in tables:
                    examples.append("Show all departments and locations")
                if "products" in tables:
                    examples.extend([
                        "Product categories and stock levels",
                        "Products by category"
                    ])
            else:  # VIEWER
                if "departments" in tables:
                    examples.append("Show department names and locations")
                if "products" in tables:
                    examples.extend([
                        "List product categories",
                        "Show product names by category"
                    ])
        else:
            # Multi-database examples
            if user_role == UserRole.ADMIN:
                examples = [
                    "Show employee count across all databases",
                    "Compare department budgets across databases",
                    "Find highest paid employees in any database",
                    "List all products from all sources",
                    "Cross-database sales analysis"
                ]
            elif user_role == UserRole.ANALYST:
                examples = [
                    "Count departments across all databases",
                    "Show product categories from all sources",
                    "Compare employee counts by database",
                    "List all available product categories"
                ]
            else:  # VIEWER
                examples = [
                    "Show all department locations",
                    "List product categories from all sources",
                    "Count departments across databases",
                    "Show available product types"
                ]
        
        # Display examples with better styling
        for i, example in enumerate(examples):
            if st.button(f"💡 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()
        
        # Use example query if selected
        if 'example_query' in st.session_state:
            user_query = st.session_state.example_query
            del st.session_state.example_query

    # Action buttons with improved spacing and styling
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    st.markdown("**🚀 Execute Analysis**")
    col_btn1, col_btn2, col_btn3 = st.columns(3, gap="medium")

    with col_btn1:
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    with col_btn2:
        if st.button("💾 Export Results", use_container_width=True):
            filename = f"results_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            result = st.session_state.get("current_result", {})
            data_to_export = None

            # First try: execution data
            if result.get("execution_data") and result["execution_data"].get("data"):
                data_to_export = result["execution_data"]["data"]
                st.info(f"📊 Exporting {len(data_to_export)} rows from execution data")

            # Second try: system-level cache
            elif hasattr(app.system, "last_execution_result") and app.system.last_execution_result:
                last_result = app.system.last_execution_result
                if last_result.get("data"):
                    data_to_export = last_result["data"]
                    st.info(f"📊 Exporting {len(data_to_export)} rows from cached result")

            # Final fallback: export generic message
            if not data_to_export:
                data_to_export = [{
                    "message": "No query data available to export",
                    "user": username,
                    "timestamp": datetime.now().isoformat(),
                    "llm_provider": selected_llm_display,
                    "databases": ", ".join(selected_databases)
                }]
                st.warning("⚠️ No query result found — exporting placeholder metadata.")

            export_result = app.system.export_results(data_to_export, filename)
            st.success(f"✅ {export_result}")

    with col_btn3:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.current_result = None
            st.session_state.feedback_mode = False
            st.session_state.iteration_count = 0
            st.rerun()

    # Execute analysis
    if run_analysis and user_query.strip():
        st.session_state.feedback_mode = False
        st.session_state.iteration_count = 1
        
        with st.spinner("🤖 Multi-agent analysis in progress..."):
            result = app.system.process_request_with_feedback_loop(
                user_request=user_query,
                username=username,
                create_chart=create_chart,
                chart_type=chart_type,
                data_source=data_source if len(selected_databases) == 1 else "auto",
                selected_databases=selected_databases
            )
            
            st.session_state.current_result = result
            
            # Add to user-specific query history
            if username not in st.session_state.query_history:
                st.session_state.query_history[username] = []
            
            st.session_state.query_history[username].append({
                'timestamp': datetime.now(),
                'query': user_query,
                'user': username,
                'success': result.get('success', False),
                'iteration': 1,
                'llm_provider': selected_llm_display,
                'databases': selected_databases
            })

    # Display results with enhanced styling
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        # Results container
        st.markdown("""
        <div class="results-container">
        """, unsafe_allow_html=True)
        
        # Results header with iteration info
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.markdown("## 📋 Analysis Results")
        with col_header2:
            if st.session_state.iteration_count > 0:
                st.markdown(f"""
                <span class="iteration-badge">Iteration {st.session_state.iteration_count}</span>
                """, unsafe_allow_html=True)
        
        if result.get("success"):
            # Success metrics in cards
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            
            with col_metric1:
                st.metric("Status", "✅ Success", delta="Completed")
            with col_metric2:
                st.metric("User Role", result.get("user_role", "Unknown"))
            with col_metric3:
                st.metric("Agents Used", result.get("agents_used", 0))
            with col_metric4:
                st.metric("AI Provider", selected_llm_display.split()[0])
            
            # SQL Query display with enhanced styling
            if result.get("execution_data"):
                exec_data = result["execution_data"]
                
                st.markdown("### 📝 Generated SQL Query")
                sql_query = exec_data.get("sql_query", "No SQL returned")
                st.code(sql_query, language="sql")
                
                # Results data
                st.markdown("### 📊 Query Results")
                
                if "data" in exec_data and exec_data["data"]:
                    df = pd.DataFrame(exec_data["data"])
                    
                    # Check if this is an error message result
                    if len(df) == 1 and any(col for col in df.columns if 'error' in col.lower() or 'message' in col.lower()):
                        # This is likely an error or info message
                        first_row = df.iloc[0]
                        
                        for col, value in first_row.items():
                            if 'error' in col.lower():
                                st.error(f"❌ {value}")
                            elif 'info' in col.lower():
                                st.info(f"ℹ️ {value}")
                            elif 'access_denied' in col.lower():
                                st.warning(f"🔒 {value}")
                            elif 'restriction' in col.lower():
                                st.warning(f"⚠️ {value}")
                            else:
                                st.info(f"📋 {col}: {value}")
                        
                        # Still show the dataframe but with a note
                        st.caption("*Raw result data:*")
                        st.dataframe(df, use_container_width=True)
                        
                    else:
                        # Normal data results
                        # Show row count and basic info
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("Rows Returned", len(df), delta="Records")
                        with col_info2:
                            st.metric("Columns", len(df.columns), delta="Fields")
                        
                        # Display data table with enhanced styling
                        st.dataframe(df, use_container_width=True, height=400)
                        
                        # Chart display
                        if create_chart and len(df) > 0:
                            chart_prefix = f"chart_{chart_type}"
                            chart_files = [f for f in os.listdir('.') if f.startswith(chart_prefix) and f.endswith('.png')]
                            
                            if chart_files:
                                # Sort by modified time (newest first)
                                chart_files = sorted(chart_files, key=lambda x: os.path.getmtime(x), reverse=True)
                                latest_chart = chart_files[0]
                                
                                st.markdown("### 📈 Generated Visualization")
                                st.image(latest_chart, caption=f"📊 {chart_type.title()} Chart - Generated by CrewAI", use_container_width=True)
                            else:
                                st.warning("📈 Chart generation was requested but no chart file was found.")
                else:
                    st.warning("No data returned from query")
            else:
                st.warning("No execution data available")
            
            # Feedback section with enhanced styling
            if not st.session_state.feedback_mode:
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                
                col_feedback1, col_feedback2 = st.columns([3, 1])
                
                with col_feedback1:
                    st.markdown("### 🔄 Feedback & Iteration")
                    st.markdown("*Not satisfied with the results? Provide specific feedback to improve the query!*")
                
                with col_feedback2:
                    if st.button("💬 Provide Feedback", type="secondary", use_container_width=True):
                        st.session_state.feedback_mode = True
                        st.rerun()
            
            # Feedback input form with enhanced styling
            if st.session_state.feedback_mode:
                st.markdown("""
                <div class="feedback-box">
                    <h4>💬 Provide Specific Feedback</h4>
                    <p>Tell the AI exactly how to improve the query. Be specific about what you want changed or added.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback examples
                with st.expander("💡 Feedback Examples", expanded=False):
                    st.markdown("""
                    **Good feedback examples:**
                    - "Add ORDER BY salary DESC to sort by highest salary first"
                    - "Remove the LIMIT clause to show all results"
                    - "Group by department instead of individual employees"
                    - "Show only Engineering department employees"
                    - "Include hire_date column in the results"
                    - "Change this to show percentages instead of counts"
                    """)
                
                feedback_text = st.text_area(
                    "Your feedback:",
                    placeholder="e.g., 'Add ORDER BY salary DESC', 'Show only Engineering department', 'Include hire_date column'",
                    height=100
                )
                
                col_feedback_btn1, col_feedback_btn2 = st.columns(2)
                
                with col_feedback_btn1:
                    if st.button("🔄 Apply Feedback", type="primary", use_container_width=True):
                        if feedback_text.strip():
                            st.session_state.iteration_count += 1
                            
                            with st.spinner(f"🔄 Applying feedback (Iteration {st.session_state.iteration_count})..."):
                                new_result = app.system.apply_feedback_and_retry(
                                    previous_result=result,
                                    feedback=feedback_text,
                                    username=username,
                                    create_chart=create_chart,
                                    chart_type=chart_type,
                                    data_source=data_source if len(selected_databases) == 1 else "auto",
                                    selected_databases=selected_databases
                                )
                                
                                st.session_state.current_result = new_result
                                st.session_state.feedback_mode = False
                                
                                # Add to user-specific history
                                if username not in st.session_state.query_history:
                                    st.session_state.query_history[username] = []
                                
                                st.session_state.query_history[username].append({
                                    'timestamp': datetime.now(),
                                    'query': f"FEEDBACK: {feedback_text}",
                                    'user': username,
                                    'success': new_result.get('success', False),
                                    'iteration': st.session_state.iteration_count,
                                    'llm_provider': selected_llm_display,
                                    'databases': selected_databases
                                })
                            
                            st.rerun()
                        else:
                            st.warning("Please provide feedback before applying.")
                
                with col_feedback_btn2:
                    if st.button("❌ Cancel Feedback", use_container_width=True):
                        st.session_state.feedback_mode = False
                        st.rerun()
        
        else:
            # Error display with enhanced styling
            st.error("❌ Query Analysis Failed")
            
            error_msg = result.get("error", "No error information available")
            st.markdown(f"""
            <div class="feedback-box" style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); border-left-color: #f44336;">
                <h4>⚠️ Error Details</h4>
                <p>{error_msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if result.get("user_role"):
                st.info(f"🔒 **Note:** You are logged in as a **{result['user_role']}** user with specific access restrictions.")
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close results container

    # Query history sidebar (user-specific) with enhanced design
    if username in st.session_state.query_history and st.session_state.query_history[username]:
        with st.sidebar:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("""
            <div class="sidebar-header">
                📝 Your Query History
            </div>
            """, unsafe_allow_html=True)
            
            user_history = st.session_state.query_history[username]
            
            # Show last 5 queries for current user
            for i, entry in enumerate(reversed(user_history[-5:])):
                status_icon = "✅" if entry['success'] else "❌"
                iteration_text = f"(Iter {entry['iteration']})" if entry['iteration'] > 1 else ""
                llm_info = entry.get('llm_provider', 'Unknown')[:10]
                
                with st.expander(f"{status_icon} {entry['timestamp'].strftime('%H:%M')} {iteration_text}"):
                    st.markdown(f"**Query:** {entry['query'][:80]}...")
                    st.markdown(f"**Status:** {'Success' if entry['success'] else 'Failed'}")
                    st.markdown(f"**Iteration:** {entry['iteration']}")
                    st.markdown(f"**AI Model:** {llm_info}")
                    if 'databases' in entry:
                        st.markdown(f"**Databases:** {len(entry['databases'])}")

else:
    # Show initialization message with better styling
    st.markdown("""
    <div class="content-card">
        <h2>🔧 System Configuration Required</h2>
        <p>Please configure the system settings in the sidebar to continue.</p>
        <p>Select an LLM provider and database(s) to initialize the CrewAI system.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer with enhanced professional design
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

if st.session_state.app_initialized and 'app' in st.session_state:
    help_text = {
        UserRole.ADMIN: "🔓 **Admin Access:** Full access to all data including sensitive information like salaries and personal details.",
        UserRole.ANALYST: "📊 **Analyst Access:** Access to most data with some personal details restricted for privacy.",
        UserRole.VIEWER: "👀 **Viewer Access:** Read-only access to non-sensitive, aggregated data only."
    }
    
    if user_role in help_text:
        st.markdown(f"""
        <div class="database-info">
            {help_text[user_role]}
        </div>
        """, unsafe_allow_html=True)

# Professional footer
st.markdown(f"""
<div class="footer">
    <h3>🧠 Enhanced CrewAI SQL Assistant</h3>
    <p><strong>Multi-Agent Intelligence with Human-in-the-Loop Feedback</strong></p>
    <p style="font-size: 0.9rem; opacity: 0.8;">
        Role-Based Access Control • Memory-Driven Learning • Multi-Source Data Queries • Configurable LLM
    </p>
    <p style="font-size: 0.8rem; opacity: 0.7;">
        Current: {selected_llm_display if 'selected_llm_display' in locals() else 'Not configured'} | 
        Databases: {len(selected_databases) if 'selected_databases' in locals() else 0} | 
        Powered by CrewAI
    </p>
</div>
""", unsafe_allow_html=True)
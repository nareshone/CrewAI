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

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .role-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .admin-badge { background-color: #dc3545; color: white; }
    .analyst-badge { background-color: #ffc107; color: black; }
    .viewer-badge { background-color: #28a745; color: white; }
    .feedback-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .iteration-badge {
        background-color: #17a2b8;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 10px;
        font-size: 0.75rem;
    }
    .llm-status {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .database-info {
        background-color: #f0f8ff;
        border: 1px solid #007bff;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9em;
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
    st.header("⚙️ System Configuration")
    
    # LLM Provider Selection
    st.subheader("🧠 LLM Provider")
    available_llms = get_available_llm_providers()
    
    selected_llm_display = st.selectbox(
        "Choose LLM Provider",
        options=list(available_llms.keys()),
        index=0,
        help="Select the AI model provider for query generation"
    )
    
    selected_llm = available_llms[selected_llm_display]
    
    # Show LLM status
    if "✅" in selected_llm_display:
        st.markdown('<div class="llm-status">✅ Provider Available</div>', unsafe_allow_html=True)
    elif "⚠️" in selected_llm_display:
        st.markdown('<div class="llm-status">⚠️ Provider Partially Available</div>', unsafe_allow_html=True)
    elif "❌" in selected_llm_display:
        st.markdown('<div class="llm-status">❌ Provider Not Available</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="llm-status">🔧 Auto-Detection</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Database Selection
    st.subheader("🗄️ Database Selection")
    available_dbs = get_available_databases()
    
    # Add "All" option
    db_options = ["All"] + available_dbs
    
    selected_dbs = st.multiselect(
        "Select Databases",
        options=db_options,
        default=["All"],
        help="Choose which databases to query. 'All' enables cross-database queries."
    )
    
    # If "All" is selected, use all databases
    if "All" in selected_dbs:
        selected_databases = available_dbs
        st.info("📊 All databases selected - Cross-database queries enabled")
    else:
        selected_databases = [db for db in selected_dbs if db != "All"]
    
    # Show database information
    if selected_databases:
        st.markdown("**Selected Database(s):**")
        total_tables = 0
        for db in selected_databases:
            tables = get_database_tables(db)
            total_tables += len(tables)
            st.markdown(f"• **{db}**: {len(tables)} tables")
            if len(tables) > 0 and len(selected_databases) == 1:  # Show details for single DB
                for table in tables[:5]:  # Show first 5 tables
                    st.markdown(f"  - {table}")
                if len(tables) > 5:
                    st.markdown(f"  - ... and {len(tables) - 5} more")
        
        st.markdown(f"**Total Tables Available: {total_tables}**")
    
    st.divider()
    
    # User Configuration
    st.header("👤 User Configuration")
    
    username = st.selectbox(
        "Select your role", 
        options=["admin", "analyst", "viewer"], 
        index=0,
        help="Different roles have different data access levels"
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
        # Role display with styling
        if user_role:
            role_class = f"{user_role.value}-badge"
            st.markdown(f"""
            <div class="role-badge {role_class}">
                {user_role.value.upper()} USER
            </div>
            """, unsafe_allow_html=True)
            
            # Show accessible tables for current user
            st.markdown("**Your Accessible Tables:**")
            if accessible_tables:
                for table in accessible_tables[:10]:  # Limit display
                    st.markdown(f"• {table}")
                if len(accessible_tables) > 10:
                    st.markdown(f"• ... and {len(accessible_tables) - 10} more")
            else:
                st.markdown("• No accessible tables")
        
        st.divider()
        
        # System stats
        st.header("📊 System Statistics")
        stats = app.system.get_stats(username)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", stats['total_queries'])
            st.metric("Success Rate", f"{(stats['successful_queries']/max(stats['total_queries'],1)*100):.1f}%")
        
        with col2:
            st.metric("Feedback Sessions", stats['feedback_sessions'])
            st.metric("Registered Files", stats['registered_files'])
        
        if 'user_stats' in stats:
            st.markdown("**Your Statistics:**")
            st.markdown(f"• Queries: {stats['user_stats']['total']}")
            st.markdown(f"• Success Rate: {stats['user_stats']['success_rate']}")
            st.markdown(f"• Feedback Given: {stats['user_stats']['feedback_given']}")
        
        st.divider()
        
        # File upload section (role-based)
        if app.system.role_manager.check_permission(username, "register_files"):
            st.header("📁 File Upload")
            uploaded_file = st.file_uploader(
                "Upload CSV, Excel, or JSON file", 
                type=['csv', 'xlsx', 'xls', 'json'],
                help="Upload data files to query alongside database tables"
            )
            
            if uploaded_file:
                file_name_input = st.text_input(
                    "File registration name", 
                    value=uploaded_file.name.split('.')[0],
                    help="Name to use when querying this file"
                )
                
                if st.button("📥 Register File", type="primary"):
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
            st.info("🔒 File upload requires analyst or admin role")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Query Interface")
        
        # Show current configuration
        st.markdown(f"""
        <div class="database-info">
        🧠 <strong>LLM:</strong> {selected_llm_display}<br>
        🗄️ <strong>Databases:</strong> {len(selected_databases)} database(s)<br>
        📊 <strong>Accessible Tables:</strong> {len(accessible_tables)} table(s)
        </div>
        """, unsafe_allow_html=True)
        
        # Query input
        user_query = st.text_area(
            "🔍 Enter your data query in natural language",
            value="Show average salary by department" if user_role != UserRole.VIEWER else "Show departments and their locations",
            height=100,
            help="Ask questions about your data in plain English"
        )
        
        # Query configuration
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            if len(selected_databases) == 1:
                data_source = selected_databases[0]
                st.info(f"🗄️ Using: {data_source}")
            else:
                data_source = st.selectbox(
                    "🗄️ Primary database", 
                    options=selected_databases + ["Auto-detect"], 
                    index=len(selected_databases),  # Default to Auto-detect
                    help="Primary database for the query"
                )
        
        with col_config2:
            create_chart = st.checkbox(
                "📊 Generate chart", 
                value=True,
                help="Create a visualization of the results"
            )
        
        with col_config3:
            chart_type = st.selectbox(
                "📈 Chart type", 
                options=["bar", "pie"], 
                index=0,
                disabled=not create_chart
            )

    with col2:
        st.header("🎯 Query Examples")
        
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
        
        for i, example in enumerate(examples):
            if st.button(f"💡 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()
        
        # Use example query if selected
        if 'example_query' in st.session_state:
            user_query = st.session_state.example_query
            del st.session_state.example_query

    # Main action buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)

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

    # Display results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        # Results header with iteration info
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.header("📋 Analysis Results")
        with col_header2:
            if st.session_state.iteration_count > 0:
                st.markdown(f"""
                <span class="iteration-badge">Iteration {st.session_state.iteration_count}</span>
                """, unsafe_allow_html=True)
        
        if result.get("success"):
            # Success metrics
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            
            with col_metric1:
                st.metric("Status", "✅ Success")
            with col_metric2:
                st.metric("User Role", result.get("user_role", "Unknown"))
            with col_metric3:
                st.metric("Agents Used", result.get("agents_used", 0))
            with col_metric4:
                st.metric("LLM Provider", selected_llm_display.split()[0])
            
            # SQL Query display
            if result.get("execution_data"):
                exec_data = result["execution_data"]
                
                st.subheader("📝 Generated SQL Query")
                sql_query = exec_data.get("sql_query", "No SQL returned")
                st.code(sql_query, language="sql")
                
                # Results data
                st.subheader("📊 Query Results")
                
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
                        st.caption("Raw result:")
                        st.dataframe(df, use_container_width=True)
                        
                    else:
                        # Normal data results
                        # Show row count and basic info
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("Rows Returned", len(df))
                        with col_info2:
                            st.metric("Columns", len(df.columns))
                        
                        # Display data table
                        st.dataframe(df, use_container_width=True)
                        
                        # Chart display
                        if create_chart and len(df) > 0:
                            chart_prefix = f"chart_{chart_type}"
                            chart_files = [f for f in os.listdir('.') if f.startswith(chart_prefix) and f.endswith('.png')]
                            
                            if chart_files:
                                # Sort by modified time (newest first)
                                chart_files = sorted(chart_files, key=lambda x: os.path.getmtime(x), reverse=True)
                                latest_chart = chart_files[0]
                                
                                st.subheader("📈 Generated Visualization")
                                st.image(latest_chart, caption=f"📊 {chart_type.title()} Chart - Generated by CrewAI", use_container_width=True)
                            else:
                                st.warning("📈 Chart generation was requested but no chart file was found.")
                else:
                    st.warning("No data returned from query")
            else:
                st.warning("No execution data available")
            
            # Feedback section
            if not st.session_state.feedback_mode:
                st.divider()
                
                col_feedback1, col_feedback2 = st.columns([3, 1])
                
                with col_feedback1:
                    st.subheader("🔄 Feedback & Iteration")
                    st.markdown("Not satisfied with the results? Provide feedback to improve the query!")
                
                with col_feedback2:
                    if st.button("💬 Provide Feedback", type="secondary", use_container_width=True):
                        st.session_state.feedback_mode = True
                        st.rerun()
            
            # Feedback input form
            if st.session_state.feedback_mode:
                st.markdown("""
                <div class="feedback-box">
                    <h4>💬 Provide Specific Feedback</h4>
                    <p>Tell the AI how to improve the query. Be specific about what you want changed.</p>
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
            # Error display
            st.error("❌ Query failed")
            st.text(result.get("error", "No error information available"))
            
            if result.get("user_role"):
                st.info(f"🔒 Note: You are logged in as a {result['user_role']} user with limited access to certain data.")

    # Query history sidebar (user-specific)
    if username in st.session_state.query_history and st.session_state.query_history[username]:
        with st.sidebar:
            st.divider()
            st.header("📝 Your Query History")
            
            user_history = st.session_state.query_history[username]
            
            # Show last 5 queries for current user
            for i, entry in enumerate(reversed(user_history[-5:])):
                status_icon = "✅" if entry['success'] else "❌"
                iteration_text = f"(Iter {entry['iteration']})" if entry['iteration'] > 1 else ""
                llm_info = entry.get('llm_provider', 'Unknown')[:10]
                
                with st.expander(f"{status_icon} {entry['timestamp'].strftime('%H:%M')} {iteration_text}"):
                    st.markdown(f"**Query:** {entry['query'][:100]}...")
                    st.markdown(f"**Success:** {entry['success']}")
                    st.markdown(f"**Iteration:** {entry['iteration']}")
                    st.markdown(f"**LLM:** {llm_info}")
                    if 'databases' in entry:
                        st.markdown(f"**DBs:** {len(entry['databases'])}")

else:
    # Show initialization message
    st.warning("🔧 Please configure the system settings in the sidebar to continue.")
    st.info("Select an LLM provider and database(s) to initialize the CrewAI system.")

# Footer with role-specific help
st.divider()

if st.session_state.app_initialized and 'app' in st.session_state:
    help_text = {
        UserRole.ADMIN: "🔓 **Admin Access:** You have full access to all data including sensitive information like salaries and personal details.",
        UserRole.ANALYST: "📊 **Analyst Access:** You have access to most data but some personal details are restricted.",
        UserRole.VIEWER: "👀 **Viewer Access:** You have read-only access to non-sensitive data only."
    }
    
    if user_role in help_text:
        st.info(help_text[user_role])

st.markdown(f"""
---
**🧠 Enhanced CrewAI SQL Assistant** | Multi-Agent Intelligence with Human-in-the-Loop Feedback  
*Role-Based Access Control • Memory-Driven Learning • Multi-Source Data Queries • Configurable LLM*  
*Current: {selected_llm_display if 'selected_llm_display' in locals() else 'Not configured'} | Databases: {len(selected_databases) if 'selected_databases' in locals() else 0}*
""")
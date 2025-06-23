import streamlit as st
import pandas as pd
import os
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
</style>
""", unsafe_allow_html=True)

# Initialize app
@st.cache_resource
def load_system():
    return CrewAIApp()

# Initialize session state
def init_session_state():
    if 'app' not in st.session_state:
        st.session_state.app = load_system()
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'feedback_mode' not in st.session_state:
        st.session_state.feedback_mode = False
    if 'iteration_count' not in st.session_state:
        st.session_state.iteration_count = 0
    if 'query_history' not in st.session_state:
        st.session_state.query_history = {}  # Changed to dict to store per user

init_session_state()
app = st.session_state.app

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧠 Enhanced CrewAI SQL Assistant</h1>
    <p>Intelligent Multi-Agent Data Analysis with Role-Based Access & Human-in-the-Loop Feedback</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for user configuration
with st.sidebar:
    st.header("👤 User Configuration")
    
    # User role selection
    username = st.selectbox(
        "Select your role", 
        options=["admin", "analyst", "viewer"], 
        index=0,
        help="Different roles have different data access levels"
    )
    
    # Get user role and permissions
    user_role = app.system.role_manager.get_user_role(username)
    accessible_tables = app.system.get_user_accessible_tables(username)
    
    # Role display with styling
    if user_role:
        role_class = f"{user_role.value}-badge"
        st.markdown(f"""
        <div class="role-badge {role_class}">
            {user_role.value.upper()} USER
        </div>
        """, unsafe_allow_html=True)
        
        # Show accessible tables
        st.markdown("**Accessible Tables:**")
        for table in accessible_tables:
            st.markdown(f"• {table}")
    
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
        data_source = st.selectbox(
            "📁 Data source", 
            options=["database", "files", "both"], 
            index=0,
            help="Choose which data sources to query"
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
    
    # Role-based example queries
    if user_role == UserRole.ADMIN:
        examples = [
            "Show all employees with salaries > 70000",
            "Average salary by department",
            "Employee count by hire year",
            "Departments with highest budgets",
            "Top selling products this month"
        ]
    elif user_role == UserRole.ANALYST:
        examples = [
            "Count employees by department",
            "Show all departments and locations",
            "Product categories and stock levels",
            "Sales trends by product",
            "Department employee counts"
        ]
    else:  # VIEWER
        examples = [
            "Show department names and locations",
            "List product categories",
            "Count departments",
            "Show product names by category",
            "Department locations"
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
    if st.session_state.current_result and st.session_state.current_result.get('execution_data'):
        if st.button("💾 Export Results", use_container_width=True):
            filename = f"results_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            export_result = app.system.export_results(None, filename)
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
            data_source=data_source
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
            'iteration': 1
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
            st.metric("LLM Provider", result.get("llm_provider", "None"))
        
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
                
                # Show row count and basic info
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("Rows Returned", len(df))
                with col_info2:
                    st.metric("Columns", len(df.columns))
                
                # Display data table
                st.dataframe(df, use_container_width=True)
                
                # Chart display
                if create_chart:
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
                                data_source=data_source
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
                                'iteration': st.session_state.iteration_count
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
            
            with st.expander(f"{status_icon} {entry['timestamp'].strftime('%H:%M')} {iteration_text}"):
                st.markdown(f"**Query:** {entry['query'][:100]}...")
                st.markdown(f"**Success:** {entry['success']}")
                st.markdown(f"**Iteration:** {entry['iteration']}")

# Footer with role-specific help
st.divider()

help_text = {
    UserRole.ADMIN: "🔓 **Admin Access:** You have full access to all data including sensitive information like salaries and personal details.",
    UserRole.ANALYST: "📊 **Analyst Access:** You have access to most data but some personal details are restricted.",
    UserRole.VIEWER: "👀 **Viewer Access:** You have read-only access to non-sensitive data only."
}

if user_role in help_text:
    st.info(help_text[user_role])

st.markdown("""
---
**🧠 Enhanced CrewAI SQL Assistant** | Multi-Agent Intelligence with Human-in-the-Loop Feedback  
*Role-Based Access Control • Memory-Driven Learning • Multi-Source Data Queries*
""")
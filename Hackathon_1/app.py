import streamlit as st
from main import CrewAIApp
import os

# ✅ This must come first
st.set_page_config(page_title="CrewAI SQL Assistant", layout="wide")

# Initialize app
@st.cache_resource

def load_system():
    return CrewAIApp()

app = load_system()

st.title("🧠 CrewAI-Powered SQL Assistant")
st.markdown("Use natural language to query your data with LLM + Agent collaboration")

# Select user role
username = st.selectbox("👤 Select your role", options=["admin", "analyst", "viewer"], index=0)

# Query input
user_query = st.text_area("🔍 Enter your data query in natural language", "Show average salary by department")

# Data source options
data_source = st.selectbox("📁 Select data source", options=["database", "files", "both"], index=0)

# Visualization
create_chart = st.checkbox("📊 Generate chart", value=True)
chart_type = st.selectbox("📈 Chart type", options=["bar", "pie"], index=0)

# Submit button
if st.button("🚀 Run Analysis"):
    with st.spinner("Running multi-agent analysis..."):
        result = app.system.process_request(
            user_request=user_query,
            username=username,
            create_chart=create_chart,
            chart_type=chart_type,
            data_source=data_source
        )

        if result.get("success"):
            st.success("✅ Query processed successfully!")
            st.subheader("📝 Final SQL Query")
            if result.get("execution_data"):
                st.code(result["execution_data"].get("sql_query", "No SQL returned"), language="sql")
            
            st.subheader("📋 Query Results")
            if result["execution_data"] and "data" in result["execution_data"]:
                import pandas as pd
                df = pd.DataFrame(result["execution_data"]["data"])
                st.dataframe(df)
            else:
                st.warning("No data returned from query")

            # Show chart if generated
            if create_chart:
                #chart_path = os.path.join(os.getcwd(), f"chart_{chart_type}")
                #matching = [f for f in os.listdir('.') if f.startswith(chart_path) and f.endswith('.png')]
                #if matching:
                #    st.subheader("📊 Generated Chart")
                #    st.image(matching[-1])
                #else:
                #    st.warning("No chart image found")
                
                chart_prefix = f"chart_{chart_type}"
                chart_files = [f for f in os.listdir('.') if f.startswith(chart_prefix) and f.endswith('.png')]
                if chart_files:
                    # Sort by modified time (newest last)
                    chart_files = sorted(chart_files, key=lambda x: os.path.getmtime(x), reverse=True)
                    latest_chart = chart_files[0]
                    st.image(latest_chart, caption="📊 Latest Generated Chart")
                else:
                    st.warning("No chart image found.")

        else:
            st.error("❌ Query failed")
            st.text(result.get("error", "No error info"))

# Optional: File upload
with st.expander("📎 Upload and register new file"):
    uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON file")
    file_name_input = st.text_input("Name to register this file as")
    if st.button("📥 Register File"):
        if uploaded_file and file_name_input:
            save_path = os.path.join("uploads", uploaded_file.name)
            os.makedirs("uploads", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            status = app.system.register_file(file_name_input, save_path, username=username)
            if status.get("success"):
                st.success(f"✅ File '{file_name_input}' registered successfully")
            else:
                st.error(f"❌ Failed to register: {status.get('message', 'Unknown error')}")
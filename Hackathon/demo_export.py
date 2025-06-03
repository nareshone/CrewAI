#!/usr/bin/env python3
"""
Demonstration of Excel export with actual query results
"""

from main import CrewAISQLSystem
from datetime import datetime
import pandas as pd

def demo_excel_export():
    """Demonstrate proper Excel export with query results"""
    
    print("🚀 CrewAI SQL Analysis - Excel Export Demo")
    print("="*60)
    
    # Initialize system
    print("📊 Initializing system...")
    system = CrewAISQLSystem()
    
    # Example 1: Direct SQL Query with Export
    print("\n1️⃣ Example 1: Direct SQL Query")
    print("-"*40)
    
    sql_query = "SELECT department, COUNT(*) as employee_count, AVG(salary) as avg_salary FROM employees GROUP BY department"
    result = system.direct_query(sql_query)
    
    if result.get('success') and result.get('data'):
        filename1 = f"department_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Export using the system's export method
        export_result = system.export_results(None, filename1)
        print(f"✅ {export_result}")
        print(f"   Check the file: {filename1}")
    
    # Example 2: Natural Language Query with Export
    print("\n2️⃣ Example 2: Natural Language Query with CrewAI")
    print("-"*40)
    
    # Process a natural language request
    nl_result = system.process_request(
        user_request="Show me top 5 employees by salary with their departments",
        username="admin",
        create_chart=False,
        data_source="database"
    )
    
    if nl_result.get('success') and nl_result.get('execution_data'):
        exec_data = nl_result['execution_data']
        
        # Create Excel file with the actual query results
        filename2 = f"top_employees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        if 'data' in exec_data and exec_data['data']:
            with pd.ExcelWriter(filename2, engine='openpyxl') as writer:
                # Query results
                df = pd.DataFrame(exec_data['data'])
                df.to_excel(writer, sheet_name='Query Results', index=False)
                
                # Metadata
                metadata = pd.DataFrame({
                    'Query Type': ['Natural Language'],
                    'Original Request': ['Show me top 5 employees by salary with their departments'],
                    'SQL Generated': [exec_data.get('sql_query', 'N/A')],
                    'Rows Returned': [exec_data.get('row_count', len(df))],
                    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'LLM Used': [nl_result.get('llm_provider', 'Unknown')]
                })
                metadata.to_excel(writer, sheet_name='Query Info', index=False)
                
            print(f"✅ Natural language query results exported to: {filename2}")
            print(f"   SQL Generated: {exec_data.get('sql_query', 'N/A')}")
            print(f"   Rows: {exec_data.get('row_count', 0)}")
    
    # Example 3: Query with User Feedback
    print("\n3️⃣ Example 3: Query with Modifications")
    print("-"*40)
    
    # First query
    print("Initial query: Average salary by department")
    result1 = system.process_request(
        user_request="Show average salary by department",
        username="admin"
    )
    
    # Query with feedback
    print("\nModified query with feedback...")
    result2 = system.process_request(
        user_request="Show average salary by department",
        username="admin",
        feedback="Also include the employee count and minimum/maximum salary for each department"
    )
    
    if result2.get('execution_data'):
        exec_data = result2['execution_data']
        filename3 = f"salary_analysis_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Store and export
        system.last_execution_result = exec_data
        export_result = system.export_results(None, filename3)
        print(f"✅ {export_result}")
    
    print("\n" + "="*60)
    print("📊 Demo Complete!")
    print("Check the generated Excel files for the actual query results.")
    print("Each file contains:")
    print("  - Query Results sheet with the actual data")
    print("  - Metadata/Query Info sheet with query details")
    print("="*60)

if __name__ == "__main__":
    demo_excel_export()
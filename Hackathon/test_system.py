def test_openai_api():
    """Test if OpenAI API is properly configured"""
    print("🔍 Testing OpenAI API configuration...")
    print("   (Note: Main system prioritizes Groq, but OpenAI is tested for fallback)")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("   Please add: OPENAI_API_KEY=your_key_here")
        print("   Get your key from: https://platform.openai.com/")
        return False
    
    print("✅ OPENAI_API_KEY found in environment")
    
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=openai_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=100  # Reduce tokens for faster test
        )
    except Exception as e:
        print(f"❌ Failed to import ChatOpenAI: {e}")
        #!/usr/bin/env python3
"""
Test script to verify CrewAI SQL Analysis System functionality

Note: This test checks OpenAI API configuration, but the main system
prioritizes Groq API if available. You can have either or both configured.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_api():
    """Test if OpenAI API is properly configured"""
    print("🔍 Testing OpenAI API configuration...")
    print("   (Note: Main system prioritizes Groq, but OpenAI is tested for fallback)")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("   Please add: OPENAI_API_KEY=your_key_here")
        print("   Get your key from: https://platform.openai.com/")
        return False
    
    print("✅ OPENAI_API_KEY found in environment")
    
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=openai_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=100  # Reduce tokens for faster test
        )
        
        # Test the connection
        response = llm.invoke("Say 'Hello, CrewAI system is working!'")
        print(f"✅ OpenAI API test successful")
        print(f"   Response: {response.content if hasattr(response, 'content') else response}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        if "401" in str(e) or "invalid" in str(e).lower():
            print("   🔑 Check your API key is valid")
        elif "rate" in str(e).lower():
            print("   ⏱️  Rate limit exceeded")
        elif "connection" in str(e).lower():
            print("   🌐 Check internet connection")
        return False

def test_database():
    """Test database connectivity"""
    print("\n🔍 Testing database...")
    
    import sqlite3
    import pandas as pd
    
    try:
        conn = sqlite3.connect('sample.db')
        
        # Test query
        df = pd.read_sql_query("SELECT COUNT(*) as count FROM employees", conn)
        count = df['count'][0]
        
        print(f"✅ Database test successful")
        print(f"   Employees table has {count} records")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_excel_export():
    """Test Excel export functionality"""
    print("\n🔍 Testing Excel export...")
    
    import pandas as pd
    from datetime import datetime
    
    try:
        # Create test data
        test_data = {
            'id': [1, 2, 3],
            'name': ['Test 1', 'Test 2', 'Test 3'],
            'value': [100, 200, 300]
        }
        df = pd.DataFrame(test_data)
        
        # Export to Excel
        filename = f"test_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Test Data', index=False)
            
            # Add metadata
            metadata = pd.DataFrame({
                'Test': ['Excel Export'],
                'Status': ['Success'],
                'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        print(f"✅ Excel export test successful")
        print(f"   Created file: {filename}")
        
        # Clean up
        os.remove(filename)
        print(f"   Cleaned up test file")
        
        return True
        
    except Exception as e:
        print(f"❌ Excel export test failed: {e}")
        return False

def test_full_system():
    """Test the full CrewAI system"""
    print("\n🔍 Testing full CrewAI system...")
    
    try:
        from main import CrewAISQLSystem
        
        # Initialize system
        system = CrewAISQLSystem()
        
        print(f"✅ System initialized")
        print(f"   LLM Provider: {type(system.llm).__name__ if system.llm else 'None'}")
        
        # Check which API key is being used
        if system.llm:
            llm_type = type(system.llm).__name__
            if "ChatGroq" in llm_type:
                print(f"   Using Groq API (prioritized)")
            elif "ChatOpenAI" in llm_type:
                print(f"   Using OpenAI API")
            else:
                print(f"   Using: {llm_type}")
        
        # Test direct query
        result = system.direct_query("SELECT name, salary FROM employees LIMIT 3")
        
        if result.get('success'):
            print(f"✅ Direct query test successful")
            print(f"   Returned {result.get('row_count', 0)} rows")
            
            # Test export
            if result.get('data'):
                export_result = system.export_results()
                print(f"✅ Export test: {export_result}")
                
                # Clean up exported file
                if "exported to:" in export_result:
                    filename = export_result.split("exported to: ")[1]
                    if os.path.exists(filename):
                        os.remove(filename)
                        print(f"   Cleaned up: {filename}")
        else:
            print(f"❌ Direct query failed: {result.get('error', 'Unknown error')}")
            
        return True
        
    except Exception as e:
        print(f"❌ Full system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_api_keys():
    """Check status of all API keys"""
    print("\n📋 API Key Status:")
    print("-" * 40)
    
    groq_key = os.getenv('GROQ_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    print(f"GROQ_API_KEY:      {'✅ Set' if groq_key else '❌ Not set'} (Prioritized by main system)")
    print(f"OPENAI_API_KEY:    {'✅ Set' if openai_key else '❌ Not set'} (Tested in this script)")
    print(f"ANTHROPIC_API_KEY: {'✅ Set' if anthropic_key else '❌ Not set'} (Alternative)")
    print("-" * 40)

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 CrewAI SQL Analysis System Test Suite")
    print("="*60)
    
    # Check API keys status first
    check_api_keys()
    
    tests = [
        ("OpenAI API", test_openai_api),
        ("Database", test_database),
        ("Excel Export", test_excel_export),
        ("Full System", test_full_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nRun the system with:")
        print("  python main.py          # CLI mode")
        print("  uvicorn fastapi_app:app --reload  # Web UI")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. Make sure OPENAI_API_KEY is in your .env file")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check internet connection for API access")
        print("4. If using Groq instead, ensure GROQ_API_KEY is set")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
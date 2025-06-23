import re
from typing import Dict, List, Optional
from crewai.tools import BaseTool

class SQLConverterTool(BaseTool):
    name: str = "sql_converter"
    description: str = "Convert SQLite SQL queries to other database dialects (SQL Server, PostgreSQL, DB2)"
    
    def _run(self, sql_query: str, target_db: str = "postgresql") -> str:
        """Convert SQLite SQL to target database dialect"""
        try:
            # Clean the input query
            query = sql_query.strip()
            if not query.upper().startswith('SELECT'):
                return f"Error: Only SELECT queries are supported for conversion"
            
            # Choose conversion method based on target database
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
        
        # PostgreSQL-specific conversions
        conversions = [
            # Date functions
            (r'\bDATE\s*\(\s*([^)]+)\s*\)', r"TO_DATE(\1, 'YYYY-MM-DD')"),
            (r'\bDATETIME\s*\(\s*([^)]+)\s*\)', r"TO_TIMESTAMP(\1, 'YYYY-MM-DD HH24:MI:SS')"),
            (r'\bstrftime\s*\(\s*[\'"]%Y[\'"],\s*([^)]+)\s*\)', r"EXTRACT(YEAR FROM \1)"),
            (r'\bstrftime\s*\(\s*[\'"]%m[\'"],\s*([^)]+)\s*\)', r"EXTRACT(MONTH FROM \1)"),
            (r'\bstrftime\s*\(\s*[\'"]%d[\'"],\s*([^)]+)\s*\)', r"EXTRACT(DAY FROM \1)"),
            
            # String functions
            (r'\|\|', r"||"),  # Concatenation is the same
            (r'\bLENGTH\s*\(', r"LENGTH("),
            (r'\bSUBSTR\s*\(', r"SUBSTRING("),
            
            # Limit clause (PostgreSQL uses LIMIT, same as SQLite)
            # No change needed
            
            # Data types
            (r'\bINTEGER\b', r"INTEGER"),
            (r'\bTEXT\b', r"VARCHAR"),
            (r'\bREAL\b', r"DECIMAL"),
            
            # Boolean values
            (r"'active'", r"'active'"),  # No change needed
            
            # Quote identifiers if they might be reserved words
            (r'\b(user|order|group|table|schema)\b', r'"\1"'),
        ]
        
        for pattern, replacement in conversions:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        
        # Add PostgreSQL-specific optimizations
        if "GROUP BY" in converted.upper():
            # PostgreSQL is stricter about GROUP BY
            converted = self._fix_postgresql_group_by(converted)
        
        return f"-- PostgreSQL Query\n{converted}"
    
    def _convert_to_sqlserver(self, query: str) -> str:
        """Convert SQLite query to SQL Server"""
        converted = query
        
        # SQL Server-specific conversions
        conversions = [
            # Limit to TOP
            (r'\bLIMIT\s+(\d+)\s*;?\s*$', r""),  # Remove LIMIT first
            
            # Date functions
            (r'\bDATE\s*\(\s*([^)]+)\s*\)', r"CAST(\1 AS DATE)"),
            (r'\bDATETIME\s*\(\s*([^)]+)\s*\)', r"CAST(\1 AS DATETIME)"),
            (r'\bstrftime\s*\(\s*[\'"]%Y[\'"],\s*([^)]+)\s*\)', r"YEAR(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%m[\'"],\s*([^)]+)\s*\)', r"MONTH(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%d[\'"],\s*([^)]+)\s*\)', r"DAY(\1)"),
            
            # String concatenation
            (r'\|\|', r"+"),
            
            # String functions
            (r'\bSUBSTR\s*\(', r"SUBSTRING("),
            (r'\bLENGTH\s*\(', r"LEN("),
            
            # Data types
            (r'\bINTEGER\b', r"INT"),
            (r'\bTEXT\b', r"NVARCHAR(MAX)"),
            (r'\bREAL\b', r"DECIMAL(18,2)"),
            
            # Quote identifiers with square brackets
            (r'"([^"]+)"', r'[\1]'),
        ]
        
        for pattern, replacement in conversions:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        
        # Handle LIMIT -> TOP conversion
        limit_match = re.search(r'\bLIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            limit_value = limit_match.group(1)
            # Add TOP clause after SELECT
            converted = re.sub(r'\bSELECT\b', f"SELECT TOP {limit_value}", converted, count=1, flags=re.IGNORECASE)
        
        return f"-- SQL Server Query\n{converted}"
    
    def _convert_to_db2(self, query: str) -> str:
        """Convert SQLite query to DB2"""
        converted = query
        
        # DB2-specific conversions
        conversions = [
            # Date functions
            (r'\bDATE\s*\(\s*([^)]+)\s*\)', r"DATE(\1)"),
            (r'\bDATETIME\s*\(\s*([^)]+)\s*\)', r"TIMESTAMP(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%Y[\'"],\s*([^)]+)\s*\)', r"YEAR(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%m[\'"],\s*([^)]+)\s*\)', r"MONTH(\1)"),
            (r'\bstrftime\s*\(\s*[\'"]%d[\'"],\s*([^)]+)\s*\)', r"DAY(\1)"),
            
            # String concatenation
            (r'\|\|', r"||"),  # DB2 supports ||
            
            # String functions
            (r'\bSUBSTR\s*\(', r"SUBSTR("),
            (r'\bLENGTH\s*\(', r"LENGTH("),
            
            # Data types
            (r'\bINTEGER\b', r"INTEGER"),
            (r'\bTEXT\b', r"VARCHAR(1000)"),
            (r'\bREAL\b', r"DECIMAL(15,2)"),
            
            # LIMIT to FETCH FIRST
            (r'\bLIMIT\s+(\d+)\s*;?\s*$', r"FETCH FIRST \1 ROWS ONLY"),
        ]
        
        for pattern, replacement in conversions:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        
        return f"-- DB2 Query\n{converted}"
    
    def _fix_postgresql_group_by(self, query: str) -> str:
        """Fix PostgreSQL GROUP BY requirements"""
        # PostgreSQL requires all non-aggregate columns in SELECT to be in GROUP BY
        # This is a simplified fix - in practice, you might need more sophisticated parsing
        return query
    
    def get_supported_databases(self) -> List[str]:
        """Get list of supported target databases"""
        return ["postgresql", "sqlserver", "db2"]
    
    def get_conversion_features(self, target_db: str) -> Dict[str, str]:
        """Get information about conversion features for target database"""
        features = {
            "postgresql": {
                "strengths": "Full SQL standard compliance, advanced data types, excellent date/time functions",
                "limitations": "Stricter GROUP BY requirements, case-sensitive identifiers",
                "notes": "Recommended for complex analytical queries"
            },
            "sqlserver": {
                "strengths": "Excellent integration with Microsoft stack, powerful T-SQL features",
                "limitations": "Uses TOP instead of LIMIT, different string concatenation",
                "notes": "Good for enterprise environments"
            },
            "db2": {
                "strengths": "High performance, excellent for large datasets, OLAP functions",
                "limitations": "More strict syntax requirements, different pagination approach",
                "notes": "Ideal for enterprise and mainframe environments"
            }
        }
        return features.get(target_db.lower(), {})
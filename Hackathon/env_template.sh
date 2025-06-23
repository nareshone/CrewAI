# CrewAI SQL Assistant - Environment Configuration
# Copy this file to .env and add your actual API keys

# =============================================================================
# LLM API KEYS (Choose one or more)
# =============================================================================

# Groq API Key (Recommended - Free tier available)
# Get your key at: https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# OpenAI API Key (Paid service)
# Get your key at: https://platform.openai.com/
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (Paid service)  
# Get your key at: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Debug mode (True for development, False for production)
DEBUG=True

# Server configuration
HOST=0.0.0.0
PORT=8000

# Maximum upload file size (in MB)
MAX_UPLOAD_SIZE=100

# Session timeout (in minutes)
SESSION_TIMEOUT=60

# =============================================================================
# DATABASE SETTINGS
# =============================================================================

# Database directory path
DATABASE_PATH=./databases/

# Default database files (comma-separated)
DEFAULT_DATABASES=sample.db

# Enable database backups
ENABLE_BACKUPS=True

# Backup interval (in hours)
BACKUP_INTERVAL=24

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

# Secret key for session management (generate a random string)
SECRET_KEY=your_secret_key_here

# Allowed origins for CORS (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000

# Rate limiting (requests per minute per user)
RATE_LIMIT=60

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Log file path
LOG_FILE=./logs/crewai_assistant.log

# Enable request logging
LOG_REQUESTS=True

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Enable chart generation
ENABLE_CHARTS=True

# Enable file uploads
ENABLE_FILE_UPLOADS=True

# Enable data exports
ENABLE_EXPORTS=True

# Enable feedback system
ENABLE_FEEDBACK=True

# Maximum feedback iterations
MAX_FEEDBACK_ITERATIONS=5

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Maximum concurrent connections
MAX_CONNECTIONS=100

# Query timeout (in seconds)
QUERY_TIMEOUT=300

# Cache size (number of queries to cache)
CACHE_SIZE=1000

# =============================================================================
# ADVANCED SETTINGS (Optional)
# =============================================================================

# Redis connection (if using Redis for sessions)
# REDIS_URL=redis://localhost:6379/0

# PostgreSQL connection (if using PostgreSQL instead of SQLite)
# DATABASE_URL=postgresql://user:password@localhost:5432/crewai

# Ollama server URL (if using local Ollama)
# OLLAMA_URL=http://localhost:11434

# Custom LLM endpoint
# CUSTOM_LLM_URL=

# =============================================================================
# USAGE NOTES
# =============================================================================

# 1. At minimum, add one LLM API key (Groq recommended for free usage)
# 2. Generate a random SECRET_KEY for security
# 3. Adjust paths according to your setup
# 4. Set DEBUG=False for production deployment
# 5. Configure ALLOWED_ORIGINS for your domain in production

# Example with Groq (free option):
# GROQ_API_KEY=gsk_1234567890abcdef
# DEBUG=True
# SECRET_KEY=your-super-secret-random-key-here

# Without API keys (testing mode):
# Just set DEBUG=True and use "Mock LLM" option in the interface
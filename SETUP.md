# AskDB Setup & Configuration Guide

This guide provides detailed instructions for setting up and configuring AskDB for your environment.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 1GB free space
- **Network**: Internet connection for API access

### Required Accounts & API Keys
- **Google AI**: [Get Google AI API Key](https://makersuite.google.com/app/apikey)
- **LangChain**: [Get LangChain API Key](https://cloud.langchain.com/)

### Database Requirements
- Any SQL database (MySQL, PostgreSQL, SQLite, SQL Server)
- Database user with read permissions
- Network access to database server

## 🛠️ Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd AskDB
```

### Step 2: Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv askdb_env
source askdb_env/bin/activate  # On Windows: askdb_env\Scripts\activate

# Or using conda
conda create -n askdb_env python=3.9
conda activate askdb_env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import langchain; import flask; print('Installation successful!')"
```

## 🔧 Configuration

### Environment Variables Setup

Create a `.env` file in the project root:

```bash
# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key_here

# LangChain Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=askdb_project
LANGCHAIN_API_KEY=your_langchain_api_key_here

# Optional: Database Configuration (if not using direct connection)
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
```

### Database Configuration

#### MySQL Setup
```python
# In untitled0.py, replace the database connection:
db = SQLDatabase.from_uri("mysql+pymysql://username:password@host:port/database_name")

# Example:
db = SQLDatabase.from_uri("mysql+pymysql://admin:admin123@localhost:3306/mydb")
```

#### PostgreSQL Setup
```python
# Install psycopg2 first: pip install psycopg2-binary
db = SQLDatabase.from_uri("postgresql://username:password@host:port/database_name")

# Example:
db = SQLDatabase.from_uri("postgresql://postgres:mypassword@localhost:5432/mydb")
```

#### SQLite Setup
```python
# For local development
db = SQLDatabase.from_uri("sqlite:///path/to/your/database.db")

# Example:
db = SQLDatabase.from_uri("sqlite:///./data/mydatabase.db")
```

#### SQL Server Setup
```python
# Install pyodbc first: pip install pyodbc
db = SQLDatabase.from_uri("mssql+pyodbc://username:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server")

# Example:
db = SQLDatabase.from_uri("mssql+pyodbc://sa:mypassword@localhost:1433/mydb?driver=ODBC+Driver+17+for+SQL+Server")
```

### Database Schema Configuration

Create `database_table_descriptions.csv` with your table descriptions:

```csv
table_name,description
users,"Contains user information including id, name, email, and registration date"
orders,"Stores order data with order_id, user_id, total_amount, and order_date"
products,"Product catalog with product_id, name, price, and category"
customers,"Customer information with contact details and preferences"
payments,"Payment records with transaction details and status"
```

### Example Schema Files

#### E-commerce Database
```csv
table_name,description
customers,"Customer profiles with personal information, contact details, and preferences"
orders,"Order records with customer_id, order_date, total_amount, and status"
order_items,"Individual items in orders with product_id, quantity, and price"
products,"Product catalog with name, description, price, category, and inventory"
categories,"Product categories and subcategories"
payments,"Payment transactions with order_id, amount, payment_method, and status"
```

#### CRM Database
```csv
table_name,description
contacts,"Contact information including name, email, phone, and company"
leads,"Lead records with source, status, and conversion probability"
opportunities,"Sales opportunities with value, stage, and close date"
accounts,"Company accounts with industry, size, and contact information"
activities,"Activities and interactions with contacts and leads"
deals,"Deal records with value, stage, and probability"
```

## 🚀 Running AskDB

### Development Mode
```bash
# Start the Flask server
python code1.py

# Server will be available at http://localhost:5000
```

### Production Mode
```bash
# Using gunicorn (install first: pip install gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 code1:app

# Using uwsgi (install first: pip install uwsgi)
uwsgi --http :5000 --wsgi-file code1.py --callable app
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "code1.py"]
```

```bash
# Build and run
docker build -t askdb .
docker run -p 5000:5000 askdb
```

## 🧪 Testing Your Setup

### Test Database Connection
```python
# Test script: test_db.py
from untitled0 import db

try:
    # Test connection
    result = db.run("SELECT 1 as test")
    print("Database connection successful!")
    print(f"Test result: {result}")
except Exception as e:
    print(f"Database connection failed: {e}")
```

### Test API Endpoint
```bash
# Test the API
curl -X POST http://localhost:5000/api \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users are in the database?"}'
```

### Test with Sample Data
```sql
-- Create sample tables and data
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_date DATE
);

INSERT INTO users VALUES (1, 'John Doe', 'john@example.com', '2023-01-01');
INSERT INTO users VALUES (2, 'Jane Smith', 'jane@example.com', '2023-01-02');
```

## 🔧 Advanced Configuration

### Customizing AI Behavior

#### Adjust Model Parameters
```python
# In untitled0.py, modify the LLM configuration:
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1,  # Lower for more consistent results
    max_tokens=2048,  # Adjust based on your needs
    top_p=0.9,        # Control response diversity
    top_k=40          # Control response diversity
)
```

#### Customizing Example Selection
```python
# Adjust the number of examples retrieved
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    vectorstore,
    k=3,  # Increase for more examples
    input_keys=["input"]
)
```

### Performance Optimization

#### Database Connection Pooling
```python
# For better performance with multiple connections
from sqlalchemy import create_engine

engine = create_engine(
    "mysql+pymysql://user:password@host:port/database",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
db = SQLDatabase(engine)
```

#### Caching Configuration
```python
# Add caching for frequently asked questions
import redis
from functools import lru_cache

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=1000)
def cached_query_generation(question):
    # Your query generation logic here
    pass
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Error: "Access denied for user"
# Solution: Check database credentials and permissions

# Error: "Connection refused"
# Solution: Verify database server is running and accessible
```

#### 2. API Key Issues
```bash
# Error: "Invalid API key"
# Solution: Verify your Google AI and LangChain API keys

# Error: "Quota exceeded"
# Solution: Check your API usage limits
```

#### 3. Import Errors
```bash
# Error: "ModuleNotFoundError"
# Solution: Install missing dependencies
pip install -r requirements.txt
```

#### 4. Memory Issues
```bash
# Error: "Out of memory"
# Solution: Increase system memory or optimize queries
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add to your Flask app
app.debug = True
```

### Performance Monitoring
```python
# Add performance monitoring
import time

def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## 🔒 Security Best Practices

### Production Security
1. **Use HTTPS**: Configure SSL/TLS certificates
2. **API Authentication**: Implement JWT or API key authentication
3. **Input Validation**: Sanitize all user inputs
4. **Database Security**: Use least privilege database users
5. **Environment Variables**: Never commit API keys to version control

### Example Security Configuration
```python
# Add authentication middleware
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_KEY'):
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api', methods=['POST'])
@require_api_key
def master1():
    # Your existing code here
    pass
```

## 📊 Monitoring & Logging

### Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('askdb.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)
```

### Health Check Endpoint
```python
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "database": "connected",
        "ai_model": "available"
    })
```

This setup guide should help you get AskDB running in any environment. For additional help, check the main README or create an issue in the repository.

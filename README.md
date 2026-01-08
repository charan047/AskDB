# AskDB - Natural Language to SQL Query System

AskDB is an intelligent database querying system that allows users to ask questions in plain English and get answers from any SQL database. The system uses AI/LLM technology to automatically convert natural language questions into SQL queries and provides meaningful responses.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Any SQL database (MySQL, PostgreSQL, SQLite, etc.)
- Google AI API key
- LangChain API key

### Installation

1. **Clone and install dependencies**
   ```bash
   git clone <repository-url>
   cd AskDB
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   export GOOGLE_API_KEY="your_google_api_key"
   export LANGCHAIN_TRACING_V2="true"
   export LANGCHAIN_PROJECT="your_project_name"
   export LANGCHAIN_API_KEY="your_langchain_api_key"
   ```

3. **Configure your database**
   Update the database connection in `untitled0.py`:
   ```python
   # For MySQL
   db = SQLDatabase.from_uri("mysql+pymysql://user:password@host:port/database")
   
   # For PostgreSQL
   db = SQLDatabase.from_uri("postgresql://user:password@host:port/database")
   
   # For SQLite
   db = SQLDatabase.from_uri("sqlite:///path/to/database.db")
   ```

4. **Create your table descriptions**
   Create `database_table_descriptions.csv`:
   ```csv
   table_name,description
   users,"Contains user information including id, name, email"
   orders,"Stores order data with order_id, user_id, total_amount"
   ```

5. **Start the server**
   ```bash
   python code1.py
   ```

### Usage

Send questions to the API:
```bash
curl -X POST http://localhost:5000/api \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the price of 1968 Ford Mustang?"}'
```

**Response:**
```json
{
  "answer": "The price of 1968 Ford Mustang is $95,000."
}
```

## 🎯 Key Features

- **Natural Language Processing**: Ask questions in plain English
- **Universal Database Support**: Works with any SQL database
- **AI-Powered Query Generation**: Uses Google Gemini 1.5 Pro
- **Context Awareness**: Maintains conversation history
- **Dynamic Learning**: Improves accuracy through example-based learning
- **Web API**: RESTful API for easy integration

## 📁 Project Structure

```
AskDB/
├── README.md                           # This file - Quick start guide
├── TECHNICAL.md                        # Advanced AI techniques & implementation
├── SETUP.md                           # Detailed installation & configuration
├── API.md                             # Complete API documentation
├── code1.py                           # Flask API server
├── untitled0.py                       # AI engine and query processing
├── test.py                           # MongoDB test file
├── database_table_descriptions.csv    # Database schema documentation
└── requirements.txt                   # Python dependencies
```

## 🔍 Example Questions

- "List all customers in France with a credit limit over 20,000"
- "Get the highest payment amount made by any customer"
- "Show product details for products in the 'Motorcycles' product line"
- "Find all employees who joined after 2020"
- "Calculate the total sales for each department"

## 🤖 How It Works

1. **Question Input**: User submits natural language question
2. **Table Selection**: AI identifies relevant database tables
3. **Query Generation**: Converts question to SQL using examples
4. **Query Execution**: Runs SQL against your database
5. **Answer Formatting**: Returns natural language response

## 📚 Documentation

- **[TECHNICAL.md](TECHNICAL.md)** - Advanced AI techniques, LangChain integration, and implementation details
- **[SETUP.md](SETUP.md)** - Detailed installation, configuration, and troubleshooting
- **[API.md](API.md)** - Complete API reference and examples

## 🛡️ Security

- Store API keys securely using environment variables
- Implement authentication for production use
- Validate user inputs
- Use parameterized queries to prevent SQL injection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

- Create an issue in the repository
- Check the documentation in the `/docs` folder
- Review example queries in the codebase


## Local Setup (VS Code)
### 1) Create venv + install deps
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure env vars
Copy `.env.example` → `.env` and fill:
- `GOOGLE_API_KEY`
- `DATABASE_URL`

### 3) (Optional) Create a demo SQLite DB
```bash
python scripts/init_sqlite.py
```

### 4) Run
```bash
python code1.py
```
API: `POST http://localhost:5000/api` with JSON:
```json
{"question":"What is the price of 1968 Ford Mustang?","session_id":"demo"}
```

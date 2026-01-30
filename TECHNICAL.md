# AskDB Technical Documentation

This document covers the advanced AI techniques, LangChain integration, and implementation details of AskDB.

##  Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │   AI Engine     │
│   (Web UI)      │◄──►│   (code1.py)    │◄──►│ (untitled0.py)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Conversation   │    │   SQL Database  │
                       │    History      │    │   (Any Type)    │
                       └─────────────────┘    └─────────────────┘
```

## 🧠 Advanced AI Techniques

AskDB implements several sophisticated AI techniques to provide accurate and context-aware database querying:

### 🔄 Dynamic Example Selection
- **Semantic Similarity**: Uses Google's embedding model to find the most relevant examples
- **Vector Database**: ChromaDB stores and retrieves similar question-query pairs
- **Adaptive Learning**: System improves by learning from similar past queries
- **Top-K Retrieval**: Selects the 2 most similar examples for each query

### 🎯 Dynamic Table Selection
- **Structured Output**: Uses Pydantic models to ensure consistent table identification
- **Semantic Understanding**: AI analyzes question intent to identify relevant tables
- **Multi-Table Support**: Can handle queries spanning multiple tables
- **Schema Awareness**: Leverages table descriptions for better understanding

### 💬 Context Awareness
- **Conversation History**: Maintains chat history using LangChain's ChatMessageHistory
- **Follow-up Queries**: Understands references to previous questions
- **Contextual References**: Handles pronouns like "it", "that", "them" based on conversation
- **Session Management**: Preserves context across multiple API calls

### 🎓 Few-Shot Learning
- **Example-Based Training**: Uses curated question-query pairs as training examples
- **Pattern Recognition**: Learns common query patterns and structures
- **Domain Adaptation**: Adapts to specific database schemas through examples
- **Query Templates**: Builds reusable query patterns for similar questions

### 🔧 Query Processing Pipeline
- **Multi-Stage Processing**: Chain-based architecture for modular processing
- **Error Handling**: Robust error handling with fallback mechanisms
- **Query Validation**: Ensures generated SQL is syntactically correct
- **Result Formatting**: Converts raw database results into natural language

## 🔗 LangChain Integration

AskDB is built on LangChain, leveraging its powerful framework for building AI applications:

### 🏗️ Core LangChain Components

#### **LLM Integration**
```python
# Google Gemini integration via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
```

#### **Prompt Management**
```python
# Few-shot prompt templates
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}\nSQLQuery:"),
    ("ai", "{query}"),
])

# System prompts with table context
table_details_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return the names of ALL the SQL tables that MIGHT be relevant..."),
    ("human", "{question}")
])
```

#### **Chain Orchestration**
```python
# Complex chain with multiple stages
chain = (
    RunnablePassthrough.assign(table_names_to_use=select_table) |
    RunnablePassthrough.assign(query=generate_query | RunnableLambda(clean_sql_query)) |
    RunnablePassthrough.assign(result=itemgetter("query") | execute_query) |
    rephrase_answer
)
```

#### **Output Parsing**
```python
# String output parser for final responses
rephrase_answer = answer_prompt | llm | StrOutputParser()

# Structured output for table selection
structured_llm = llm.with_structured_output(Table)
```

### 🧠 LangChain Features Used

#### **RunnablePassthrough**
- Enables data flow between chain components
- Maintains context across processing stages
- Allows conditional execution paths

#### **RunnableLambda**
- Custom function integration within chains
- SQL query cleaning and validation
- Data transformation between stages

#### **Example Selectors**
```python
# Semantic similarity example selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    vectorstore,
    k=2,
    input_keys=["input"]
)
```

#### **Vector Stores**
```python
# ChromaDB integration for example storage
vectorstore = Chroma()
vectorstore.delete_collection()
```

#### **Database Tools**
```python
# SQL database integration
execute_query = QuerySQLDataBaseTool(db=db)
generate_query = create_sql_query_chain(llm, db, final_prompt)
```

#### **Message History**
```python
# Conversation state management
history = ChatMessageHistory()
formatted_messages = [
    {"role": "user" if msg.type == "user" else "assistant", "content": msg.content}
    for msg in history.messages
]
```

### 🔄 LangChain Chain Flow

```
User Input
    ↓
[ChatMessageHistory] → Context Preservation
    ↓
[SemanticSimilarityExampleSelector] → Dynamic Example Retrieval
    ↓
[Structured Output LLM] → Table Selection
    ↓
[SQL Query Chain] → Query Generation
    ↓
[RunnableLambda] → Query Cleaning
    ↓
[QuerySQLDataBaseTool] → Database Execution
    ↓
[Answer Prompt Chain] → Response Formatting
    ↓
Final Answer
```

### 🎯 LangChain Benefits in AskDB

- **Modularity**: Each component can be independently tested and modified
- **Composability**: Easy to add new features by extending existing chains
- **Observability**: Built-in tracing and monitoring capabilities
- **Flexibility**: Support for multiple LLM providers and databases
- **Scalability**: Efficient handling of complex multi-stage processing
- **Reliability**: Robust error handling and fallback mechanisms

## 🧩 Technical Implementation Details

### Vector Embeddings
```python
# Uses Google's embedding model for semantic similarity
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    task_type="retrieval_query"
)
```

### Structured Output
```python
# Pydantic model for structured table selection
class Table(BaseModel):
    name: List[str] = Field(description="List of relevant table names")
```

### Chain Architecture
```python
# Modular chain for processing
chain = (
    RunnablePassthrough.assign(table_names_to_use=select_table) |
    RunnablePassthrough.assign(query=generate_query) |
    rephrase_answer
)
```

### Context Management
```python
# Conversation history maintenance
history = ChatMessageHistory()
history.add_user_message(question)
history.add_ai_message(response)
```

### SQL Query Cleaning
```python
def clean_sql_query(text: str) -> str:
    """
    Clean SQL query by removing code block syntax, various SQL tags, backticks,
    prefixes, and unnecessary whitespace while preserving the core SQL query.
    """
    # Remove code block syntax
    block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    text = re.sub(block_pattern, r"\1", text, flags=re.DOTALL)
    
    # Handle prefixes
    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQL)\s*:\s*"
    text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE)
    
    # Extract SQL statement
    sql_statement_pattern = r"(SELECT.*?;)"
    sql_match = re.search(sql_statement_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if sql_match:
        text = sql_match.group(1)
    
    # Remove backticks and normalize whitespace
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
```

## 🔍 Detailed Example Flow

```
User Question: "What is the price of 1968 Ford Mustang?"

1. Dynamic Table Selection: AI identifies 'products' table using structured output
2. Dynamic Example Selection: Retrieves similar examples like "what is price of product X"
3. Context-Aware Query Generation: "SELECT buyPrice, MSRP FROM products WHERE productName = '1968 Ford Mustang' LIMIT 1;"
4. Query Execution: Database returns price data
5. Answer Formatting: "The price of 1968 Ford Mustang is $95,000."
6. Response: JSON with formatted answer
```

## 🗄️ Database Schema Management

AskDB works with any database schema. The system automatically discovers your database structure and uses it to generate appropriate queries. You just need to provide a CSV file (`database_table_descriptions.csv`) that describes your tables:

| Column | Description |
|--------|-------------|
| `table_name` | Name of the table in your database |
| `description` | Human-readable description of what the table contains |

### Example Schema File
```csv
table_name,description
users,"Contains user information including id, name, email, and registration date"
orders,"Stores order data with order_id, user_id, total_amount, and order_date"
products,"Product catalog with product_id, name, price, and category"
```

The AI will use these descriptions to understand your database structure and generate accurate queries.

## 🔧 Configuration Details

### Environment Variables
- `GOOGLE_API_KEY`: Your Google AI API key for Gemini access
- `LANGCHAIN_TRACING_V2`: Enable LangChain tracing (set to "true")
- `LANGCHAIN_PROJECT`: Your LangChain project name
- `LANGCHAIN_API_KEY`: Your LangChain API key

### Database Configuration Examples
```python
# MySQL
db = SQLDatabase.from_uri("mysql+pymysql://user:password@host:port/database")

# PostgreSQL
db = SQLDatabase.from_uri("postgresql://user:password@host:port/database")

# SQLite
db = SQLDatabase.from_uri("sqlite:///path/to/database.db")

# SQL Server
db = SQLDatabase.from_uri("mssql+pyodbc://user:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server")
```




## Async execution (RQ + Redis)

AskDB supports background execution for long queries.
- POST /api with `{ "async": true }` returns `202 { job_id }`.
- Poll `GET /job/<job_id>` until status is `finished`.

## Observability (Prometheus metrics)

- `GET /metrics` exposes Prometheus-format metrics.
- Tracks request latency, DB vs LLM time, cache hits, and job counts.

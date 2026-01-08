# AskDB API Documentation

Complete API reference for the AskDB Natural Language to SQL Query System.

## 🚀 Base URL

- **Development**: `http://localhost:5000`
- **Production**: `https://your-domain.com`

## 📋 Authentication

Currently, the API doesn't require authentication. For production use, implement API key authentication:

```bash
# Add API key to headers
curl -H "X-API-Key: your_api_key" \
  -X POST http://localhost:5000/api \
  -d '{"question": "Your question here"}'
```

## 🔗 Endpoints

### POST `/api`

Send natural language questions to get database answers.

#### Request Format

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "question": "string"
}
```

#### Response Format

**Success Response (200):**
```json
{
  "answer": "string"
}
```

**Error Response (400/500):**
```json
{
  "error": "string"
}
```

#### Example Requests

##### Basic Query
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

##### Complex Query
```bash
curl -X POST http://localhost:5000/api \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me all customers in France with a credit limit over 20,000"}'
```

**Response:**
```json
{
  "answer": "Here are the customers in France with a credit limit over 20,000:\n\n1. Jean Dupont - Credit Limit: $25,000\n2. Marie Martin - Credit Limit: $30,000\n3. Pierre Durand - Credit Limit: $22,500"
}
```

##### Follow-up Query
```bash
# First query
curl -X POST http://localhost:5000/api \
  -H "Content-Type: application/json" \
  -d '{"question": "How many orders do we have?"}'

# Follow-up query (maintains context)
curl -X POST http://localhost:5000/api \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total value of those orders?"}'
```

**Response:**
```json
{
  "answer": "The total value of all orders is $1,250,000."
}
```

## 📊 Query Examples by Category

### Customer Queries
```json
{
  "question": "List all customers in France"
}
```

```json
{
  "question": "Find customers with credit limit over 50,000"
}
```

```json
{
  "question": "Show me the top 10 customers by order value"
}
```

### Product Queries
```json
{
  "question": "What products are in the Motorcycles category?"
}
```

```json
{
  "question": "Find products with stock quantity less than 100"
}
```

```json
{
  "question": "What is the average price of products in each category?"
}
```

### Order Queries
```json
{
  "question": "Show orders from the last 30 days"
}
```

```json
{
  "question": "Calculate total sales for each month this year"
}
```

```json
{
  "question": "Find orders with status 'pending'"
}
```

### Employee Queries
```json
{
  "question": "List employees who joined after 2020"
}
```

```json
{
  "question": "Show employees and their managers"
}
```

```json
{
  "question": "Find employees in the Sales department"
}
```

### Analytics Queries
```json
{
  "question": "What is our monthly revenue trend?"
}
```

```json
{
  "question": "Which products are selling the best?"
}
```

```json
{
  "question": "Show customer retention rates by region"
}
```

## 🔄 Conversation Context

The API maintains conversation history to support follow-up questions and contextual references.

### Context Features

- **Follow-up Questions**: "What about the ones from Germany?"
- **Pronoun Resolution**: "Show me their contact details"
- **Comparative Queries**: "Which ones have higher values?"
- **Contextual Filters**: "Only show the active ones"

### Example Conversation Flow

```bash
# Initial query
curl -X POST http://localhost:5000/api \
  -d '{"question": "Show me customers in France"}'

# Response: Lists French customers

# Follow-up with context
curl -X POST http://localhost:5000/api \
  -d '{"question": "What is the total credit limit for these customers?"}'

# Response: Calculates total for French customers only

# Another follow-up
curl -X POST http://localhost:5000/api \
  -d '{"question": "Show me their recent orders"}'

# Response: Shows orders for French customers
```

## ⚠️ Error Handling

### Common Error Codes

#### 400 Bad Request
```json
{
  "error": "Missing 'question' in the request body"
}
```

**Causes:**
- Missing question field
- Invalid JSON format
- Empty question

#### 500 Internal Server Error
```json
{
  "error": "Database connection failed"
}
```

**Causes:**
- Database connection issues
- AI model errors
- Query generation failures

### Error Response Examples

#### Invalid Question Format
```bash
curl -X POST http://localhost:5000/api \
  -H "Content-Type: application/json" \
  -d '{"invalid_field": "test"}'
```

**Response:**
```json
{
  "error": "Missing 'question' in the request body"
}
```

#### Database Connection Error
```bash
# When database is unavailable
curl -X POST http://localhost:5000/api \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me all users"}'
```

**Response:**
```json
{
  "error": "Database connection failed: Connection refused"
}
```

#### AI Model Error
```bash
# When AI model is unavailable
curl -X POST http://localhost:5000/api \
  -H "Content-Type: application/json" \
  -d '{"question": "Complex query here"}'
```

**Response:**
```json
{
  "error": "AI model error: Rate limit exceeded"
}
```

## 🔧 Advanced Usage

### Batch Processing

For multiple questions, send them sequentially:

```bash
#!/bin/bash

questions=(
  "How many customers do we have?"
  "What is the total revenue this year?"
  "Show me the top 5 products"
)

for question in "${questions[@]}"; do
  echo "Question: $question"
  curl -X POST http://localhost:5000/api \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$question\"}" | jq '.answer'
  echo "---"
done
```

### Using with JavaScript

```javascript
async function askQuestion(question) {
  try {
    const response = await fetch('http://localhost:5000/api', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question })
    });
    
    const data = await response.json();
    
    if (response.ok) {
      return data.answer;
    } else {
      throw new Error(data.error);
    }
  } catch (error) {
    console.error('Error:', error.message);
    throw error;
  }
}

// Usage
askQuestion("What is the price of 1968 Ford Mustang?")
  .then(answer => console.log(answer))
  .catch(error => console.error(error));
```

### Using with Python

```python
import requests
import json

def ask_question(question, base_url="http://localhost:5000"):
    """Send a question to AskDB API"""
    url = f"{base_url}/api"
    headers = {"Content-Type": "application/json"}
    data = {"question": question}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["answer"]
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

# Usage
answer = ask_question("What is the price of 1968 Ford Mustang?")
print(answer)
```

### Using with cURL Scripts

```bash
#!/bin/bash

# Function to ask questions
ask_db() {
    local question="$1"
    curl -s -X POST http://localhost:5000/api \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"$question\"}" | jq -r '.answer'
}

# Example usage
echo "Customer count:"
ask_db "How many customers do we have?"

echo -e "\nRevenue:"
ask_db "What is our total revenue this year?"

echo -e "\nTop products:"
ask_db "Show me the top 5 products by sales"
```

## 📈 Performance Considerations

### Response Times

- **Simple queries**: 1-3 seconds
- **Complex queries**: 3-8 seconds
- **Multi-table joins**: 5-15 seconds

### Optimization Tips

1. **Be Specific**: "Show me customers in France" vs "Show me everything"
2. **Use Filters**: "Orders from last month" vs "All orders"
3. **Limit Results**: "Top 10 customers" vs "All customers"
4. **Avoid Ambiguity**: "Customers with credit limit > 50,000" vs "Big customers"

### Rate Limiting

Currently no rate limiting is implemented. For production use, consider:

```python
# Example rate limiting implementation
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api', methods=['POST'])
@limiter.limit("10 per minute")
def master1():
    # Your existing code
    pass
```

## 🔍 Debugging

### Enable Debug Mode

```python
# In code1.py
app.debug = True
```

### Log Requests

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/api', methods=['POST'])
def master1():
    logger.info(f"Received question: {request.json.get('question')}")
    # ... rest of your code
```

### Test Database Connection

```bash
# Test if database is accessible
curl -X POST http://localhost:5000/api \
  -H "Content-Type: application/json" \
  -d '{"question": "SELECT 1 as test"}'
```

## 📚 Best Practices

### Writing Good Questions

✅ **Good Examples:**
- "Show me customers in France with credit limit over 20,000"
- "What is the total revenue for Q1 2024?"
- "List the top 10 products by sales volume"

❌ **Avoid:**
- "Show me everything"
- "Get all the data"
- "What's in the database?"

### Error Handling

Always handle potential errors in your client code:

```javascript
try {
  const answer = await askQuestion("Your question here");
  console.log(answer);
} catch (error) {
  console.error("API Error:", error.message);
  // Handle error appropriately
}
```

### Security

- Never send sensitive data in questions
- Use HTTPS in production
- Implement authentication for production use
- Validate and sanitize inputs

This API documentation should help you integrate AskDB into your applications effectively. For additional help, check the main README or create an issue in the repository.

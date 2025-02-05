# WhatsApp Chat Analyzer 💬

## Introduction

WhatsApp Chat Analyzer is an advanced Python-based tool that provides deep insights into your conversations by leveraging sophisticated natural language processing and machine learning techniques. This analyzer breaks down chat dynamics, communication patterns, and relationship nuances using state-of-the-art language models.

## Key Features

- 🔍 Comprehensive Chat Analysis
- 🕒 Response Time Tracking
- 📊 Communication Pattern Detection
- 🤖 AI-Powered Insights
- 🔒 Privacy-Focused Processing

## Data Flow and Processing

1. **Zip File Import**: Export WhatsApp chat without media
2. **Text Preprocessing**:
   - Emoji Removal
   - Tokenization
   - Stopword Elimination
3. **Chunked Analysis**:
   - Break conversation into manageable chunks
   - Analyze using GROQ's Language Model
4. **Insight Generation**:
   - Relationship Dynamics
   - Emotional Undertones
   - Communication Evolution

## Prerequisites

### Installation

Install all required libraries with a single pip command:

```bash
pip install zipfile os re emoji time logging nltk pickle datetime collections groq
```

### GROQ API Setup

1. Sign up at [Groq Developer Portal](https://console.groq.com/)
2. Generate an API key
3. Set the environment variable:
   ```bash
   export GROQ_API_KEY='your_api_key_here'
   ```

## How to Use

1. Export your WhatsApp chat as a `.txt` file
   - **Important**: Export WITHOUT media
   - Ensure UTF-8 encoding

2. Run the script:
   ```python
   analyzer = EnhancedChunkedChatAnalyzer()
   full_path = "/path/to/your/WhatsApp_Chat_Export.zip"
   analysis, context = analyzer.extract_and_analyze(full_path)
   ```

3. Interactive Querying:
   ```python
   # After initial analysis, ask follow-up questions
   additional_query = "Summarize our communication frequency"
   response = analyzer.generate_query_response(additional_query, context)
   print(response)
   ```

## Use Cases

- 📈 Personal Communication Analytics
- 🤝 Relationship Pattern Understanding
- 🤖 Conversational AI Training
- 🔍 Personal Message Automator
   - Generate personalized responses based on conversation history
   - Create chatbots mimicking individual communication styles

## Performance Optimization

- 🚀 Lightweight Token Processing
- 📊 Efficient Chunk-based Analysis
- 🔬 Minimal Resource Consumption

## Privacy & Security

- 🔒 No Cloud Storage
- 💻 Local Processing and Groq security policy
- 🚫 Emoji and Sensitive Data Removal

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

## Disclaimer

This tool is for personal analysis. Always respect privacy and obtain necessary consent before analyzing conversations.

---

**Happy Analyzing! 🕵️‍♀️📱**

```markdown
# 🤖 Self-RAG Chatbot with Docling & GIST Embeddings

> A sophisticated RAG (Retrieval-Augmented Generation) chatbot implementing Self-RAG methodology with advanced PDF processing and semantic understanding capabilities.

## ✨ Features

- 🧠 **Self-RAG Implementation**: Advanced retrieval-augmented generation with self-reflection and iterative refinement
- 📄 **Smart PDF Processing**: High-quality document parsing using Docling for accurate text extraction
- 🎯 **Semantic Understanding**: GIST embeddings for superior document comprehension and retrieval
- ⚡ **Rate Limiting**: Built-in intelligent rate limiting for Groq API to prevent quota exhaustion
- 🖥️ **Interactive Web UI**: Clean and intuitive Streamlit interface with real-time chat
- 🔄 **Document Grading**: Automatic relevance scoring and hallucination detection
- 📊 **Usage Monitoring**: Real-time API usage tracking and limit visualization

## 🛠️ Technology Stack

### **Core Framework**
- 🌟 **Streamlit** - Interactive web application framework
- 🏗️ **LangGraph** - Workflow orchestration and state management

### **AI/ML Components**
- 🤖 **Groq API** - Fast LLM inference (gemma2-9b-it model)
- 🧬 **LangChain** - RAG pipeline and prompt management
- 🎯 **GIST Embeddings** - Advanced semantic text embeddings
- 📚 **SentenceTransformers** - Embedding model infrastructure

### **Document Processing**
- 📄 **Docling** - High-quality PDF parsing and text extraction
- ✂️ **Semantic Chunker** - Intelligent document segmentation

### **Vector Storage**
- 🗃️ **ChromaDB** - Efficient vector database for embeddings
- 🔍 **Similarity Search** - Fast semantic document retrieval

### **Data & Utilities**
- 📋 **Pydantic** - Data validation and structured outputs
- 🔒 **Threading** - Thread-safe rate limiting
- ⏰ **DateTime** - Time-based API quota management

## 🔄 Self-RAG Workflow

```
graph TD
    A[🚀 START] --> B[🔍 Retrieve Documents]
    B --> C[📊 Grade Document Relevance]
    C --> D{📋 Relevant Docs Found?}
    
    D -->|✅ Yes| E[🎯 Generate Answer]
    D -->|❌ No| F[🔄 Transform Query]
    
    F --> B
    
    E --> G[🔍 Grade Generation Quality]
    G --> H{🤔 Quality Check}
    
    H -->|✅ Useful & Grounded| I[🎉 END - Return Answer]
    H -->|❌ Not Useful| F
    H -->|⚠️ Hallucinated| F
    
    style A fill:#e1f5fe
    style I fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#f3e5f5
```

## 🚀 Quick Start

### Prerequisites
- 🐍 Python 3.8 or higher
- 🔑 Groq API key

### Installation

1. **Clone the repository:**
```
git clone https://github.com/yourusername/self-rag-chatbot.git
cd self-rag-chatbot
```

2. **Install dependencies:**
```
pip install -r requirements.txt
```

3. **Configure API Key:**
Update the API key in `app.py` at line 628, or set it as an environment variable:
```
groq_api_key = "your_groq_api_key_here"
```

4. **Run the application:**
```
streamlit run app.py
```

5. **Open your browser:**
Navigate to `http://localhost:8501` to start using the chatbot! 🎉

## 📖 How to Use

1. **📁 Upload Documents**: Use the sidebar to upload PDF files or enter URLs
2. **⚙️ Process Documents**: Click "Process Documents" to create the vector knowledge base
3. **💬 Start Chatting**: Ask questions about your documents in the chat interface
4. **📊 Monitor Usage**: Check API rate limits in the sidebar dashboard

## 🔧 Configuration

### **Model Settings**
- **LLM Model**: `gemma2-9b-it` (Groq)
- **Embedding Model**: `avsolatorio/GIST-large-Embedding-v0`
- **Fallback Model**: `sentence-transformers/all-MiniLM-L6-v2`

### **Rate Limits**
- 📈 **RPM**: 30 requests per minute
- 🚀 **TPM**: 15,000 tokens per minute  
- 📅 **RPD**: 14,400 requests per day
- 💾 **TPD**: 500,000 tokens per day

### **Self-RAG Parameters**
- **Recursion Limit**: 10 iterations max
- **Chunk Size**: Minimum 50 characters
- **Semantic Threshold**: 80th percentile breakpoint

## 🏗️ Architecture

The application follows a modular architecture:

```
📦 Self-RAG Chatbot
├── 🤖 GroqLLM - API client with rate limiting
├── 🧬 GISTEmbeddings - Semantic embedding model
├── 📊 SelfRAGChatbot - Main orchestration class
├── 🔄 StateGraph - LangGraph workflow engine
├── 📋 Pydantic Models - Data validation schemas
└── 🖥️ Streamlit UI - Web interface components
```

## 🎯 Self-RAG Process

1. **🔍 Document Retrieval**: Query vector database for relevant chunks
2. **📊 Relevance Grading**: AI evaluates document relevance to question
3. **🎯 Answer Generation**: Create response using relevant context
4. **🔍 Quality Assessment**: Check for hallucinations and usefulness
5. **🔄 Iterative Refinement**: Retry with improved queries if needed

## 🛡️ Error Handling

- ⚡ **Rate Limit Protection**: Automatic waiting and retry logic
- 🔄 **Fallback Mechanisms**: Graceful degradation when limits exceeded  
- 🛠️ **Recursion Prevention**: Max iteration limits to avoid infinite loops
- 📝 **Comprehensive Logging**: Detailed error reporting and user feedback

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🤖 **Groq** for fast LLM inference
- 📄 **Docling** team for excellent PDF processing
- 🎯 **GIST Embeddings** for semantic understanding
- 🏗️ **LangChain** community for RAG frameworks
- 🌟 **Streamlit** for the amazing web framework

---



**Built with ❤️ using cutting-edge AI technologies**

[⭐ Star this repo](https://github.com/yourusername/self-rag-chatbot) • [🐛 Report Bug](https://github.com/yourusername/self-rag-chatbot/issues) • [✨ Request Feature](https://github.com/yourusername/self-rag-chatbot/issues)


```

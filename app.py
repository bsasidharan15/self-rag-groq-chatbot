import streamlit as st
import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import List, Any
from pathlib import Path
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_docling import DoclingLoader
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from sentence_transformers import SentenceTransformer
from collections import deque

# Set page config
st.set_page_config(
    page_title="Self-RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rate limiting class for Groq API
class GroqRateLimiter:
    def __init__(self):
        # gemma2-9b-it rate limits
        self.rpm_limit = 30  # Requests Per Minute
        self.tpm_limit = 15000  # Tokens Per Minute
        self.rpd_limit = 14400  # Requests Per Day
        self.tpd_limit = 500000  # Tokens Per Day
        
        # Tracking queues
        self.request_times = deque()
        self.token_usage = deque()
        self.daily_requests = deque()
        self.daily_tokens = deque()
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
    
    def _clean_old_entries(self):
        """Remove entries older than the time window"""
        now = datetime.now()
        
        # Clean minute-based tracking
        minute_ago = now - timedelta(minutes=1)
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()
        while self.token_usage and self.token_usage[0][0] < minute_ago:
            self.token_usage.popleft()
        
        # Clean day-based tracking
        day_ago = now - timedelta(days=1)
        while self.daily_requests and self.daily_requests[0] < day_ago:
            self.daily_requests.popleft()
        while self.daily_tokens and self.daily_tokens[0][0] < day_ago:
            self.daily_tokens.popleft()
    
    def can_make_request(self, estimated_tokens=100):
        """Check if we can make a request without exceeding limits"""
        with self.lock:
            self._clean_old_entries()
            
            # Check RPM limit
            if len(self.request_times) >= self.rpm_limit:
                return False, "RPM limit reached (30 requests/minute)"
            
            # Check TPM limit
            current_minute_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_minute_tokens + estimated_tokens > self.tpm_limit:
                return False, f"TPM limit would be exceeded ({current_minute_tokens + estimated_tokens}/{self.tpm_limit})"
            
            # Check RPD limit
            if len(self.daily_requests) >= self.rpd_limit:
                return False, "RPD limit reached (14400 requests/day)"
            
            # Check TPD limit
            current_daily_tokens = sum(tokens for _, tokens in self.daily_tokens)
            if current_daily_tokens + estimated_tokens > self.tpd_limit:
                return False, f"TPD limit would be exceeded ({current_daily_tokens + estimated_tokens}/{self.tpd_limit})"
            
            return True, "OK"
    
    def record_request(self, tokens_used):
        """Record a successful request"""
        with self.lock:
            now = datetime.now()
            self.request_times.append(now)
            self.token_usage.append((now, tokens_used))
            self.daily_requests.append(now)
            self.daily_tokens.append((now, tokens_used))
    
    def get_status(self):
        """Get current rate limit status"""
        with self.lock:
            self._clean_old_entries()
            
            current_minute_tokens = sum(tokens for _, tokens in self.token_usage)
            current_daily_tokens = sum(tokens for _, tokens in self.daily_tokens)
            
            return {
                "rpm_used": len(self.request_times),
                "rpm_limit": self.rpm_limit,
                "tpm_used": current_minute_tokens,
                "tpm_limit": self.tpm_limit,
                "rpd_used": len(self.daily_requests),
                "rpd_limit": self.rpd_limit,
                "tpd_used": current_daily_tokens,
                "tpd_limit": self.tpd_limit
            }
    
    def wait_time_needed(self):
        """Calculate how long to wait before next request"""
        with self.lock:
            if not self.request_times:
                return 0
            
            # Time until oldest request in current minute expires
            oldest_request = self.request_times[0]
            wait_time = 60 - (datetime.now() - oldest_request).total_seconds()
            return max(0, wait_time)

# Custom GIST Embeddings using SentenceTransformers
class GISTEmbeddings:
    def __init__(
        self,
        model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        **kwargs: Any,
    ) -> None:
        """Initialize GIST embeddings using SentenceTransformers."""
        try:
            self.model = SentenceTransformer(model_name)
            st.success(f"✅ Loaded GIST model: {model_name}")
        except Exception as e:
            st.warning(f"⚠️ Failed to load {model_name}, falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        embeddings = self.model.encode([query], convert_to_tensor=False)
        return embeddings[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [emb.tolist() for emb in embeddings]

# Groq LLM wrapper with rate limiting
class GroqLLM:
    def __init__(self, model="gemma2-9b-it", api_key=None):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.rate_limiter = GroqRateLimiter()
    
    def _estimate_tokens(self, text):
        """Rough estimation of tokens (1 token ≈ 4 characters)"""
        return max(len(text) // 4, 10)
    
    def with_structured_output(self, pydantic_class):
        return GroqStructuredOutput(self.client, self.model, pydantic_class, self.rate_limiter)
    
    def invoke(self, inputs):
        if isinstance(inputs, dict):
            if 'context' in inputs and 'question' in inputs:
                content = f"Question: {inputs['question']}\n\nContext: {inputs['context']}\n\nAnswer:"
            elif 'question' in inputs:
                content = inputs['question']
            else:
                content = str(inputs)
        elif isinstance(inputs, str):
            content = inputs
        else:
            content = str(inputs)
        
        # Estimate tokens for this request
        estimated_tokens = self._estimate_tokens(content) + 100  # Add buffer for response
        
        # Check rate limits
        can_proceed, reason = self.rate_limiter.can_make_request(estimated_tokens)
        if not can_proceed:
            wait_time = self.rate_limiter.wait_time_needed()
            if wait_time > 0:
                st.warning(f"⏳ Rate limit reached: {reason}. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)  # Add 1 second buffer
            else:
                st.error(f"❌ Daily rate limit exceeded: {reason}")
                return "I've reached the daily API limit. Please try again tomorrow."
        
        messages = [{"role": "user", "content": content}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
                stream=False,
            )
            
            # Record successful request
            actual_tokens = estimated_tokens  # In production, use response.usage.total_tokens if available
            self.rate_limiter.record_request(actual_tokens)
            
            return response.choices[0].message.content
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                wait_time = self.rate_limiter.wait_time_needed()
                st.warning(f"⏳ API rate limit hit. Waiting {wait_time + 10:.1f} seconds before retry...")
                time.sleep(wait_time + 10)
                
                # Retry once
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=1024,
                        top_p=1,
                        stream=False,
                    )
                    self.rate_limiter.record_request(actual_tokens)
                    return response.choices[0].message.content
                except Exception as retry_error:
                    st.error(f"❌ Retry failed: {str(retry_error)}")
                    return "I encountered an API error. Please try again in a moment."
            else:
                st.error(f"❌ API Error: {str(e)}")
                return "I encountered an error processing your request."

class GroqStructuredOutput:
    def __init__(self, client, model, pydantic_class, rate_limiter):
        self.client = client
        self.model = model
        self.pydantic_class = pydantic_class
        self.rate_limiter = rate_limiter
    
    def _estimate_tokens(self, text):
        """Rough estimation of tokens (1 token ≈ 4 characters)"""
        return max(len(text) // 4, 10)
    
    def invoke(self, inputs):
        prompt = f"""
        Please respond with a JSON object that matches this schema:
        {self.pydantic_class.__doc__}
        
        Required format:
        {{"binary_score": "yes" or "no"}}
        
        Question: {inputs.get('question', '')}
        Document: {inputs.get('document', '')}
        Documents: {inputs.get('documents', '')}
        Generation: {inputs.get('generation', '')}
        
        Respond only with the JSON object.
        """
        
        # Estimate tokens for this request
        estimated_tokens = self._estimate_tokens(prompt) + 20  # Small response expected
        
        # Check rate limits
        can_proceed, reason = self.rate_limiter.can_make_request(estimated_tokens)
        if not can_proceed:
            wait_time = self.rate_limiter.wait_time_needed()
            if wait_time > 0:
                st.info(f"⏳ Rate limit check: {reason}. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)
            else:
                st.warning(f"⚠️ Daily limit reached: {reason}")
                # Return default response to continue workflow
                return self.pydantic_class(binary_score="yes")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
                top_p=1,
                stream=False,
            )
            
            # Record successful request
            self.rate_limiter.record_request(estimated_tokens)
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                wait_time = self.rate_limiter.wait_time_needed()
                st.info(f"⏳ API rate limit during grading. Waiting {wait_time + 5:.1f} seconds...")
                time.sleep(wait_time + 5)
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=50,
                        top_p=1,
                        stream=False,
                    )
                    self.rate_limiter.record_request(estimated_tokens)
                except Exception:
                    # Fallback to default response
                    return self.pydantic_class(binary_score="yes")
            else:
                # Fallback to default response
                return self.pydantic_class(binary_score="yes")
        
        result = response.choices[0].message.content.strip()
        try:
            if '{' in result and '}' in result:
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                json_str = result[json_start:json_end]
                parsed = json.loads(json_str)
                return self.pydantic_class(**parsed)
            else:
                if 'yes' in result.lower():
                    return self.pydantic_class(binary_score="yes")
                else:
                    return self.pydantic_class(binary_score="no")
        except:
            return self.pydantic_class(binary_score="yes")

# Data models for Self-RAG
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# Graph state
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def create_groq_runnable(model="gemma2-9b-it", api_key=None):
    llm = GroqLLM(model, api_key)
    return RunnableLambda(llm.invoke)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class SelfRAGChatbot:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.llm_instance = GroqLLM(api_key=groq_api_key)
        self.llm = create_groq_runnable(api_key=groq_api_key)
        self.vectorstore = None
        self.retriever = None
        self.app = None
        self._setup_components()
    
    def _setup_components(self):
        # Setup graders
        self.structured_llm_grader = self.llm_instance.with_structured_output(GradeDocuments)
        self.hallucination_grader = self.llm_instance.with_structured_output(GradeHallucinations)
        self.answer_grader = self.llm_instance.with_structured_output(GradeAnswer)
        
        # Setup prompts
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
            ("human", "Question: {question} \nContext: {context} \nAnswer:")
        ])
        
        self.question_rewriter_prompt = ChatPromptTemplate.from_messages([
            ("system", """You a question re-writer that converts an input question to a better version that is optimized 
             for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])
        
        # Create chains
        self.rag_chain = self.rag_prompt | self.llm
        self.question_rewriter = self.question_rewriter_prompt | self.llm
    
    def load_documents(self, file_paths):
        """Load and process documents using Docling"""
        all_docs = []
        
        for file_path in file_paths:
            if file_path:
                try:
                    st.info(f"📄 Loading document: {file_path}")
                    loader = DoclingLoader(file_path=file_path)
                    docs = loader.load()
                    
                    # Clean up text content to fix spacing issues
                    for doc in docs:
                        # Fix common PDF parsing issues
                        content = doc.page_content
                        # Add space before capital letters that follow lowercase
                        import re
                        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
                        # Fix number/currency formatting
                        content = re.sub(r'(\d)([A-Za-z])', r'\1 \2', content)
                        content = re.sub(r'([A-Za-z])(\d)', r'\1 \2', content)
                        # Fix common word concatenations
                        content = content.replace('perminute', ' per minute')
                        content = content.replace('wouldcost', ' would cost')
                        content = content.replace('whileother', ' while other')
                        content = content.replace('serviceslike', ' services like')
                        # Clean up multiple spaces
                        content = re.sub(r'\s+', ' ', content).strip()
                        
                        doc.page_content = content
                    
                    all_docs.extend(docs)
                    st.success(f"✅ Loaded {len(docs)} chunks from {file_path}")
                except Exception as e:
                    st.error(f"❌ Error loading {file_path}: {str(e)}")
        
        return all_docs
    
    def setup_vectorstore(self, documents):
        """Setup vectorstore with semantic chunking and GIST embeddings"""
        if not documents:
            return
        
        # Filter complex metadata that Chroma can't handle
        st.info("🔧 Filtering complex metadata from documents...")
        filtered_documents = filter_complex_metadata(
            documents, 
            allowed_types=(str, bool, int, float)
        )
        
        # Initialize GIST embeddings
        st.info("🔧 Loading GIST embeddings model...")
        gist_embeddings = GISTEmbeddings()
        
        # Use semantic chunker
        st.info("🔧 Setting up semantic chunker...")
        semantic_chunker = SemanticChunker(
            embeddings=gist_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.8,
            buffer_size=1,
            min_chunk_size=50
        )
        
        # Split documents
        st.info("🔧 Chunking documents semantically...")
        doc_splits = semantic_chunker.split_documents(filtered_documents)
        
        # Filter metadata from splits as well (in case chunker adds complex metadata)
        doc_splits = filter_complex_metadata(
            doc_splits,
            allowed_types=(str, bool, int, float)
        )
        
        # Create vectorstore
        st.info("🔧 Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="self-rag-chroma",
            embedding=gist_embeddings,
        )
        self.retriever = self.vectorstore.as_retriever()
        
        # Setup the graph
        self._setup_graph()
        
        st.success(f"✅ Vector store created with {len(doc_splits)} chunks!")
    
    def _setup_graph(self):
        """Setup the Self-RAG graph with recursion limits"""
        workflow = StateGraph(GraphState)
        
        # Define nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        
        # Build graph with better flow control
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "transform_query",  # Changed from "generate" to avoid loop
                "useful": END,
                "not useful": "transform_query",
            },
        )
        
        # Compile with recursion limit
        self.app = workflow.compile()
        
        # Set a reasonable recursion limit to prevent infinite loops
        try:
            from langchain_core.runnables.config import RunnableConfig
            self.config = RunnableConfig(recursion_limit=10)
        except:
            self.config = {"recursion_limit": 10}
    
    def retrieve(self, state):
        """Retrieve documents"""
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def generate(self, state):
        """Generate answer"""
        question = state["question"]
        documents = state["documents"]
        
        context = format_docs(documents)
        generation = self.rag_chain.invoke({"context": context, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state):
        """Grade document relevance"""
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        for d in documents:
            score = self.structured_llm_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade == "yes":
                filtered_docs.append(d)
        
        return {"documents": filtered_docs, "question": question}
    
    def transform_query(self, state):
        """Transform the query"""
        question = state["question"]
        documents = state["documents"]
        
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}
    
    def decide_to_generate(self, state):
        """Decide whether to generate or transform query"""
        filtered_documents = state["documents"]
        
        if not filtered_documents:
            return "transform_query"
        else:
            return "generate"
    
    def grade_generation_v_documents_and_question(self, state):
        """Grade generation against documents and question"""
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score.binary_score
        
        if grade == "yes":
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                return "useful"
            else:
                return "not useful"
        else:
            return "not supported"
    
    def chat(self, question):
        """Main chat function with better error handling"""
        if not self.app:
            return "Please upload and process documents first."
        
        inputs = {"question": question}
        final_state = None
        max_iterations = 5  # Prevent infinite loops
        iteration_count = 0
        
        try:
            # Use the app with recursion limit
            for output in self.app.stream(inputs, config=getattr(self, 'config', {})):
                iteration_count += 1
                if iteration_count > max_iterations:
                    st.warning("⚠️ Max iterations reached. Providing best available answer.")
                    break
                    
                for key, value in output.items():
                    final_state = value
                    
                # If we have a generation, that's good enough
                if final_state and "generation" in final_state and final_state["generation"]:
                    break
        
        except Exception as e:
            if "recursion" in str(e).lower():
                st.warning("⚠️ Graph recursion limit reached. Trying simplified approach...")
                # Fallback: simple retrieval + generation
                try:
                    question = inputs["question"]
                    documents = self.retriever.invoke(question)
                    if documents:
                        context = format_docs(documents[:3])  # Use top 3 docs
                        generation = self.rag_chain.invoke({"context": context, "question": question})
                        return generation
                    else:
                        return "I couldn't find relevant information in the documents to answer your question."
                except Exception as fallback_error:
                    return f"I encountered an error: {str(fallback_error)}"
            else:
                return f"I encountered an error: {str(e)}"
        
        if final_state and "generation" in final_state and final_state["generation"]:
            return final_state['generation']
        else:
            return "I'm sorry, I couldn't generate a response to your question."

def main():
    st.title("🤖 Self-RAG Chatbot with Docling & GIST Embeddings")
    st.markdown("A sophisticated RAG chatbot using Self-RAG, Docling for PDF parsing, and GIST embeddings for semantic understanding.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key - directly embedded
        groq_api_key = "enter-your-groq-key"
        
        # Optional: Display API key input for override (commented out for security)
        # groq_api_key = st.text_input("Groq API Key", value=groq_api_key, type="password", help="Enter your Groq API key")
        
        st.success("✅ API Key configured")
        
        if not groq_api_key:
            st.warning("Please enter your Groq API key to continue.")
            st.stop()
        
        # Rate limit status
        if 'chatbot' in st.session_state and st.session_state.chatbot:
            status = st.session_state.chatbot.llm_instance.rate_limiter.get_status()
            st.header("📊 Rate Limit Status")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Requests/Min", f"{status['rpm_used']}/{status['rpm_limit']}")
                st.metric("Requests/Day", f"{status['rpd_used']}/{status['rpd_limit']}")
            with col2:
                st.metric("Tokens/Min", f"{status['tpm_used']}/{status['tpm_limit']}")
                st.metric("Tokens/Day", f"{status['tpd_used']}/{status['tpd_limit']}")
            
            # Warning indicators
            if status['rpm_used'] / status['rpm_limit'] > 0.8:
                st.warning("⚠️ Approaching RPM limit")
            if status['tpm_used'] / status['tpm_limit'] > 0.8:
                st.warning("⚠️ Approaching TPM limit")
            if status['rpd_used'] / status['rpd_limit'] > 0.9:
                st.error("🚨 Approaching daily request limit")
            if status['tpd_used'] / status['tpd_limit'] > 0.9:
                st.error("🚨 Approaching daily token limit")
        
        st.header("📁 Document Upload")
        
        # File upload options
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to create the knowledge base"
        )
        
        # URL input
        url_input = st.text_area(
            "Or enter URLs (one per line)",
            help="Enter URLs to PDF documents, one per line",
            placeholder="https://example.com/document.pdf"
        )
        
        # Process documents button
        process_docs = st.button("🔄 Process Documents", type="primary")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    # Process documents
    if process_docs:
        if uploaded_files or url_input.strip():
            with st.spinner("🔄 Processing documents... This may take a few minutes."):
                try:
                    # Initialize chatbot
                    st.session_state.chatbot = SelfRAGChatbot(groq_api_key)
                    
                    # Prepare file paths
                    file_paths = []
                    
                    # Handle uploaded files
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            # Save uploaded file temporarily
                            temp_path = f"temp_{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(temp_path)
                    
                    # Handle URLs
                    if url_input.strip():
                        urls = [url.strip() for url in url_input.strip().split('\n') if url.strip()]
                        file_paths.extend(urls)
                    
                    # Load and process documents
                    documents = st.session_state.chatbot.load_documents(file_paths)
                    st.session_state.chatbot.setup_vectorstore(documents)
                    
                    # Clean up temporary files
                    for path in file_paths:
                        if path.startswith("temp_") and os.path.exists(path):
                            os.remove(path)
                    
                    st.session_state.documents_processed = True
                    st.success(f"✅ Successfully processed {len(documents)} document chunks!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
        else:
            st.warning("Please upload files or enter URLs before processing.")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    if not st.session_state.documents_processed:
        st.info("👆 Please upload and process documents using the sidebar to start chatting.")
        return
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()

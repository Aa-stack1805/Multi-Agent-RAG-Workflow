import os
import asyncio
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, TypedDict
from dataclasses import dataclass
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

@dataclass
class WorkflowConfig:
    """Configuration for the multi-agent workflow"""
    openrouter_api_key: str
    openrouter_base_url: str
    gemini_api_key: str
    vector_store_path: str = "./chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieved_docs: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llama_model: str = "meta-llama/llama-3.3-70b-instruct:free"


config = WorkflowConfig(
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
    openrouter_base_url="https://openrouter.ai/api/v1",
    gemini_api_key=os.getenv("GEMINI_API_KEY", "")
)

class WorkflowState(TypedDict):
    """State that flows through the workflow"""
    query: str
    documents: List[Document]
    research_analysis: str
    synthesis_result: str
    metadata: Dict[str, Any]
    messages: List[BaseMessage]

class OpenRouterClient:
    """HTTP client for OpenRouter API"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/multi-agent-rag",
            "X-Title": "Multi-Agent RAG Workflow"
        }

    def chat_completion(self, messages: List[Dict], temperature: float = 0.1, 
                       max_tokens: int = 4000) -> str:
        """Make chat completion request"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenRouter API Error: {str(e)}"

class RAGSystem:
    """Retrieval system using HuggingFace embeddings"""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.vector_store = None

    def initialize_vector_store(self, documents: List[str] = None):
        """Initialize or load vector store"""
        if documents:
            docs = [Document(page_content=doc) for doc in documents]
            splits = self.text_splitter.split_documents(docs)
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.config.vector_store_path
            )
            print(f"✓ Vector store created with {len(splits)} document chunks")
        else:
            try:
                self.vector_store = Chroma(
                    persist_directory=self.config.vector_store_path,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"No existing vector store found: {e}")
                self.vector_store = Chroma(
                    persist_directory=self.config.vector_store_path,
                    embedding_function=self.embeddings
                )

    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents"""
        if not self.vector_store:
            return []

        k = k or self.config.max_retrieved_docs
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

class LlamaAgent:
    """Research Agent using Llama 3.3 8B"""

    def __init__(self, config: WorkflowConfig):
        self.client = OpenRouterClient(
            config.openrouter_api_key,
            config.openrouter_base_url,
            config.llama_model
        )

    def analyze_documents(self, query: str, documents: List[Document]) -> str:
        """Analyze documents and provide insights"""
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}"
                              for i, doc in enumerate(documents[:3])])

        messages = [
            {
                "role": "system",
                "content": """You are an expert research analyst. Analyze documents thoroughly and provide:
1. Key findings relevant to the query
2. Information quality assessment
3. Gaps and limitations
4. Contradictions if any
5. Specific insights
6. Confidence level (High/Medium/Low)

Be precise, objective, and structure your response clearly."""
            },
            {
                "role": "user",
                "content": f"""Query: "{query}"

Documents:
{context}

Provide comprehensive analysis with the 6 points mentioned in your system prompt."""
            }
        ]

        return self.client.chat_completion(messages)


class GeminiAgent:
    """Synthesis Agent using Gemini"""

    def __init__(self, config: WorkflowConfig):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=config.gemini_api_key,
            temperature=0.3
        )

    def synthesize_response(self, query: str, research_analysis: str,
                          documents: List[Document]) -> str:
        """Create final synthesized response"""
        doc_summaries = "\n".join([f"- {doc.page_content[:200]}..."
                                  for doc in documents[:3]])

        prompt = f"""Create a comprehensive response to the user's query based on the research analysis.

Query: "{query}"

Research Analysis:
{research_analysis}

Document Summaries:
{doc_summaries}

Provide:
1. Clear, direct answer
2. Supporting evidence
3. Context and background
4. Important caveats
5. Actionable insights

Structure with clear headings and make it engaging and accurate."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error in Gemini synthesis: {str(e)}"

def retrieve_documents_node(state: WorkflowState) -> WorkflowState:
    """Retrieve relevant documents"""
    rag_system = RAGSystem(config)
    rag_system.initialize_vector_store()

    documents = rag_system.retrieve_documents(state["query"])

    state["documents"] = documents
    state["metadata"]["retrieval_timestamp"] = datetime.now().isoformat()
    state["metadata"]["num_documents_retrieved"] = len(documents)

    return state


def llama_research_node(state: WorkflowState) -> WorkflowState:
    """Analyze documents with Llama"""
    llama_agent = LlamaAgent(config)

    analysis = llama_agent.analyze_documents(
        state["query"],
        state["documents"]
    )

    state["research_analysis"] = analysis
    state["metadata"]["research_timestamp"] = datetime.now().isoformat()
    state["messages"].append(AIMessage(content=f"Research Analysis: {analysis}"))

    return state


def gemini_synthesis_node(state: WorkflowState) -> WorkflowState:
    """Synthesize final response with Gemini"""
    gemini_agent = GeminiAgent(config)

    synthesis = gemini_agent.synthesize_response(
        state["query"],
        state["research_analysis"],
        state["documents"]
    )

    state["synthesis_result"] = synthesis
    state["metadata"]["synthesis_timestamp"] = datetime.now().isoformat()
    state["messages"].append(AIMessage(content=f"Final Response: {synthesis}"))

    return state


def should_continue_research(state: WorkflowState) -> str:
    """Decide if more research is needed"""
    if len(state["documents"]) == 0:
        return "insufficient_data"
    elif "Low" in state.get("research_analysis", ""):
        return "needs_more_research"
    else:
        return "proceed_to_synthesis"

def create_workflow() -> StateGraph:
    """Create the workflow graph"""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("retrieval", retrieve_documents_node)
    workflow.add_node("llama_research", llama_research_node)
    workflow.add_node("gemini_synthesis", gemini_synthesis_node)

    # Add edges
    workflow.add_edge("retrieval", "llama_research")
    workflow.add_conditional_edges(
        "llama_research",
        should_continue_research,
        {
            "insufficient_data": END,
            "needs_more_research": "retrieval",
            "proceed_to_synthesis": "gemini_synthesis"
        }
    )
    workflow.add_edge("gemini_synthesis", END)

    # Set entry point
    workflow.set_entry_point("retrieval")

    return workflow

class MultiAgentRAGWorkflow:
    """Main workflow interface"""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.workflow = create_workflow()
        self.app = self.workflow.compile()
        self.rag_system = RAGSystem(config)

    def add_documents(self, documents: List[str]):
        """Add documents to knowledge base"""
        self.rag_system.initialize_vector_store(documents)
        print(f"✓ Added {len(documents)} documents to knowledge base")
        print(f"✓ Using HuggingFace embeddings: {self.config.embedding_model}")

    async def run(self, query: str) -> Dict[str, Any]:
        """Run the workflow"""
        initial_state = {
            "query": query,
            "documents": [],
            "research_analysis": "",
            "synthesis_result": "",
            "metadata": {
                "workflow_start": datetime.now().isoformat(),
                "query": query,
                "embedding_model": self.config.embedding_model
            },
            "messages": [HumanMessage(content=query)]
        }

        try:
            final_state = await self.app.ainvoke(initial_state)
            final_state["metadata"]["workflow_end"] = datetime.now().isoformat()

            return {
                "answer": final_state["synthesis_result"],
                "research_analysis": final_state["research_analysis"],
                "num_documents_used": len(final_state["documents"]),
                "metadata": final_state["metadata"],
                "conversation_history": final_state["messages"]
            }
        except Exception as e:
            return {
                "error": f"Workflow execution failed: {str(e)}",
                "partial_state": initial_state
            }

    def run_sync(self, query: str) -> Dict[str, Any]:
        """Synchronous version"""
        return asyncio.run(self.run(query))

async def demo_workflow():
    """Demonstrate the workflow"""
    print("=== Multi-Agent RAG Workflow Demo ===\n")

    # Check API keys
    if not config.openrouter_api_key or not config.gemini_api_key:
        print("❌ Error: API keys not found in .env file")
        print("Please set OPENROUTER_API_KEY and GEMINI_API_KEY")
        return

    # Initialize workflow
    workflow = MultiAgentRAGWorkflow(config)

    # Sample documents
    sample_documents = [
        """
        Artificial Intelligence (AI) has revolutionized various industries in recent years. 
        Machine learning algorithms have become more sophisticated, enabling applications 
        in healthcare, finance, transportation, and education. Deep learning models, 
        particularly large language models, have shown remarkable capabilities in natural 
        language processing tasks.
        """,
        """
        The healthcare industry has seen significant adoption of AI technologies. 
        Medical imaging analysis, drug discovery, and personalized treatment plans 
        are areas where AI has made substantial contributions. AI-powered diagnostic 
        tools can now detect certain conditions with accuracy comparable to or 
        exceeding human specialists.
        """,
        """
        Ethical considerations in AI development have become increasingly important. 
        Issues such as bias in algorithms, privacy concerns, and the potential for 
        job displacement require careful consideration. Responsible AI development 
        practices are essential to ensure beneficial outcomes for society.
        """
    ]

    # Add documents
    workflow.add_documents(sample_documents)

    # Test queries
    queries = [
        "How is AI being used in healthcare?",
        "What are the main ethical concerns with AI development?",
        "What are the latest advances in machine learning?"
    ]

    print(f"✓ Using Llama model: {config.llama_model}")
    print(f"✓ Using OpenRouter endpoint: {config.openrouter_base_url}")
    print("-" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 50)

        result = await workflow.run(query)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"📄 Documents used: {result['num_documents_used']}")
            print(f"⏱️  Processing time: {result['metadata'].get('workflow_end', 'N/A')}")
            print(f"\n📝 Answer:\n{result['answer'][:500]}...")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_workflow())
import streamlit as st
import os
import logging
from datetime import datetime
from typing import Annotated, Literal, TypedDict, List, Dict, Any, Optional
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from langchain_core.prompts import ChatPromptTemplate
import time
import sqlite3
import pandas as pd

# === STREAMLIT CONFIG ===
st.set_page_config(
    page_title="SQL Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    .sql-query {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #5a6fd8;
    }
</style>
""", unsafe_allow_html=True)

# === LOGGING SETUP ===
@st.cache_resource
def get_logger(name: str) -> logging.Logger:
    """Create and return a logger that saves ALL log levels to file only."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        return logger
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = get_logger("SQLAgent_Streamlit")

# === STATE CLASS ===
class State(TypedDict):
    """Represents the state of our graph."""
    messages: Annotated[list[AnyMessage], add_messages]
    query_attempts: int
    final_answer: Optional[str]

# === SQL AGENT CLASS (Same as before but with progress tracking) ===
class SQLAgent:
    """SQL Agent that uses LangGraph to interact with a SQLite database."""
    
    def __init__(self, db_path: str, model_name: str = "llama-3.1-8b-instant", groq_api_key: Optional[str] = None):
        """Initialize the SQL Agent with a SQLite database connection and Groq LLM."""
        logger.info("Initializing SQLAgent...")
        self.connection_string = f"sqlite:///{db_path}"
        self.db = SQLDatabase.from_uri(self.connection_string)
        
        self.llm = ChatGroq(
            model=model_name,
            api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        
        self._setup_tools()
        self._setup_prompts()
        self._build_graph()
        logger.info("SQLAgent initialization completed.")

    def _setup_tools(self) -> None:
        """Set up the required tools for database interaction."""
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()
        
        self.list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        self.get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
        
        @tool
        def db_query_tool(query: str) -> str:
            """Execute a SQL query against the SQLite database and get back the result."""
            try:
                result = self.db.run_no_throw(query)
                if result is None or result == "" or result == []:
                    return "Query executed successfully but returned no results."
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        self.db_query_tool = db_query_tool

    def _setup_prompts(self) -> None:
        """Set up the system prompts for query generation and checking."""
        self.query_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQLite database assistant. Your task is to generate SQL queries to answer user questions.
                IMPORTANT RULES:
                1. Generate ONLY valid SQLite SELECT queries
                2. Use proper SQLite syntax (no MySQL/PostgreSQL specific functions)
                3. Be careful with table and column names - use exact names from the schema
                4. When in doubt about column names, use SELECT * to see all columns first
                5. Use LIMIT to prevent huge result sets
                6. Return ONLY the SQL query, nothing else"""),
            ("placeholder", "{messages}"),
        ])
        
        self.interpret_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst. Your job is to interpret SQL query results and provide clear answers.
                Given the query results, provide a clear, human-readable answer that directly addresses the user's question.
                Start your response with "Answer: " followed by your interpretation.
                If the results are empty, explain that no matching records were found.
                If there's an error, suggest what might be wrong and how to fix it."""),
            ("placeholder", "{messages}"),
        ])
        
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier. Determine if the user's message is:
                - A greeting only (e.g., 'hi', 'hello', 'good morning', 'how are you')
                - OR a real question about data (e.g., asking for email, orders, users)

                Respond ONLY with:
                - "greeting" → if it's small talk with no request
                - "query" → if there's any data request, even after a greeting"""),
            ("human", "{input}")
        ])

    def _create_tool_node_with_fallback(self, tools: list) -> RunnableWithFallbacks:
        """Create a tool node with error handling."""
        def handle_tool_error(state: Dict) -> Dict:
            error = state.get("error")
            tool_calls = state.get("messages", [])[-1].tool_calls if state.get("messages") else []
            return {
                "messages": [
                    ToolMessage(content=f"Error: {repr(error)}", tool_call_id=tc["id"])
                    for tc in tool_calls
                ]
            }
        return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")

    def _build_graph(self) -> None:
        """Build the LangGraph workflow."""
        workflow = StateGraph(State)

        def first_tool_call(state: State) -> Dict:
            return {
                "messages": [AIMessage(content="", tool_calls=[{"name": "sql_db_list_tables", "args": {}, "id": "tool_abcd123"}])],
                "query_attempts": 0,
                "final_answer": None
            }

        def model_get_schema(state: State) -> Dict:
            messages = state["messages"]
            chat_with_get_schema = self.llm.bind_tools([self.get_schema_tool])
            result = chat_with_get_schema.invoke(messages)
            return {"messages": [result]}

        def query_gen_node(state: State) -> Dict:
            messages = state["messages"]
            query_attempts = state.get("query_attempts", 0) + 1
            
            if query_attempts > 3:
                return {
                    "messages": [AIMessage(content="Unable to generate a working query after multiple attempts.")],
                    "query_attempts": query_attempts,
                    "final_answer": "Unable to generate a working query after multiple attempts."
                }
            
            query_response = (self.query_gen_prompt | self.llm).invoke({"messages": messages})
            return {"messages": [query_response], "query_attempts": query_attempts}

        def execute_query_node(state: State) -> Dict:
            messages = state["messages"]
            last_message = messages[-1]
            sql_query = last_message.content.strip()
            
            if "SELECT" in sql_query.upper():
                lines = sql_query.split('\n')
                for line in lines:
                    if 'SELECT' in line.upper():
                        sql_query = line.strip()
                        if sql_query.endswith('.'):
                            sql_query = sql_query[:-1]
                        break
            
            try:
                result = self.db.run_no_throw(sql_query)
                if result is None or result == "" or result == []:
                    content = "Query executed successfully but returned no results."
                else:
                    content = str(result)
                return {"messages": [ToolMessage(content=content, tool_call_id="manual_query_execution")]}
            except Exception as e:
                return {"messages": [ToolMessage(content=f"Error executing query: {str(e)}", tool_call_id="manual_query_execution")]}

        def interpret_results_node(state: State) -> Dict:
            messages = state["messages"]
            interpretation = (self.interpret_prompt | self.llm).invoke({"messages": messages})
            return {"messages": [interpretation], "final_answer": interpretation.content}

        def should_continue_after_query_gen(state: State) -> Literal[END, "execute_query"]:
            messages = state["messages"]
            if not messages:
                return END
            last_message = messages[-1]
            query_attempts = state.get("query_attempts", 0)
            if query_attempts > 3:
                return END
            if (hasattr(last_message, 'content') and last_message.content and 'SELECT' in last_message.content.upper()):
                return "execute_query"
            return END

        def should_continue_after_execution(state: State) -> Literal[END, "interpret_results", "query_gen"]:
            messages = state["messages"]
            if not messages:
                return END
            last_message = messages[-1]
            if (isinstance(last_message, ToolMessage) and last_message.content.startswith("Error")):
                query_attempts = state.get("query_attempts", 0)
                if query_attempts >= 3:
                    return END
                return "query_gen"
            if isinstance(last_message, ToolMessage):
                return "interpret_results"
            return END

        def should_continue_after_interpretation(state: State) -> Literal[END]:
            return END

        # Add nodes
        workflow.add_node("first_tool_call", first_tool_call)
        workflow.add_node("list_tables_tool", self._create_tool_node_with_fallback([self.list_tables_tool]))
        workflow.add_node("model_get_schema", model_get_schema)
        workflow.add_node("get_schema_tool", self._create_tool_node_with_fallback([self.get_schema_tool]))
        workflow.add_node("query_gen", query_gen_node)
        workflow.add_node("execute_query", execute_query_node)
        workflow.add_node("interpret_results", interpret_results_node)

        # Add edges
        workflow.add_edge(START, "first_tool_call")
        workflow.add_edge("first_tool_call", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "get_schema_tool")
        workflow.add_edge("get_schema_tool", "query_gen")
        workflow.add_conditional_edges("query_gen", should_continue_after_query_gen)
        workflow.add_conditional_edges("execute_query", should_continue_after_execution)
        workflow.add_conditional_edges("interpret_results", should_continue_after_interpretation)

        self.app = workflow.compile()

    def query(self, question: str, recursion_limit: int = 10) -> Dict[str, Any]:
        """Execute a query against the database using the agent."""
        try:
            intent_chain = self.intent_prompt | self.llm
            intent_response = intent_chain.invoke({"input": question})
            intent = intent_response.content.strip().lower()
            
            if intent == "greeting":
                return {"sql_query": None, "answer": "Hello! How can I assist you today?"}
        except Exception as e:
            logger.warning(f"Failed to classify intent: {e}")

        try:
            config = {"recursion_limit": recursion_limit}
            messages = self.app.invoke(
                {"messages": [HumanMessage(content=question)], "query_attempts": 0, "final_answer": None},
                config=config
            )
            
            final_sql_query = self._extract_final_sql_query(messages)
            final_answer = messages.get("final_answer")
            
            if not final_answer:
                last_message = messages["messages"][-1] if messages["messages"] else None
                if last_message and hasattr(last_message, 'content'):
                    final_answer = last_message.content
            
            return {"sql_query": final_sql_query, "answer": final_answer}
            
        except GraphRecursionError:
            return {
                "sql_query": None,
                "answer": "Sorry, I couldn't process your request. Please make sure your question is about retrieving data."
            }
        except Exception as e:
            logger.error(f"Error in query execution: {str(e)}")
            return {"sql_query": None, "answer": "Sorry, something went wrong. Please try again."}

    def _extract_final_sql_query(self, messages: Dict) -> Optional[str]:
        """Extract the final SQL query from the message history."""
        for msg in reversed(messages.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                content = msg.content
                if 'SELECT' in content.upper():
                    lines = content.split('\n')
                    for line in lines:
                        if 'SELECT' in line.upper():
                            query = line.strip()
                            if query.endswith('.'):
                                query = query[:-1]
                            return query
        return None

    def get_table_info(self) -> str:
        """Get information about all tables in the database."""
        try:
            return self.db.get_table_info()
        except Exception as e:
            return f"Error getting table information: {str(e)}"

    def list_tables(self) -> List[str]:
        """Get list of all table names in the database."""
        try:
            return self.db.get_usable_table_names()
        except Exception as e:
            return [f"Error: {str(e)}"]

# === UTILITY FUNCTIONS ===
def get_db_preview(db_path: str, table_name: str, limit: int = 5) -> pd.DataFrame:
    """Get a preview of the table data as DataFrame."""
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error previewing table: {str(e)}")
        return pd.DataFrame()

def validate_db_connection(db_path: str) -> bool:
    """Validate if the database file exists and is accessible."""
    try:
        if not os.path.exists(db_path):
            return False
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return len(tables) > 0
    except Exception:
        return False

# === STREAMLIT APP ===
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 SQL Agent - Database Query Assistant</h1>
        <p>Ask natural language questions about your database and get SQL queries with results!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False

    # === CONFIGURATION SECTION ===
    st.header("⚙️ Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.subheader("Database Setup")
        db_option = st.radio("Choose database source:", ["Upload SQLite File", "Enter File Path"])
        
        db_path = None
        if db_option == "Upload SQLite File":
            uploaded_file = st.file_uploader("Choose SQLite database file", type=['db', 'sqlite', 'sqlite3'])
            if uploaded_file is not None:
                temp_dir = "temp_db"
                os.makedirs(temp_dir, exist_ok=True)
                db_path = os.path.join(temp_dir, uploaded_file.name)
                with open(db_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Database uploaded: {uploaded_file.name}")
        else:
            db_path = st.text_input(
                "Database Path:", 
                value="database/final_ecommerce.db",
                help="Enter the full path to your SQLite database file"
            )

    with config_col2:
        st.subheader("API Configuration")
        groq_api_key = st.text_input(
            "Groq API Key:", 
            type="password", 
            value=os.getenv("GROQ_API_KEY", ""),
            help="Enter your Groq API key"
        )
        
        model_name = st.selectbox(
            "Model:", 
            ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
            help="Choose the language model to use"
        )
        
        recursion_limit = st.slider("Recursion Limit:", 5, 20, 15)

    # === VALIDATION AND INITIALIZATION ===
    if db_path and groq_api_key:
        if validate_db_connection(db_path):
            if st.button("🚀 Initialize SQL Agent", type="primary"):
                with st.spinner("Initializing SQL Agent..."):
                    try:
                        st.session_state.agent = SQLAgent(db_path, model_name, groq_api_key)
                        st.session_state.db_path = db_path
                        st.session_state.agent_initialized = True
                        st.success("✅ SQL Agent initialized successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error initializing agent: {str(e)}")
        else:
            st.error("❌ Invalid database file or path. Please check your database.")
    elif not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key.")
    elif not db_path:
        st.info("👆 Please select or upload your database file.")

    # === MAIN APPLICATION ===
    if st.session_state.agent_initialized:
        st.markdown("---")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Database Info", "📋 Query History"])
        
        with tab1:
            st.header("Chat with your Database")
            
            # Chat container
            chat_container = st.container()
            
            with chat_container:
                # Display chat messages
                for i, message in enumerate(st.session_state.messages):
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**Assistant:** {message['content']}")
                        if message.get("sql_query"):
                            st.code(message["sql_query"], language="sql")
                    st.markdown("---")
            
            # Chat input
            user_input = st.text_input("Ask a question about your database:", key="chat_input")
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button("Send", type="primary"):
                    if user_input:
                        # Add user message
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        
                        # Get agent response
                        with st.spinner("🧠 Thinking..."):
                            result = st.session_state.agent.query(user_input, recursion_limit)
                        
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": result["answer"],
                            "sql_query": result["sql_query"]
                        })
                        
                        st.rerun()
            
            with col2:
                if st.button("Clear Chat"):
                    st.session_state.messages = []
                    st.rerun()
        
        with tab2:
            st.header("Database Information")
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.subheader("📋 Tables")
                tables = st.session_state.agent.list_tables()
                if tables and not any("Error" in str(table) for table in tables):
                    for table in tables:
                        st.write(f"• **{table}**")
                else:
                    st.error("Could not load tables")
            
            with info_col2:
                st.subheader("👀 Table Preview")
                if 'tables' in locals() and tables:
                    selected_table = st.selectbox("Select table:", tables)
                    if selected_table and st.button("Load Preview"):
                        df = get_db_preview(st.session_state.db_path, selected_table)
                        if not df.empty:
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No data found in this table.")
            
            st.subheader("🏗️ Schema Information") 
            if st.button("Load Complete Schema"):
                with st.spinner("Loading schema..."):
                    schema_info = st.session_state.agent.get_table_info()
                    st.text_area("Schema Info:", value=schema_info, height=300)
        
        with tab3:
            st.header("Query History")
            if st.session_state.messages:
                for i, message in enumerate(st.session_state.messages):
                    if message["role"] == "assistant" and message.get("sql_query"):
                        with st.expander(f"Query {i//2 + 1}: {st.session_state.messages[i-1]['content'][:50]}..."):
                            st.code(message["sql_query"], language="sql")
                            st.write("**Result:**", message["content"])
            else:
                st.info("No queries executed yet.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="info-box warning-box">
        <strong>⚠️ Safety Note:</strong> This agent can only execute SELECT queries for safety. No data modification is allowed.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
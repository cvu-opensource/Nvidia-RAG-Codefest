"""
The centerpiece multi-agent service coordinating all from high above. Truly Chair Force.

First, initialize the relevant llm/embedder python objects, and any other configuration things.
"""
# commons?
import getpass, os, base64, requests
from typing import List, Optional, Annotated, Literal
from typing_extensions import TypedDict
from IPython.display import Image, display
from collections.abc import Iterable
from random import randint

# API packages
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File


# langshit
from langchain.text_splitter import CharacterTextSplitter

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langchain_core.documents import Document

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import InjectedState

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from infer_neo4j import GraphRAG

# Initialise objects for models used
main_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", base_url="http://10.149.8.40:8000/v1")

milvus_embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", base_url="http://10.149.8.40:8001/v1", truncate="END")
milvusDB_api_loc = "http://10.149.8.40:9998"

rerank_client = NVIDIARerank(model="nvidia/nv-rerankqa-mistral-4b-v3", base_url="http://10.149.8.40:8002")

neo4j_embedder = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://10.149.8.40:11434")
neo4j_llm = OllamaFunctions(model="llama3.1", temperature=0, format="json", base_url="http://10.149.8.40:11434")
neo4j_uri="neo4j://10.149.8.40:7687"
neo4j_username="neo4j"
neo4j_password="cringemfpassword"

graphRAG = GraphRAG(
    uri=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password,
    embedder=neo4j_embedder,
)
    
app = FastAPI()

# tvly_api_key = getpass.getpass("Enter your tvly API key: ")
# assert tvly_api_key.startswith("tvly-"), f"{tvly_api_key[:5]}... is not a valid key"
os.environ["TAVILY_API_KEY"] = "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"

class Conversation(TypedDict):
    """State representing the customer's conversation."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The `add_messages` annotation indicates to LangGraph
    # that state is updated by appending returned messages, not replacing
    # them.
    messages: Annotated[list, add_messages]

    # Flag indicating that the order is placed and completed.
    finished: bool
    
# The system instruction defines how the chatbot is expected to behave and includes
# rules for when to call different functions, as well as rules for the conversation, such
# as tone and what is permitted for discussion.
SYSTEM_INSTRUCTIONS = (
    "system",  # 'system' indicates the message is a system instruction.
    "You are a Legal Advice Chat Bot, you provide users with concise, accurate, and Singapore-specific legal information and guidance related to business operations, compliance, and regulations. "
    "A human will ask you about any questions they regarding the legal domain relating to business and you will answer any questions "
    "they have (and only about questions in the legal domain regarding business - no off topic discussions) "
    "Only provide legal information relevant to Singapore’s business laws, corporate regulations, and compliance requirements. "
    "Avoid discussing laws or business practices from other jurisdictions unless explicitly asked for comparison, and clarify that it is outside Singapore's context."
    "\n\n"
    "Provide general guidance on topics such as company incorporation, employment regulations, tax compliance, intellectual property (IP), contract law, and data protection in Singapore."
    "Avoid offering personalized legal advice, contract drafting, or reviews. Encourage users to consult qualified professionals for specific situations."
    "\n\n"
    "Business-specific topics include: Company incorporation and legal structures (e.g., Sole Proprietorship, LLP, Pte Ltd), "
    "Employment regulations under the Employment Act, "
    "Tax compliance (e.g., GST registration, corporate income tax), "
    "Licensing and permits for businesses, "
    "Data privacy laws, including compliance with the Personal Data Protection Act (PDPA), "
    "Commercial contract fundamentals and enforceability, "
    "and IP rights, including trademarks, copyrights, and patents in Singapore."
    "\n\n"
    "Use straightforward, business-friendly language. "
    "Explain legal concepts with examples or simplified analogies when possible. "
    "Provide links to official Singapore government resources (e.g., ACRA, IRAS, MOM) where applicable. "
    "\n\n"
    "Ensure responses are aligned with the latest Singaporean laws, regulations, and best practices. "
    "If unsure of current laws, advise users to verify with government authorities or legal experts. "
    "\n\n"
    "Do not provide guidance that could facilitate illegal activities or tax evasion. "
    "Avoid speculating on outcomes of legal disputes or offering advice that requires knowledge of specific business circumstances. "
    "\n\n"
    "Politely decline if a query falls outside Singapore’s legal context or business-related scope. Redirect users to appropriate resources or professionals. "
    "Clearly state when information is general and not a substitute for professional legal counsel. "
    "\n\n"
    'Examples of Accepted Queries: Topics the bot can assist with include "How do I register a private limited company in Singapore?" "What are the requirements for hiring foreign employees?" "Do I need to register for GST if my business turnover exceeds $1 million?" and "What steps should I take to trademark my business logo?"'
    'Examples of Declined Queries: Requests such as “Can you draft a shareholders’ agreement for me?” or “What are corporate tax laws in Hong Kong?” will be declined politely, with users advised to consult professionals or explore suitable external resources. Similarly, unethical queries, such as “How can I avoid CPF contributions for employees?” will be met with a clear explanation of the legal obligations in Singapore.'
    "\n\n"
    "Remember to provide citation to any information you provide to the user. "
    "If the information you retrieve is old, please fact chat the information with more recent sources to ensure the information is accurate. "
    "If the word 'image' is in the text, it may imply that part of the text may be descriptions of an image. Respond with the context of the image."
)

# This is the message with which the system opens the conversation.
WELCOME_MSG = "Welcome to the ASPER LOVERS LEGAL Bot. Type `q` to quit. How may I serve you today?"

class Tools:
    """
    Class to house tool related attributes and functions.
    """
    def __init__(self,
        milvus_embedder,
        rerank_client,
        milvusDB_api_loc,
        neo4j_embedder,
        neo4j_llm,
        graphRAG,
        tavily = TavilySearchResults(max_results=3),
        ):
        self.milvus_embedder = milvus_embedder
        self.rerank_client = rerank_client
        self.milvusDB_api_loc = milvusDB_api_loc
        self.neo4j_embedder = neo4j_embedder
        self.neo4j_llm = neo4j_llm
        self.graphRAG = graphRAG
        self.tavily = tavily
        
    @staticmethod
    def query_milvus(query, embedded_query, data_api_url):
        """
        Do a similarity search sgt!
        """
        # Crafting request to milvus to get k(x) result. 
        # Output(list of results) = ["result1", "result2", ...]
        milvus_response = requests.post(                    
            self.milvusDB_api_loc + "/search",
            json={"question": query, "embedded_question": embedded_query, "top_k":1},
            #files={"image": uploaded_image.getvalue() if uploaded_image else None}
            files = None
            ).json()

        retrieved_results = milvus_response["results"]
        # print(retrieved_results)

        if milvus_response:
            # prepare documents from vectorstore response
            '''
            documents = [
                {
                    "content": passage.page_content,
                    "metadata": passage.metadata
                }
                for passage in milvus_response
            ]
            '''

            documents = [Document(page_content=key)
                for key, value in retrieved_results.items()
            ]
        else:
            documents = []
            print("query_milvus method returned empty list. Curious.")
            
        return documents
    
    @staticmethod
    def rerank(reranker_client, query, documents):
        """
        Do a rerank sgt!
        """
        rerank_response = sorted(reranker_client.compress_documents(query=query, documents=documents), key=lambda x:x.metadata['relevance_score'], reverse=True)

        if rerank_response:
            top_result = rerank_response[0]
            reranker_output = f"Top Result: {top_result.page_content}"
        else:
            reranker_output = "No relevant results found for the query"
        
        return reranker_output
        
    
    @tool   
    def rag_from_database(self, query: str) -> str:
        """
        A Retrieval-Augmented Generation (RAG) tool perform the following:
        1. Query MilvusDB for relevant chunks (VectorDB)
        2. Rerank queries for result relevance
        3. Query Neo4j for entity/concept/object relations
        4. Return combined response of 2 and 3 to model.

        Parameters:
            query (str): The user's input query.

        Returns:
            str: The most relevant information retrieved vector and graph RAG approach.

        """
        # Step 1: Embed the query, and retrieve from milvus
        embedded_query = self.milvus_embedder.embed_query(query)
        documents = self.query_milvus(query, embedded_query, self.DATA_API)
        
        # Step 2: Perform reranking and sorting the results with the scores in decending order.
        if documents:
            vectorRAG_output = self.rerank(self.rerank_client, query, documents)
        else:
            vectorRAG_output = ""
            
        # Step 3: Query Neo4j for relations.
        relations = self.graphRAG.retrieve(self.neo4j_llm, query)
        # print("NEO4J RELATIONS", relations)

        final_data = f"""
            Graph data: {relations}
            Vector data:{vectorRAG_output}
            """
        
        return final_data
    
    
class Agency:
    """
    Main class that controls the logic of all agents. An agency of agents. Very secret.
    If you are reading this and you are not an agent, you are going to die in 5 hours. Enjoy!
    """
    def __init__(self,
        milvus_embedder,
        rerank_client,
        milvusDB_api_loc,
        neo4j_embedder,
        neo4j_llm,
        graphRAG,
        main_llm,
    ):
        tools_collection = Tools(
            milvus_embedder,
            rerank_client,
            milvusDB_api_loc,
            neo4j_embedder,
            neo4j_llm,
            graphRAG
        )
        used_tools = [tools_collection.rag_from_database, tools_collection.tavily]
        self.tool_node = ToolNode(used_tools)
        self.llm_with_tools = main_llm.bind_tools(used_tools)
        
        self.graph_builder = StateGraph(Conversation)

        # Add the nodes, including the new tool_node.
        self.graph_builder.add_node("chatbot", self.chatbot_with_tools)
        self.graph_builder.add_node("human", self.human_node)
        self.graph_builder.add_node("tools", self.tool_node)

        # Chatbot may go to tools, or human.
        self.graph_builder.add_conditional_edges("chatbot", self.maybe_route_to_tools)
        # Human may go back to chatbot, or exit.
        self.graph_builder.add_conditional_edges("human", self.maybe_exit_human_node)

        # Tools always route back to chat afterwards.
        self.graph_builder.add_edge("tools", "chatbot")

        self.graph_builder.add_edge(START, "human")
        self.graph_builder.add_edge(END, 'end')
        self.graph_executor = self.graph_builder.compile()

        # Image(test_graphs.get_graph().draw_mermaid_png())
        
    
    def invoke_graph(self,
        text
    ):
        cope = {'text': text}
        state = self.graph_executor.invoke({"messages": [text]}, cope) # AHAHAHAHAH
        return state['messages'][-1]
    
    
    # ENTRYPOINT TO GRAPH EXECUTOR
    def human_node(self, state: Conversation, config) -> Conversation:
        """Display the last model message to the user, and receive the user's input."""
        state["messages"] = Annotated[list, add_messages]
        user_input = config['metadata']['text']
        return state | {"messages": [("user", user_input)]}

    def chatbot_with_tools(self, state: Conversation) -> Conversation:
        """The chatbot with tools. A simple wrapper around the model's own chat interface."""
        defaults = {"order": [], "finished": False}
        
        # very first return we give to the user
        output = self.llm_with_tools.invoke([SYSTEM_INSTRUCTIONS] + state["messages"])
        print("chatbot_with_tools output", output)

        # Set up some defaults if not already set, then pass through the provided state,
        # overriding only the "messages" field.
        return defaults | state | {"messages": [output]}

    def maybe_exit_human_node(self, state: Conversation) -> Literal["chatbot", "__end__"]:
        """Route to the chatbot, unless it looks like the user is exiting."""
        print("State from human node:", state)
        # print("msgs", msgs := state.get("messages", []))
        return "chatbot"

    def maybe_route_to_tools(self, state: Conversation) -> Literal["tools", "human"]:
        """Route between human or tool nodes, depending if a tool call is made."""
        if not (msgs := state.get("messages", [])):
            raise ValueError(f"No messages found when parsing state: {state}")

        # Only route based on the last message.
        msg = msgs[-1]
        #print(dir(msg))
        print("toolies", msg.tool_calls)
        messages = state.get("messages")
        ai_msg = messages[-1]
        print(ai_msg)
        print(dir(ai_msg))
        # if no AIMessage inside, forcibly call chatbot to run.
        if not any([isinstance(msg, AIMessage) for msg in messages]):
            return "chatbot"
        # When the chatbot returns tool_calls, route to the "tools" node.
        elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            return "tools"
        else:
            return "end"
        
agencie = Agency(
    milvus_embedder,
    rerank_client,
    milvusDB_api_loc,
    neo4j_embedder,
    neo4j_llm,
    graphRAG,
    main_llm,
)

class InvokeRequest(BaseModel):
    query: str
    vlm_context: str
    history: str

@app.post("/invoke")
async def execute_state_graph(
    request: InvokeRequest
):
    json = request.dict()
    history = json['history']
    vlm_context = json['vlm_context']
    query = json['query']
    text = history + f"The following comes from an image the user input: {vlm_context}" + query
    print("texties", text)
    yapping = agencie.invoke_graph(text)
    print(yapping)
    return yapping
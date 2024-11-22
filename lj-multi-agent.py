import getpass, os, base64, requests

from IPython.display import Image, display

import uvicorn
from fastapi import FastAPI

from pydantic import BaseModel

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from collections.abc import Iterable
from random import randint
from langgraph.prebuilt import InjectedState
from langchain_core.messages.tool import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

from component_notebooks.agents.infer_neo4j import GraphRAG

from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions

# tvly_api_key = getpass.getpass("Enter your tvly API key: ")
# assert tvly_api_key.startswith("tvly-"), f"{tvly_api_key[:5]}... is not a valid key"
os.environ["TAVILY_API_KEY"] = "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"

app = FastAPI()

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
)

# This is the message with which the system opens the conversation.
WELCOME_MSG = "Welcome to the ASPER LOVERS LEGAL Bot. Type `q` to quit. How may I serve you today?"



core_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", base_url="http://10.149.8.40:8000/v1")


def human_node(state: Conversation) -> Conversation:
    """Display the last model message to the user, and receive the user's input."""
    last_msg = state["messages"][-1]
    print("Model:", last_msg.content)

    user_input = input("User: ")

    # If it looks like the user is trying to quit, flag the conversation
    # as over.
    if user_input in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True

    return state | {"messages": [("user", user_input)]}

def chatbot_with_tools(state: Conversation) -> Conversation:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    defaults = {"order": [], "finished": False}

    if state["messages"]:
        new_output = llm_with_tools.invoke([SYSTEM_INSTRUCTIONS] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)

    # Set up some defaults if not already set, then pass through the provided state,
    # overriding only the "messages" field.
    return defaults | state | {"messages": [new_output]}

def maybe_exit_human_node(state: Conversation) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    print("State:", state)
    # print("msgs", msgs := state.get("messages", []))
    if state.get("finished", False):
        return END
    else:
        return "chatbot"
    
def maybe_route_to_tools(state: Conversation) -> Literal["tools", "human"]:
    """Route between human or tool nodes, depending if a tool call is made."""
    # print("State:", state)
    # print("msgs", msgs := state.get("messages", []))
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    # Only route based on the last message.
    msg = msgs[-1]
    print(dir(msg))
    print("toolies", msg.tool_calls)
    # When the chatbot returns tool_calls, route to the "tools" node.
    if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        return "tools"
    else:
        return "human"

    
class Tools:
    """
    Class to house tool related attributes and functions.
    """
    def __init__(self):
        self.milvus_embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", base_url="http://10.149.8.40:8001/v1", truncate="NONE")
    
        # Specifying reranking model
        self.rerank_client = NVIDIARerank(model="nvidia/nv-rerankqa-mistral-4b-v3", base_url="http://10.149.8.40:8002")
        
        self.DATA_API = "http://10.149.8.40:9998"
        
        self.neo4j_embedder = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://10.149.8.40:11434")
        self.neo4j_llm = OllamaFunctions(model="llama3.1", temperature=0, format="json", base_url="http://10.149.8.40:11434") 
        
        self.graphRAG = GraphRAG(
            uri="neo4j://10.149.8.40:7687",
            username="neo4j",
            password="cringemfpassword",
            embedder=self.neo4j_embedder
        )
        
    @staticmethod
    def query_milvus(query, embedded_query, data_api_url):
        """
        Do a similarity search sgt!
        """
        # Crafting request to milvus to get k(x) result. 
        # Output(list of results) = ["result1", "result2", ...]
        milvus_response = requests.post(                    
            data_api_url + "/search",
            json={"question": query, "embedded_question": embedded_query, "top_k":50},
            #files={"image": uploaded_image.getvalue() if uploaded_image else None}
            files = None
            ).json()

        retrieved_results = milvus_response["results"]

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
            
            '''
            documents = [key
                for key, value in retrieved_results.items()
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
        # print(query)
        # print(documents)
        thing = reranker_client.compress_documents(query=query, documents=documents)
        #print(thing)
        rerank_response = sorted(reranker_client.compress_documents(query=query, documents=thing), key=lambda x:x.metadata['relevance_score'], reverse=True)
        print(rerank_response)
        

        # Select top result (returns to model top result and cite)
        '''
        if rerank_response:
            top_result = rerank_response[:10]
            # reranker_output = f"Top Result: {top_result.page_content}\n\nCite: {top_result.metadata}"
            reranker_output = f"Top Result: {top_result.page_content}"
        else:
            reranker_output = "No relevant results found for the query"
        '''
            
        if rerank_response:
            top_3_results = rerank_response[:3]
            # reranker_output = f"Top Result: {top_result.page_content}\n\nCite: {top_result.metadata}"
            formatted_3 = [top_result.page_content for top_result in top_3_results]
            reranker_output = f"Top 3 Result: {formatted_3}"
        else:
            reranker_output = "No relevant results found for the query"
        
        return reranker_output
        
    
    # @tool   
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
        
        # This should be able to be parallelized. Use threading/multiprocess later?

        # Step 1: Embed the query, and retrieve from milvus
        embedded_query = self.milvus_embedder.embed_query(query)
        
        documents = self.query_milvus(query, embedded_query, self.DATA_API)
        
        #documents = [doc.page_content for doc in pre_documents]
        #print(documents)
        
        # Step 2: Perform reranking and sorting the results with the scores in decending order.
        #print(query)
        #print(documents)
        
        if documents:
            vectorRAG_output = self.rerank(self.rerank_client, query, documents)
            #print(123, vectorRAG_output, '\n')
            
        # Step 3: Query Neo4j for relations.
        relations = self.graphRAG.retrieve(self.neo4j_llm, query)
        # print("NEO4J RELATIONS", relations)

        final_data = f"""
            Graph data: {relations}
            Vector data:{vectorRAG_output}
            """

        return final_data

query = f"What are the requirements for registering a business in Singapore?"
query2 = f"How often must a company file annual returns with ACRA?"
query3 = f"What are the minimum leave entitlements for full-time workers in Singapore?"
query4 = f"How does the Employment of Foreign Manpower Act apply to SMEs in Singapore?"
query5 = "What forms of intellectual property protection are available to businesses in Singapore?"
query6 = "How can I trademark a logo in Singapore?"
query7 = "What is the Annual Value threshold for property tax assessment in this year?"
query8: "Who is considered a sanction recipient and how does it impact tax obligations?"
query9 = "What is the Workfare Income Supplement (WIS) Program for SMEs?"
query10 = "How can I apply for professional fee relief under the Community Development Co-operative (CDCC) scheme?"

# db_tools = Tools()
# test = db_tools.rag_from_database(query10)
# print(test)

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
    db_tools = Tools()
    yapping = db_tools.rag_from_database(text)
    yapping = core_llm.invoke([SYSTEM_INSTRUCTIONS] + [f"This is the user query: {text}"] + [f"This is additional context: {yapping}"] + ['You must format your response in a comprehensive summary of the query.'])
    print(yapping)
    return yapping
    

        
    
#    print("RAG from database tool called! Retrieving relevant items from milvus and neo4j...") 
#    PLACEHOLDER = """Everything is Illegal"""

#    return PLACEHOLDER


# Define the tools and create a "tools" node.

# tools = [db_tools.rag_from_database, TavilySearchResults(max_results=3)]
# tool_node = ToolNode(tools)

# # Attach the tools to the model so that it knows what it can call.
# llm_with_tools = llm.bind_tools(tools)



# graph_builder = StateGraph(Conversation)

# # Add the nodes, including the new tool_node.
# graph_builder.add_node("chatbot", chatbot_with_tools)
# graph_builder.add_node("human", human_node)
# graph_builder.add_node("tools", tool_node)

# # Chatbot may go to tools, or human.
# graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# # Human may go back to chatbot, or exit.
# graph_builder.add_conditional_edges("human", maybe_exit_human_node)

# # Tools always route back to chat afterwards.
# graph_builder.add_edge("tools", "chatbot")

# graph_builder.add_edge(START, "chatbot")
# test_graphs = graph_builder.compile()

# Image(test_graphs.get_graph().draw_mermaid_png())



# state = test_graphs.invoke({"messages": []})
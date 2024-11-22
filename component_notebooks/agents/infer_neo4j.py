# normal-ish packages
from pydantic import BaseModel, Field

# neo4gay
from neo4j import GraphDatabase, Driver

# cancerous langchain stuffs
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from langchain.output_parsers import PydanticOutputParser

# temp, please remove
from langchain_community.embeddings import OllamaEmbeddings

class Entities(BaseModel):
    """Identifying information about organizations, business or person entities that appear in the text"""

    names: list[str] = Field(
        ...,
        description="All the organizations, business or person entities that appear in the text.",
    )

class GraphRAG:
    """
    Generic class meant to communicate with neo4j database. 
    Note that while embedder is fixed, llm is variable. This is because the llm parses the
    prompt recieved during inference, but the embedder is the true one retrieving things, 
    much like classic VectorRAG. 
    """
    
    def __init__(self,
        uri="neo4j://10.149.8.40:7687",
        username="neo4j",
        password="cringemfpassword",
        embedder=OllamaEmbeddings(model="mxbai-embed-large")
    ):

        self.vector_index = Neo4jVector.from_existing_graph(
            embedder,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
            url=uri,
            username=username,
            password=password,
        )
        self.vector_retriever = self.vector_index.as_retriever()

        self.driver = GraphDatabase.driver(
            uri=uri,
            auth = (username,password)
        )
        
        self.graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password,
        )
    
    @staticmethod
    def create_fulltext_index(tx):
        query = '''
        CREATE FULLTEXT INDEX `fulltext_entity_id` 
        FOR (n:__Entity__) 
        ON EACH [n.id];
        '''
        tx.run(query)
        
    @staticmethod
    def generate_full_text_query(input: str) -> str:
        words = [el for el in remove_lucene_chars(input).split() if el]
        if not words:
            return ""
        full_text_query = " AND ".join([f"{word}~2" for word in words])
        # print(f"Generated Query: {full_text_query}")
        return full_text_query.strip()
        
    
    def retrieve(self, llm, query):
        """
        Main function to be called. Note that this code is still highly unstable; 
        from my testing as of 22:59 21/10/24, if the prompt is not a question, some 
        tool_calls error appears. I believe that this is an easy fix, but I have 
        neither the time nor the energy as of writing this documentation to fix it. 
        God help me.
        
        Args:
            - llm: Ideally OllamaFunctions class, but you are free to try anything, 
                with zero wwarranty of course.
            - prompt: You know, the prompt. If its not a question things can get fucked?
            
        Returns:
            - result: String.
        """
        # llm_with_tools = llm.bind_tools([placeholder, placeholder, placeholder])
        
        # prompt = "The following is a question. You do not need to call any tools." + prompt
        
        parser = PydanticOutputParser(pydantic_object=Entities)
        result = ""
        prompt = PromptTemplate(
            template="Answer the user query. Extract all relevant entities and information related to them. \n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # print(chain.invoke("What is IRAS and what are its relations?"))
        try:
            print("PROMPT NEO4J LLM IS RECIEVING", prompt)
            entity_chain = prompt | llm.with_structured_output(Entities) 
            entities = entity_chain.invoke({
                "query": query
            })
            for entity in entities.names:
                print("ENTITY", entity)
                response = self.graph.query(
                    """
                    CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
                    YIELD node,score
                    CALL {
                      WITH node
                      MATCH (node)-[r:!MENTIONS]->(neighbor)
                      RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                      UNION ALL
                      WITH node
                      MATCH (node)<-[r:!MENTIONS]-(neighbor)
                      RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                    }
                    RETURN output LIMIT 50
                    """,
                    {"query": entity},
                    )
                print(response)
                result += "\n".join([el['output'] for el in response])
                
        except Exception as e:
            print(f"Caught exception in retrieve method of GraphRAG class! Exception is {e}")
            print("If above error is something related to tool_calls in an AIMessage, fix should be easy if you dig into it. From what I can tell, it is related to how the llm cannot interpret the prompt as a question. until then, simply return empty string.")
            raise e
            
        return result
    
    
    
    
# graphie = GraphRAG()
# test = db_tools.rag_from_database("What is the progressive wage credit scheme?")

#sample use: Initialize the class, define your llm, then you are free to call retrieve and recieve relationships.
# keep the uri and creds. 
# neo4j_graph = GraphRAG(
#     uri="neo4j://10.149.8.40:7687",
#     username="neo4j",
#     password="cringemfpassword",
#     embedder=OllamaEmbeddings(model="mxbai-embed-large") # make sure the ollama container has pulled mxbai-embed-large
# )

# llm = OllamaFunctions(model="llama3.1", temperature=0, format="json") # ollama docker instance running elsewhere, make sure its up. Should be able to auto-detect llama instance
# probably if its running at the default port on host machine.

# answer = neo4j_graph.retrieve(llm, "What is the Employment Act related to? Who does it affect?") # if this isnt a question, could possibly fuckup. Idk. up
# to yall to test and find out.


# sample code for full retrieval below: arbitrage vector_data and vector_retriever with milvus DB's. Although fyi, i ran embedding for neo4j too, so we can maybe count on it to if you want extra boost in context?

# def full_retriever(question: str):
#     graph_data = neo4j_graph.retrieve(llm, question)
#     vector_data = [el.page_content for el in neo4j_graph.vector_retriever.invoke(question)]
#     final_data = f"""Graph data:
# {graph_data}
# vector data:
# {"#Document ". join(vector_data)}
#     """
#     return final_data


# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# Use natural language and be concise.
# Answer:"""
# prompt = ChatPromptTemplate.from_template(template)

# chain = (
#         {
#             "context": full_retriever,
#             "question": RunnablePassthrough(),
#         }
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# print(chain.invoke("What is the Employment Act related to? Who does it affect?"))
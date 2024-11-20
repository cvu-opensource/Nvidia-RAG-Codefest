# General packages 
import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

# MilvusDB packages
import hashlib
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

class MilvusDB:
    """
    Not-so-masterfully handles the vector database using Milvus and all DB related functionality.
    """

    def __init__(self, uri):
        self.uri = uri
        self.client = None
        self.collection_name = None

        self.similarity_threshold = 0.99  # Similarity between new input data and existing data
        self.set_up_db()

    def set_up_db(self):
        """
        Starts up connection to Weaviate DB and creates schema if not already created

        Parameters: None

        Output:     None
        """
        self.client = MilvusClient(uri=self.uri)

    def load_collection(self):
        """
        Loads collection into RAM for faster retrieval

        Parameters: None

        Output:     None
        """
        self.client.load_collection(
            collection_name=self.collection_name,
            replica_number=1 # Number of replicas to create on query nodes. Max value is 1 for Milvus Standalone, and no greater than `queryNode.replicas` for Milvus Cluster.
        )

    def release_collection(self):
        """
        Releases the collection from memory to save memory usage

        Parameters: None

        Output:     None
        """
        self.client.release_collection(
            collection_name=self.collection_name
        )

    def create_collection(self, collection_name, dimensions):
        """
        Creates a new collection in the DB

        Parameters: 
            - collection_name:  Name of collection to make
            - dimensions:       Number of dimensions for vector data

        Output:     None
        """     
        # Defines a schema to follow when creating the DB
        id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True)
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimensions)
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2500)  # Use VARCHAR for string types

        # Metadata
        source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512)  
        date_added_field = FieldSchema(name="date_added", dtype=DataType.VARCHAR, max_length=50)
        category_field = FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100)
        relevancy_field = FieldSchema(name="relevancy", dtype=DataType.FLOAT)

        schema = CollectionSchema(fields=[id_field, embedding_field, text_field, source_field, date_added_field, category_field, relevancy_field])

        # Creates the collection
        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimensions,
            schema=schema,
            metric_type="COSINE",             
            consistency_level="Strong",     
        )
        self.collection_name = collection_name

        # Creates an index for more efficient similarity search later on based on the metric_type and index_type
        self.index_params = MilvusClient.prepare_index_params()
        self.index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="embedding_index",
            params={ "nlist": 128 }
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=self.index_params,
            sync=False
        )

    def insert_data(self, original, embedded, metadata=None, batch_size=128):
        """
        Adds document and embedding object pairs to the DB collection if not alreadt inside 

        Parameters:
            - original:     Original documents
            - embedded:     Embedded documents
            - metadata:     List of dictionaries containing metadata for each document
            - batch_size:   Number of records per batch

        Output: None
        """
        self.load_collection()
        
        data_batch = []
        print(len(original), len(embedded), len(metadata))
        for i, embedded_line in enumerate(tqdm(embedded, desc="Inserting data...")):
            unique_id = self.generate_unique_id(embedded_line)            
            if self.check_for_similar_vectors(embedded_line) and self.check_for_similar_ids(unique_id):  # Update existing document if ID is found
                self.update_existing_document(unique_id, embedded_line, original[i])
            else:
                categories = self.generate_data_categories(original[i])
                data_batch.append(
                    {
                        "id": unique_id, 
                        "embedding": embedded_line, 
                        "text": original[i], 
                        "source": metadata[i].get("source", "unknown") if metadata else 'None',
                        "date_added": metadata[i].get("date_added", "unknown") if metadata else 'None',
                        "category": categories,
                        "relevancy": 1.0
                    }
                )

            if len(data_batch) >= batch_size:  # Inserts data batch when exceeding max batch size
                self.client.insert(collection_name=self.collection_name, data=data_batch)
                data_batch = []

        if data_batch:
            self.client.insert(collection_name=self.collection_name, data=data_batch)

        self.release_collection()

    def update_existing_document(self, doc_id, embedding, text, relevancy=1.0):
        """
        Updates an existing document in the collection by replacing its embedding and text

        Parameters:
            - doc_id:       Unique ID of the document
            - embedding:    New embedding vector for the document
            - text:         New text for the document

        Output: None
        """
        self.client.delete(collection_name=self.collection_name, filter=f"id == '{doc_id}'")
        
        updated_data = [{"id": doc_id, "embedding": embedding, "text": text, "source": "None", "date_added":'None', "category":"None", "relevancy":relevancy}]
        self.client.insert(collection_name=self.collection_name, data=updated_data)

    def update_document_scores(self, doc_id, feedback):
        """
        Updates the relevancy score of certain pieces of data based on feedback from users

        Parameters:
            - doc_id:       Unique ID of the document
            - feedback:     Feedback given by user

        Output: None
        """
        print(self.client.search(collection_name=self.collection_name, filter=f"id == '{doc_id}'", output_fields=["text", "source", "date_added", "category", "relevancy"])[0])
        embeddding, text, score = self.client.search(collection_name=self.collection_name, filter=f"id == '{doc_id}'", output_fields=["text", "source", "date_added", "category", "relevancy"])[0]
        
        # Update score based on feedback
        if feedback == "Very Relevant": score += 0.2
        elif feedback == "Not Relevant": score -= 0.1 
        else: score += 0.1

        self.update_existing_document(doc_id, embeddding, text, score)

    def generate_data_categories(self, text):
        """
        Generates tags associated with a piece of text of question

        Parameters:
            - data:         Text data to categorise

        Returns:
            - categories:   List of categories relevant to data
        """
        categories = llm.invoke(f" \
            You are an expert in Singapore's corporate laws and regulations. Your task is to analyze the given text or question, and assign it appropriate tags from the list below. Tags should capture the core topics, entities, and relevant categories mentioned in the text or question given. \
            Only return 3 words of the the top 3 most relevant tags from the list below, without giving any other comments. \
            List of Tags: \
            - Sole Proprietorship, Partnership, Company, Corporation \
            - Business Registration, Taxation, Licensing, Employment Laws, Compliance, Intellectual Property, Contracts \
            - Setup, Operations, Reporting, Closure \
            - Healthcare, Retail, Finance, Technology, Food \
            - Forms, Policies, Guidelines, FAQs \
            - Compliance Issues, Tax Problems, Employment Disputes, Licensing Delays \
            - Singapore, International Trade \
            Input Text: {text} \
            Output Format: Comma-separated list of tags \
        ").content.strip().lower()
        categories = min(categories[0:len(categories)],  categories[0:100])
        return categories

    def generate_unique_id(self, data):
        """
        Generates a unique hash ID based on the vector or text data

        Parameters:
            - data: The vector or text data used to generate the hash

        Returns:
            - id:   A unique hash ID as a string
        """
        data_str = str(data)
        unique_id = hashlib.sha256(data_str.encode()).hexdigest()
        return unique_id

    def check_for_similar_vectors(self, embedding, top_k=5):
        """
        Checks the DB for vectors that are similar to the input embedding (based on distance metric like cosine similarity or Euclidean distance)

        Parameters:
            - embedding:    Embedded documents
            - top_k:        Top K number of documents that are similar to the input embedding

        Output: 
            - check:        True or False value of whether a similar vector has been found
        """
        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.client.search(
                collection_name=self.collection_name,
                data=[embedding],
                anns_field="embedding",  # Adjust field name based on your Milvus schema
                search_params=search_params,
                limit=top_k
            )
            for result in results:
                for vector in result:
                    if vector['distance'] >= self.similarity_threshold:
                        return True  
            return False
        except Exception as e:
            print(f"Error checking if vector exists: {e}")
            return False

    def check_for_similar_ids(self, id):
        """
        Checks the DB for ids that are similar to the new one created

        Parameters:
            - id:       Unique id generated for new row

        Output: 
            - check:    True or False value of whether a similar vector has been found
        """
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f"id == '{id}'",  
                output_fields=["id"], limit=1000 
            )
            return True if results else False
        except Exception as e:
            print(f"Error checking if ID exists: {e}")
            return False

    def retrieve_data(self, embedded_question, question, top_k=5, scaling_factor=1.1):
        """
        Retrieves vector data from DB based on embedded question

        Parameters:
            - question:             Question as a string
            - embedded_question:    Embedded question as a vector

        Output:
            - search_res:           Results of vector retrieval
        """
        self.load_collection()

        category_filter = " or ".join([f'category like "%{category}%"' for category in self.generate_data_categories(question).replace(' ', '').split(',')])

        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[embedded_question],  
            limit=top_k, 
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            filter=category_filter, 
            output_fields=["text", "source", "date_added", "category"]
        )[0]

        distances = [res['distance'] for res in search_res]
        avg_distance = sum(distances) / len(distances) if distances else 0
        adaptive_threshold = avg_distance * scaling_factor  # Getting a dynamic threshold based on average distance of top-K results to improve relevance filtering

        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[embedded_question],  
            limit=20, 
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            filter=category_filter, 
            output_fields=["text", "source", "date_added", "category", "relevancy"]
        )[0]

        filtered_res = [res for res in search_res if res['distance'] > adaptive_threshold]
        self.release_collection()
        return filtered_res
    
    def get_all_records(self, limit=10000):
        """
        Retrieves all rows from the Milvus collection.
        
        Parameters:
            - limit: The maximum number of rows to fetch in one query. Adjust as needed.
        
        Returns:
            - List of all records in the collection.
        """
        self.load_collection()

        try:
            # Query all records
            all_records = self.client.query(
                collection_name=self.collection_name,
                output_fields=["id", "text", "source", "date_added", "category"],
                limit=limit  # Adjust limit as needed
            )
            return all_records
        except Exception as e:
            print(f"Error retrieving records: {e}")
            return []
        finally:
            self.release_collection()
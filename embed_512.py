from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", base_url="http://localhost:8001/v1", truncate="END")

# data handler class
import os
import requests
import pandas as pd
import urllib.parse  # To handle URL joining
import fitz

import numpy as np
from tqdm import tqdm
from io import StringIO
from bs4 import BeautifulSoup, SoupStrainer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, DataFrameLoader, CSVLoader, UnstructuredTSVLoader

class DataHandler:
    """
    Masterfully handles data scraping, preprocessing, and other data-related functionalities in this notebook.
    """

    def __init__(self, embedder, csv_path="./data/csv", pdf_path="./data/pdf"):
        # Paths and storage
        self.csv_path = csv_path
        self.pdf_path = pdf_path
        self.visited_urls = set()

        # Data containers
        self.raw_data = []
        self.all_data = []
        self.textual_data = []
        self.textual_metadata = []
        self.tabular_data = []
        self.tabular_metadata = []
        self.visual_data = []
        self.visual_metadata = []
        
        # Final data containers
        self.embedded_data = []
        self.metadata = []

        # Text splitter
        self.text_splitter = CharacterTextSplitter(chunk_size=2048, separator=" ", chunk_overlap=64)

        # Ensure directories exist
        os.makedirs(self.csv_path, exist_ok=True)
        os.makedirs(self.pdf_path, exist_ok=True)
        
        self.embedder = embedder
        
    @staticmethod
    def clean_text(text):
        """
        Cleans text to reduce storage and improve readability
        
        Args:
            - text (str):   text to clean
        """
        return " ".join(text.split())
    
    
    def filter_text(self, text, similarity_threshold=0.5):
        """
        Filters through mass amounts of data to see if there is relevance to our use case

        Args:
            - text (str):   text to check and filter through
        """
        keywords = [ "law", "compliance", "regulation", "tax", "contract", "finance", "employment", "business", "corporate", "trade", "policy", "license", "permits", "registration", "audit", "liability", "shareholders", "partnership", "incorporation", "startup", "profit", "revenue", "penalty", "customs", "dispute", "governance", "authority", "import", "export", "data", "privacy", "director", "management", "intellectual", "property", "trademark", "compliance", "ownership", "capital", "dividend", "funding", "duty", "penalty", "fine", "subsidiary", "merger", "acquisition", "bankruptcy", "insolvency"]

        def cosine_similarity(vec1, vec2):
            """
            Computes the cosine similarity between two vectors and returns cosine similarity score

            Args:
                - vec1: First embedding vector
                - vec2: Second embedding vector
            """
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) 
    
        words = set(text.lower().split())
        matches = set(keywords).intersection(words)
        keyword_score = len(matches) / len(words) if len(words) > 0 else 0
        
        if len(text) > 0:
            query_embedding = self.embedder.embed_query("Relevant and important information about corporate laws, compliance requirements, tax obligations, and regulatory guidelines for Small and Medium Enterprises (SMEs) operating in Singapore.")
            text_embedding = self.embedder.embed_query(text)
            similarity_score = cosine_similarity(query_embedding, text_embedding)
        else:
            similarity_score = 0

        return (0.1 * keyword_score) + (0.9 * similarity_score) > 0.4
    

    def embed_text(self, text):
        """
        Embeds a given text if necessary

        Args:
            - text (str):   text to embed
        """
        return self.embedder.embed_query(text)
    
    @staticmethod
    def read_csv(csv_path: str) -> CSVLoader:
        """
        Reads a CSV file and returns a LangChain CSVLoader instance.

        Args:
            - csv_path (str):   path to csv file
        """
        df = pd.read_csv(csv_path)
        return CSVLoader(
            file_path=csv_path,
            csv_args={"delimiter": ",", "quotechar": '"', "fieldnames": df.columns.tolist()}
        )
    
    def scrape_csvs(self, directory: str):
        """
        Adds CSV files from a directory to the tabular data loaders.

        Args:
            - directory (str):  directory of csv files
        """
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    loader = self.read_csv(file_path)
                    for row in loader.load()[1:]:
                        if self.filter_text(row.page_content):
                            split_data = self.text_splitter.split_text(self.clean_text(row.page_content))

                            self.tabular_data.extend(split_data)
                            self.tabular_metadata.extend([{"source": file, "date_added": "None"} for _ in split_data])
                            print(f"Current csv: {file}")


    def extract_table_elements(self, url, tables):
        """
        Extracts tabular data from HTML tables and saves them as CSV and loaders

        Args:
            - url (str):        url the table is from
            - tables (list):    list of table elements
        """
        for idx, table in enumerate(tables):
            try:
                # Clean up table superscripts
                for tag in table.find_all("sup"):
                    tag.extract()

                # Extract the table name
                header = table.find_previous(["h3", "h4"])
                table_name = self.clean_text(header.text) if header and header.text else f"table_{idx}"
                table_name = os.path.basename(url) + f" {table_name}"

                # Parse table into a DataFrame
                df = pd.read_html(StringIO(str(table)), header=0)[0]
                df["context"] = table_name
                for index, row in df.iterrows():
                    row_context = [f"Row {idx+1} - {col}: {str(row[col])}" for col in df.columns]
                    full_context = " | ".join(row_context)  # Join all columns with a separator to keep context
                    if self.filter_text(full_context):
                        split_data = self.text_splitter.split_text(self.clean_text(full_context))

                        self.tabular_data.extend(split_data)
                        self.tabular_metadata.extend([{"source": url, "date_added": "None"} for _ in split_data])
                        print(f"Current table: {table_name}")

            except Exception as e:
                print(f"Failed to extract table from {url}: {e}")

    def create_loaders(self):
        """
        Processes raw data to create LangChain loaders

        Args:   None
        """
        for url, soup in self.raw_data:
            main_content = soup.find("main") 
            if main_content:
                raw_text = main_content.get_text(separator=" ", strip=True)
                if self.filter_text(raw_text):
                    split_data = self.text_splitter.split_text(self.clean_text(raw_text))

                    self.textual_data.extend(split_data)
                    self.textual_metadata.extend([{"source": url, "date_added": "None"} for _ in split_data])

    def scrape_websites(self, urls, max_depth, depth=0):
        """
        Recursively scrapes a website for HTML content and tables and then processes the data into loaders

        Args:
            - urls (list):      list of urls to scrape from
            - max_depth (int):  how many sublinks into the main website we wish to mdig through
            - depth (int):      current/starting depth of sublinks reached
        """
        for base_url in urls:
            def _scrape(url, depth):
                if url in self.visited_urls or depth > max_depth:
                    return

                try:
                    response = requests.get(url)
                    if response.status_code != 200:
                        print(f"Failed to retrieve {url}")
                        return

                    soup = BeautifulSoup(response.content, "html.parser")
                    self.visited_urls.add(url)
                    self.raw_data.append((url, soup))

                    # Process tables
                    self.extract_table_elements(url, soup.find_all("table"))

                    # Recurse through links
                    for link in soup.find_all("a", href=True):
                        abs_url = urllib.parse.urljoin(url, link["href"])
                        if base_url in abs_url:
                            _scrape(abs_url, depth + 1)
                    print(f"Current url: {url}")

                except Exception as e:
                    print(f"Error accessing {url}: {e}")

            _scrape(base_url, depth)
            self.create_loaders()

    def scrape_pdfs(self, pdf_folder=None):
        """
        Extracts text from PDF files.
        Args:
            - pdf_folder (str): folder of pdf files
        """
        pdf_folder = pdf_folder or self.pdf_path
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, file)
                try:
                    doc = fitz.open(pdf_path)
                    pdf_text = "".join(page.get_text("text") for page in doc)
                    if self.filter_text(pdf_text):
                        split_text = self.text_splitter.split_text(self.clean_text(pdf_text))

                        self.textual_data.extend(split_text)
                        self.textual_metadata.extend([{"source": pdf_path, "date_added": "None"} for _ in split_text])
                        print(f"Current pdf: {pdf_path}")

                except Exception as e:
                    print(f"Failed to process {pdf_path}: {e}")
                
    def prepare_data_for_insertion(self):
        """
        Prepares data before data can be added into the vector db

        Args:   None
        """
        self.all_data = self.textual_data + self.tabular_data
        self.embedded_data = [self.embed_text(text) for text in tqdm(self.textual_data, desc=f"Total pieces of textual data: {len(self.textual_data)}")] + [self.embed_text(text) for text in tqdm(self.tabular_data, desc=f"Total pieces of tabular data: {len(self.tabular_data)}")]
        self.metadata = self.textual_metadata + self.tabular_metadata
        
# End data handler class

# Data loader
websites = [
    "https://www.iras.gov.sg",
    "https://www.mom.gov.sg",
    "https://www.acra.gov.sg",
    "https://singaporelegaladvice.com",
    "https://www.ipos.gov.sg",
    "https://www.enterprisesg.gov.sg",
    "https://www.skillsfuture.gov.sg",
    "https://www.hsa.gov.sg",
    "https://www.sfa.gov.sg"
]

datahandler = DataHandler(embedder=embedder)
datahandler.scrape_websites(websites, max_depth=5)
datahandler.scrape_pdfs(r'/project/data/pdf')
datahandler.scrape_csvs(r'/project/data/csv')


placeholder_textual_data = datahandler.textual_data 
placeholder_textual_metadata = datahandler.textual_metadata 
placeholder_tabular_data = datahandler.tabular_data 
placeholder_tabular_metadata = datahandler.tabular_metadata 

datahandler = DataHandler(embedder=embedder)
datahandler.textual_data = placeholder_textual_data
datahandler.textual_metadata = placeholder_textual_metadata
datahandler.tabular_data = placeholder_tabular_data
datahandler.tabular_metadata = placeholder_tabular_metadata

datahandler.prepare_data_for_insertion()

## milvus class
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import hashlib

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
                # categories = self.generate_data_categories(original[i])
                categories = 'None'
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
            
# specify milvus
database = MilvusDB('http://localhost:19530')
database.create_collection("512", 512)
database.collection_name = "512"
database.insert_data(datahandler.all_data, datahandler.embedded_data, datahandler.metadata)

# check if correct
if database.client.has_collection("512"):
    print("collection 512 exist")
else:
    print("collection 512 does not exist")
    
records = database.get_all_records()
print(len(records))
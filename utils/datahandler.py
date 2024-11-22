# common packages
import os
import numpy as np
import pandas as pd
import requests
import fitz
from tqdm import tqdm
from io import StringIO
from bs4 import BeautifulSoup, SoupStrainer

# cancerous langchain stuffs
from langchain_community.document_loaders import WebBaseLoader, DataFrameLoader, CSVLoader, UnstructuredTSVLoader, TextLoader, UnstructuredHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings # just typing, we dont define any embedder here. don't worry.

class DataHandler:
    """
    Masterfully handles data scraping, preprocessing, and other data-related functionalities in this notebook.
    """

    def __init__(self, embedder: NVIDIAEmbeddings, data_home="/raw_web_data"):
        # Paths and storage
        self.data_home = data_home
        self.csv_path = os.path.join(self.data_home, "csv")
        self.pdf_path = os.path.join(self.data_home, "pdf")
        self.html_path = os.path.join(self.data_home, "html")
        self.visited_urls = set()
        self.max_path_length = os.pathconf('/', 'PC_NAME_MAX')

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
        
        self.embedder = embedder # type hints aside, im not sure if other embeddings can work in this place.
        
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
                    # print(split_data)
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
                    
                    save_fp = os.path.join(self.html_path, url.replace("/", "_"))[:self.max_path_length - 4]+ ".txt"
                    print("FILEPATH SAVING TO IS", save_fp ) 
                    with open( save_fp, "wb") as f:
                        f.write(response.content)

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
            
            
    def from_cached_websites(self, raws_folder):
        """
        Recursively scrapes a website for HTML content and tables and then processes the data into loaders

        Args:
            - urls (list):      list of urls to scrape from
            - max_depth (int):  how many sublinks into the main website we wish to mdig through
            - depth (int):      current/starting depth of sublinks reached
        """
        print(len(os.listdir(raws_folder)))
        for file in os.listdir(raws_folder):
            try:
                with open( os.path.join(raws_folder, file), "r") as f:
                    htmlstring = f.read()

                soup = BeautifulSoup(htmlstring, "html.parser")
                self.visited_urls.add(file) # supposed to be a url, but we just pass in file for now for placeholder variable. later fix.
                self.raw_data.append((file, soup))
                print("file", file)
                # Process tables
                self.extract_table_elements(url=file, tables=soup.find_all("table"))

            except Exception as e:
                print(f"Caught error in function from_cached_websites, {e}")
        
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
        
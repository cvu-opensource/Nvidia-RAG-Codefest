# Project description
Using NIM APIs to implement a RAG system that ingests data scraped from the web, pdfs, or images and returns an output ranked by similarity
I love NIM! NIM is the goat!

# Current list of to-dos:
1. **Simple User Interface**
  - Send in queries with texts or images
  - Display responses in a chat-like format
  - User feedback

2. **Data Handler**
  - Embed images with CLiP/ViT/CNN or something, maybe use a separate model to extract features 
  - Store more metadata (eg. tags, date)

3. **MilvusDB**
  - Find a way to check/monitor sites for changes regularly and upsert into db
  - Use metadata stored to search for more relevant data instead of searching through all vectors
  - Implement reranking 

4. **Prompt Engineering**
  - Figure out how

5. **Containerise**
  - 4 containers for 4 segmented parts
    - UI
    - DB
    - Ingestion
    - Inference

# Our current competition:

### 1) IRAC 28 + digispecs2... bob.
### 2) A fireside chat with the IRAS bot. It reads:

Me: I am a permanent-resident earning $50,000 income. What is my tax rate?

Bot: You may refer to the table below for the Resident income tax rates from YA 2017 to YA 2023.
 
  Do note that the tax rates are applied on the chargeable income. 
   
  Chargeable income = (Total taxable income - allowable expenses - approved donations - personal reliefs)
  (translation options)
  
  To achieve greater progressivity, the top marginal personal income tax rate will be increased with effect from YA 2024.
  
  (some tabular data)
  
  You may refer to the table below for resident income tax rates from YA 2024:

  (other tabular data)


Me: I am blind. I cannot see tabular data. Can you tell me through text?

Bot: Sorry. I didn't catch that. Would you like to rephrase your question?

    Here are some tips:
    
    ✔ Ask your question in a single message. Keep it short and concise.
    
    ✔ Include keywords in your questions e.g. How to pay tax via GIRO
    Original
    
    I found some information from Google which may be of help to you.

    VOLUME 47 | NUMBER 1 | FREE PAPERS | JANUARY 2018
    Territorial Complementarities and Competition for Oil and Gas FDI in ...
    A*DAX: A Platform for Cross-domain Data Linking, Sharing and ...

### I have concluded something. IRAS bot is weak. And fragile. And pathetic. 

We must be better.

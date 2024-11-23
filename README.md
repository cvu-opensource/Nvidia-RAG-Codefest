# README

## Retrieval-Augmented Generation (RAG) Project  
### Nvidia x SMC CodeFest Singapore - Product of team Asper Lovers

This project was developed as part of the **Nvidia x SMC CodeFest Singapore**, where participants were challenged to create a robust Retrieval-Augmented Generation (RAG) architecture leveraging **Nvidia NIM APIs**.  

### Project Highlights  

- **Infrastructure**:  
  We utilized cutting-edge clusters provided by **Sustainable Metal Cloud (SMC)**, tailored specifically for the event as per Nvidia's requirements. These clusters were equipped with:
  - **8 NVIDIA A100 GPUs** (80GB RAM each).  
  - Immersive cooling systems for optimal performance.  

- **Development Stack**:  
  The project involved extensive use of **Python scripts**, **Jupyter notebooks**, and **Docker** to orchestrate a modular RAG architecture.  

---

### Components Overview  

Our architecture consists of several modular services, all organized within the `RAG components` folder. Each component is self-contained with the following:  
- **Python scripts**  
- **Dockerfiles**  
- **Docker Compose files**  

The major **components** are:  

1. **Milvus Database Instance**  
   - For efficient storage and retrieval of vector embeddings.  

2. **Data APIs**  
   - Used for data ingestion and processing pipelines.  

3. **Multi-Agent Tools**  
   - Facilitating interaction and task delegation across different agents, binded to our LLM

4. **Vision-Language Model (VLM) Instance**  
   - To enable multimodal query processing capabilities.  

The **models** used are:

1. Large Language Models
    - llama-3.1-8b-instruct

2. Embedders
    - nv-embedqa-e5-v5
    - neo4j embedder?

3. Vision Language Models
    - Qwen2-VL-7B-Instruct

4. Reranker
    - nv-rerankqa-mistral-4b-v3

---

### **Product Showcase**  

Here are some snapshots of our completed RAG system and its components:  

#### **1. User Interface**  

*Description*: The **main interface** of the RAG system, showcasing how users can interact with the chatbot to retrieve information.  

<img src="static\ui\main-ui.jpg" alt="Main UI" width="400">

*Description*: Chatbot's **response** to user query, including source citation.

<img src="static\ui\model-response.jpg" alt="Model Response" width="400">

*Description*: User **feedback** portion after receiving chatbot's response, along with viewable **conversation history** (incorporated into further LLM context). 

<img src="static\ui\feedback-and-history.jpg" alt="User feedback and Conversation History" width="400">

*Description*: Verification of model's accuracy using very specific detail found in tabular data.

<img src="static\ui\pwcs-example.jpg" alt="PWCS Example" width="400">

#### **2. System Architecture Overview**  
*Description*: A high-level diagram of the system architecture, illustrating how the different components interact.  
![System Architecture](path/to/system-architecture.png)  

#### **3. Multi-Agent Tool**  
*Description*: A visual representation of the multi-agent tool in action, demonstrating task delegation and collaboration.  
![Multi-Agent Tool](path/to/multi-agent-tool.png)  

#### **4. Vision-Language Model Integration**  
*Description*: An example of the Vision-Language Model (VLM) processing multimodal inputs and delivering results.  
![VLM Integration](path/to/vlm-integration.png)  

#### **5. Final Presentation**  
*Description*: Our final [presentation](https://docs.google.com/presentation/d/1_2nHfmNehYHGppRed6TObZtmIriuuReuNpOEMso0quc/edit#slide=id.p2)   to other corporations and Nvidia experts.


---

### How to Use  

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Navigate to the `RAG components` Folder**  
   Each service has its dedicated subfolder. Navigate to the desired service's folder to explore its configuration.  

3. **Run Services Locally**  
   - Each component has its own `Dockerfile` and `docker-compose.yml` for easy deployment.  
   - To run a specific component:  
     ```bash
     docker-compose up
     ```  

4. **Modify and Experiment**  
   - All Python scripts are modular and well-documented to allow easy experimentation and integration.  

---

### Acknowledgments  

This project would not have been possible without:  
- **Nvidia**, for their state-of-the-art NIM APIs and support.  
- **SMC**, for hosting powerful local clusters that facilitated the development and testing of our RAG architecture.  

For further details, feel free to explore the project files or reach out to the contributors.  

---  

### Contributors

| Name            | Role                          | GitHub Profile                        | LinkedIn Profile                       |
|-----------------|-------------------------------|---------------------------------------|----------------------------------------|
| **Gerard Lum**   | Full Stack Developer      | [GitHub](https://github.com/username) | [https://www.linkedin.com/in/gerardlumkaien/](https://www.linkedin.com/in/gerardlumkaien/) |
| **Benjamin Goh** | AI Specialist   | [GitHub](https://github.com/username) | [https://www.linkedin.com/in/benjamin-goh-45a0a7307/](https://www.linkedin.com/in/benjamin-goh-45a0a7307/) |
| **Skyler Lee**   | Solutions Architect, Docker Expert  | [GitHub](https://github.com/username) | [https://www.linkedin.com/in/skyler-lee-6465741a7/](https://www.linkedin.com/in/skyler-lee-6465741a7/) |
| **Ng Le Jie**    | Software Engineer, LLM Expert | [GitHub](https://github.com/username) | [https://www.linkedin.com/in/le-jie-ng-13a547211/](https://www.linkedin.com/in/le-jie-ng-13a547211/) |
| **Gavin Lim**    | Visual Designer  | [GitHub](https://github.com/username) | [https://www.linkedin.com/in/gavinlimsh/](https://www.linkedin.com/in/gavinlimsh/) |


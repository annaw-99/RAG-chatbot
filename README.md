## RAG chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload their own documents (.txt or .pdf files) and ask questions based on the content. Through a large language model (LLM) and a retrieval method, the chatbot provides accurate and context-specific answers.

### Features

- **Document Upload:** Support .txt and .pdf files
- **Advanced Text Processing:** Utilizes text splitting and embeddings for efficient retrieval.
- **Customizable Settings:** Allows customized settings for chunk size and overlaps to optimize performance.
- **User-Friendly Interface:** Implements Streamlit for an interactive web experience.

### Installation

Follow these steps to run the application locally on your machine.

#### Prerequisites

Make sure to have **Python** and **Docker** installed (and/or any dependencies) on your local machine.

**1. Clone the Repo**
   
   ```sh
   git clone [https://github.com/](https://github.com/annaw-99/RAG-chatbot)
   ```

**2. Reopen in Dev Container**

   Open your repo in a code editor (e.g. VSCode) and reopen the files in a dev container.

**3. Navigate to /workspace/docker-compose.yml**

   Replace `<API_KEY>` and `<ENDPOINT>` with your actual API key and endpoint.
   ```sh
   AZURE_OPENAI_API_KEY: <API_KEY>
   AZURE_OPENAI_ENDPOINT: <ENDPOINT>
   AZURE_OPENAI_MODEL_DEPLOYMENT: gpt-4o
   OPENAI_API_KEY: <API_KEY>
   ```

**4. Rebuild the Container**

   If you're using VSCode, navigate to the bottom-left corner of your window, click on it, and select **Rebuild Container.**

**5. Run the Steamlit App**

   Navigate to the terminal and run
   ```sh
   streamlit run chatbot_with_files.py
   ```
You should then be able to view and interact with the application. :)

#### Installation Note

Change the name in `/docker-compose.yml` if dev container name already exists.

   ```sh
   container_name: <NEW_NAME>
   ```

### Application Usage

You can upload either **.txt** or **.pdf** files through the input box. Please ensure all documents are uploaded **before** asking any questions to the chatbot. When interacting with the chatbot, make sure to ask **CLEAR** and **SPECIFIC** questions to ensure more accurate responses.

### References

This project includes code referenced from INFO 5940 and guidance from ChatGPT.

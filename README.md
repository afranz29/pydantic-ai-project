# Autonomous Report Builder (WIP)

An autonomous AI agent that can research a user's query and write a detailed, well-structured research report.

## About

This project is an autonomous report builder that uses AI agents using the PydanticAI framework to perform research and synthesize the findings into a comprehensive report. 

### Process
The application starts with `ResearchAgent`, an agent that takes a user's prompt, breaks it down into a number of sub-questions, and uses a web search tool to gather information on each. Once the research is complete, a `SynthesizerAgent` processes the collected data and writes a formal research report, complete with a title, abstract, distinct sections, and a list of references.

The application is built with FastAPI and exposes a simple web interface where a user can input a topic and receive a generated report.

## Installation and Usage

Follow these steps to set up and run the project locally. 

> [!IMPORTANT]
> You will need an LLM server for the Agents to use. Below is how to install one locally, but you can use a remote one too. Settings for changing the model used are not currently implemented, but will be soon!

### 1. Install Local LLM Server (Ollama)

This project uses a local LLM server to run the two Agents

1.  **Download and Install Ollama:**
    * Navigate to the [Ollama download website](https://ollama.com/download).
    * Download and run the installer for your operating system (macOS, Linux, or Windows).

2.  **Pull a Model:**
    * Start Ollama with
      `ollama serve`
    * Once Ollama is running, pull the models `llama3.1:8b` and `qwen3:14b` by running the following commands:
        ```bash
        ollama pull llama3.1:8b
        ollama pull qwen3:14b
        ```

3.  **Verify Installation:**
    * To ensure Ollama is running correctly, you can list the models you have downloaded:
        ```bash
        ollama list
        ```

### 2. Set Up Local Environment

It is recommended to use a virtual environment to manage project dependencies. This project uses `uv`. 

1.  **Install `uv`:**
    If you don't have `uv` installed, you can install it with pip:
    ```bash
    pip install uv
    ```
    Docs: https://docs.astral.sh/uv/

2.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install requirements:**
    Install all the necessary packages from `requirements.txt`:
    ```bash
    uv pip install -r requirements.txt
    ```

### 3. Configure the Project

The application's behavior can be configured either by modifying the `config.py` file directly or by setting environment variables in a `.env` file. The application uses a `.env` file to manage secrets and settings.

1.  **Create a `.env` file:**
    Create a file named `.env` in the root of the project directory.

2.  **Add required environment variables:**
    You will need to provide the host for your local Ollama instance. Add the following to your `.env` file:
    ```
    OLLAMA_HOST="localhost:11434"
    ```

3.  **Customize behavior (Optional):**
    You can customize the behavior of the web search tool and agents by adding nested environment variables to your `.env` file. The delimiter for nested settings is `__`.

    For example, to change the number of search results for each subquestion for the `WebSearchTool`, you would add:
    ```
    WEB_SEARCH_TOOL__NUM_SEARCH_RESULTS=5
    ```
    Available settings can be found in `config.py`.

### 4. Run the Project

Once the project is configured, you can start the application using `uvicorn`.

1.  **Run the application:**
    ```bash
    python3 main.py
    ```
    The synthesizer's prompt containing the research result and the final report output will be saved to the file defined in `save_to_file.py`. This utility is mostly for debugging/development purposes, and a setting to turn it off/on will be implemented in the future.

### 5. Interact with the App

After starting the application, you can interact with it through the web interface.

1.  Open your web browser and navigate to:
    ```
    http://localhost:8000/app/
    ```
2.  Enter a topic you want to research in the input box and click "Generate Report".
3.  The generated report and the raw JSON output will be displayed on the page.

## Resources/Docs

-   **FastAPI:** https://fastapi.tiangolo.com/
-   **Pydantic:** https://docs.pydantic.dev/latest/
-   **PydanticAI:** https://ai.pydantic.dev/
-   **duckduckgo-search:** https://pypi.org/project/duckduckgo-search/
-   **googlesearch-python:** https://pypi.org/project/googlesearch-python/
-   **crawl4ai:** https://docs.crawl4ai.com/
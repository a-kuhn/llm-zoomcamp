# QA Bot for DataTalks.Club Zoomcamp Courses

This project implements a Question-Answering (QA) Bot designed to assist participants in completing DataTalks.Club Zoomcamp courses (data engineering, ML engineering, & MLOps) The bot leverages a RAG design, using course FAQ documents indexed with ElasticSearch for retrieval & (for now) OpenAI's GPT-4o for response generation.

## Features

- **Document Retrieval**: Fetches relevant documents based on the user's query.
- **Filtering by Course Content**: Allows filtering the documents by specific course sections for more accurate answers.
- **Customized Prompts**: Builds prompts for OpenAI's API using the retrieved documents, ensuring the answers are contextually relevant.
- **OpenAI Integration**: Utilizes OpenAI's API to generate answers to user queries.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Pipenv for dependency management
- OpenAI API key

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/a-kuhn/llm-zoomcamp.git
   ```

2. Navigate to the project directory:
`cd llm-zoomcamp`
3. Install dependencies using Pipenv:
`pipenv install`
4. Activate the virtual environment:
`pipenv shell`


### Configuration
Create a `.env` file in the project root directory and add your OpenAI API key:
`OPENAI_API_KEY='your_openai_api_key'`

### Running the Application
#### Start the ElasticSearch container  
_must first create a volume & index the FAQ docs (see jupyter notebook)_  
`> ./elasticsearch/scripts/run_elasticsearch_w_volumes.sh`  
  
#### Execute the main script to start the QA Bot:  
`> python3 -m api.app -q "how to run docker" -f "machine-learning-zoomcamp"`

# Mind Recommender Streamlit App

This Streamlit app leverages the power of OpenAI embeddings and Pinecone's vector database to provide recommendations based on user queries. It's designed to showcase how easily powerful AI and vector search capabilities can be integrated into Python applications.

## Features

- User-friendly interface for query input.
- Integration with OpenAI for generating embeddings.
- Use of Pinecone to perform vector similarity search.
- Display of top recommendations based on the query.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6+
- Streamlit
- Pinecone Python client
- OpenAI Python client

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Andi5986/talent-profiles-recommender.git
cd mind-recommender
```

### Set up a virtual environment:


```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install the required packages:

``` bash
pip install -r requirements.txt
```

## Configuration
Create a secrets.toml file in the root directory of the project with your OpenAI and Pinecone API keys:

toml
OPENAI_API_KEY = "your_openai_api_key_here"
PINECONE_API_KEY = "your_pinecone_api_key_here"

## Running the App
To run the app, use the following command:

``` bash
streamlit run index_profiles_app.py
```

Navigate to the URL provided in the terminal to view the app in your web browser.

## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
MIT



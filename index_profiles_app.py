import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

# Streamlit page configuration
st.set_page_config(page_title="Query Processor", layout="wide")

# Streamlit UI
st.title("Mind Recommender by index.dev")

query_input = st.text_area("Enter your query here", height=300)
process_button = st.button("Search")

# Get OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

class OpenAIEmbedder:
    def __init__(self, api_key):
        self.embed = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=api_key)

    def get_query_vector(self, query):
        return self.embed.embed_query(query)

# Function to process the query
def process_query(query):
    # Initialize embedder
    embedder = OpenAIEmbedder(OPENAI_API_KEY)

    try:
        query_vector = embedder.get_query_vector(query)
        if not isinstance(query_vector, list):
            query_vector = query_vector.tolist()

        # Initialize Pinecone with API key from Streamlit secrets
        pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment='gcp-starter')
        index = pinecone.Index('indexprofiles')
        results = index.query(
            namespace='indexprofiles', 
            top_k=5,
            include_values=True,
            include_metadata=True,
            vector=query_vector,
        )

        st.subheader("Recommended Profiles")
     
        # Check if results contain matches
        if 'matches' in results:
            for match in results['matches']:
                with st.expander(f"ID: {match['id']}, Score: {match['score']}"):
                    st.markdown(f"""
                    - **First Name**: {match['metadata'].get('first_name', 'N/A')}
                    - **Citizenship**: {match['metadata'].get('citizenship', 'N/A')}
                    - **Country**: {match['metadata'].get('country', 'N/A')}
                    - **English Proficiency**: {match['metadata'].get('english_proficiency', 'N/A')}
                    - **Experience**: {match['metadata'].get('experience', 'N/A')}
                    - **Main Experience**: {match['metadata'].get('main_experience', 'N/A')}
                    - **Preferred Monthly Rate**: {match['metadata'].get('preferred_monthly_rate', 'N/A')} USD
                    - **Skills**: {', '.join(match['metadata'].get('skills', []))}
                    - **Short Bio**: {match['metadata'].get('short_bio', 'N/A').replace('<br/>', ' ').replace('<p>', '').replace('</p>', '')}
                    """)
        else:
            st.write("No matches found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Process the query when the button is clicked
if process_button:
    process_query(query_input)

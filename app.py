import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = FastAPI()

# Setup environment variables
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX")

# Initialize pinecone client
pinecone = Pinecone(api_key=pinecone_api_key)

# Create or retrieve an index
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # This number should match your embedding dimension
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',  # or 'gcp' if using Google Cloud
            region='us-west-2'  # or any other region
        )
    )

index = pinecone.Index(index_name)

# Middleware to secure HTTP endpoint
security = HTTPBearer()

def validate_token(
    http_auth_credentials: HTTPAuthorizationCredentials = Security(security),
):
    if http_auth_credentials.scheme.lower() == "bearer":
        token = http_auth_credentials.credentials
        if token != os.getenv("RENDER_API_TOKEN"):
            raise HTTPException(status_code=403, detail="Invalid token")
    else:
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")

class QueryModel(BaseModel):
    query: str

@app.post("/")
async def get_context(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    # convert query to embeddings
    res = openai_client.embeddings.create(
        input=[query_data.query],
        model="text-embedding-ada-002"
    )
    embedding = res.data[0].embedding

    # Search for matching Vectors
    results = index.query(embedding, top_k=6, include_metadata=True).to_dict()

    # Filter out metadata from search result
    context = [match["metadata"]["text"] for match in results["matches"]]

    # Return context
    return context


# @app.get("/")
# async def get_context(query: str = None, credentials: HTTPAuthorizationCredentials = Depends(validate_token)):

#     # convert query to embeddings
#     res = openai_client.embeddings.create(
#         input=[query],
#         model="text-embedding-ada-002"
#     )
#     embedding = res.data[0].embedding
#     # Search for matching Vectors
#     results = index.query(embedding, top_k=6, include_metadata=True).to_dict()
#     # Filter out metadata fron search result
#     context = [match['metadata']['text'] for match in results['matches']]
#     # Retrun context
#     return context

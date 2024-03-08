import os

from langchain_google_vertexai import VertexAI
from langchain.chains import RetrievalQA

from langchain_google_vertexai import VertexAIEmbeddings

from langchain.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import BigQueryVectorSearch

NUMBER_OF_RESULTS = 3
PROJECT_ID = os.environ.get("PROJECT_ID", "")
REGION = "asia-northeast1"

BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "")
BIGQUERY_TABLE = os.environ.get("BIGQUERY_TABLE", "")

USE_LLM_MODEL_NAME = os.environ.get("USE_LLM_MODEL_NAME", "text-bison-32k@002")
USE_EMBEDDING_MODEL_NAME = os.environ.get("USE_EMBEDDING_MODEL_NAME", "textembedding-gecko@latest")

embedding = VertexAIEmbeddings(
    model_name=USE_EMBEDDING_MODEL_NAME, project=PROJECT_ID
)

store = BigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=BIGQUERY_DATASET,
    table_name=BIGQUERY_TABLE,
    location=REGION,
    embedding=embedding,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)

# BigQueryにデータをセットする
all_texts = []

metadatas = [{"len": len(t)} for t in all_texts]
store.add_texts(all_texts, metadatas=metadatas)

retriever = store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": NUMBER_OF_RESULTS})

chat = VertexAI(model_name=USE_LLM_MODEL_NAME, temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)

prompt = "鈴木に苦手な食べ物はありますか？"

result = qa_chain.invoke(prompt)
print(result)

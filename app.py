import os

from langchain.chains import RetrievalQA
from langchain.vectorstores.utils import DistanceStrategy

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAI

from langchain_community.vectorstores import BigQueryVectorSearch

NUMBER_OF_RESULTS = 3
PROJECT_ID = os.environ.get("PROJECT_ID", "")
LOCATION = os.environ.get("LOCATION", "asia-northeast1")

USE_CHAT_MODEL_NAME = os.environ.get("USE_CHAT_MODEL_NAME", "text-bison-32k@002")
USE_EMBEDDING_MODEL_NAME = os.environ.get("USE_EMBEDDING_MODEL_NAME", "textembedding-gecko@latest")

BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "")
BIGQUERY_TABLE = os.environ.get("BIGQUERY_TABLE", "")


# アプリを起動します
if __name__ == "__main__":
    embedding = VertexAIEmbeddings(
        model_name=USE_EMBEDDING_MODEL_NAME, project=PROJECT_ID
    )

    vector_store = BigQueryVectorSearch(
        project_id=PROJECT_ID,
        dataset_name=BIGQUERY_DATASET,
        table_name=BIGQUERY_TABLE,
        location=LOCATION,
        embedding=embedding,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": NUMBER_OF_RESULTS}
    )

    chat = VertexAI(model_name=USE_CHAT_MODEL_NAME, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever
    )

    query = """
プロンプト
"""

    answer = qa_chain.invoke(query)
    print(answer)

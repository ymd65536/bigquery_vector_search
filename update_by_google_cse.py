import os
import datetime

from google.cloud import bigquery

from langchain.vectorstores.utils import DistanceStrategy
from langchain_google_vertexai import VertexAIEmbeddings

from langchain.tools import Tool

from langchain_community.vectorstores import BigQueryVectorSearch
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer

os.environ["GOOGLE_CSE_ID"] = ""
os.environ["GOOGLE_API_KEY"] = ""

NUMBER_OF_RESULTS = 3
PROJECT_ID = os.environ.get("PROJECT_ID", "")
REGION = "asia-northeast1"

BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "")
BIGQUERY_TABLE = os.environ.get("BIGQUERY_TABLE", "")

USE_EMBEDDING_MODEL_NAME = os.environ.get("USE_EMBEDDING_MODEL_NAME", "textembedding-gecko@latest")


def get_web_page_document(url: str, start_tag: str, end_tag: str) -> list:
    """
        Webページのドキュメントを取得してBody部分だけ取り出すメソッド

    Returns:
        _type_: WebページのうちBodyタグで囲まれた文字列（htmlパース後の文字列）
    """
    loader = RecursiveUrlLoader(url)
    documents = loader.load()

    for document in documents:
        start_index = document.page_content.find(start_tag)
        end_index = document.page_content.find(end_tag)
        documents[0].page_content = document.page_content[start_index:end_index]

    html2text = Html2TextTransformer()
    return html2text.transform_documents(documents)


def top5_results(query):
    search = GoogleSearchAPIWrapper()
    return search.results(query, 5)


if __name__ == '__main__':
    tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=top5_results,
    )

    keyword = ""

    results = tool.run(keyword)

    if results[0].get('Result') == 'No good Google Search Result was found':
        print("検索結果なし")
    else:
        client = bigquery.Client(project=PROJECT_ID, location=REGION)
        client.create_dataset(dataset=BIGQUERY_DATASET, exists_ok=True)

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

        docs = []
        metadatas = []
        DIFF_JST_FROM_UTC = 9
        result_time = datetime.datetime.utcnow() + datetime.timedelta(hours=DIFF_JST_FROM_UTC)

        for result in results:
            result["body"] = get_web_page_document(result["link"], '<body', '</body')[0].page_content
            result["body"] = "(タイトル title)：[" + result["title"] + "](" + result["link"] + ") " + result["body"]
            result["body"] = result["body"] + " ファイル名:" + result['link'].split('/')[-1]
            result["body"] = result["body"] + " データの更新日:" + f"{result_time}"

            yyyymmdd = f"{result_time.strftime('%Y')}年{result_time.strftime('%m')}月{result_time.strftime('%d')}日"
            hourminsec = f" {result_time.strftime('%H')}:{result_time.strftime('%M')}分:{result_time.strftime('%S')}秒"
            result["body"] = yyyymmdd + hourminsec + result["body"]

            docs.append(result["body"])
            metadatas.append(
                {
                    "yyyymmdd": yyyymmdd,
                    "hourminsec": hourminsec,
                    "keyword": keyword,
                    "title": result["title"],
                    "link": result['link'],
                    "len": str(len(result["body"])),
                    "html_file": result['link'].split('/')[-1],
                    "update_time": str(result_time)
                }
            )

        store.add_texts(docs, metadatas=metadatas)

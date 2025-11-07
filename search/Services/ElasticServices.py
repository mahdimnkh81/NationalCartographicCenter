from django.db import connection
from search.Services.normalizer import normalize
from search.models.postgres.models import LogUserSearch, DocumentEmbedding
from django.utils.timezone import now
import os
from elasticsearch import Elasticsearch
import numpy as np

# es = Elasticsearch("http://87.248.153.146:9200/")
es = Elasticsearch("localhost:1000/")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/faiss_index.index')

INDEX_NAME = "data_embeddings"

def create_document(data):
    response = es.index(
        index=INDEX_NAME,
        body=data
    )
    return response

def search_elastic(query):
    results = []

    query = normalize(query)
    if query:
        # جستجو بر اساس 'layer.keyword' (دقیق) و بازگشت فیلدهای 'layer', 'system', 'dataModel'
        # query_body = {
        #     "query": {
        #
        #         "match": {
        #             "layer": query  # جستجو بر اساس عبارت وارد شده در فیلد 'layer'
        #         }
        #         # "term": {
        #         #     "layer.keyword": query  # جستجو دقیق بر اساس زیرفیلد 'keyword'
        #         # }
        #     },
        #     "_source": ["layer", "system", "dataModel"],  # نمایش فیلدهای 'layer', 'system', 'dataModel'
        #     "size": 5  # محدود کردن تعداد نتایج به 10
        # }

        query_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "term": {
                                "layer.keyword": {
                                    "value": query,  # جستجو دقیق بر اساس زیرفیلد 'keyword'
                                    "boost": 2.0  # افزایش اولویت نتایج دقیق
                                }
                            }
                        },
                        {
                            "match": {
                                "layer": {
                                    "query": query,  # جستجو بر اساس عبارت وارد شده در فیلد 'layer'
                                    "boost": 1.0  # اولویت کمتر برای جستجوی تمام‌متن
                                }
                            }
                        }
                    ]
                }
            },
            "_source": ["layer", "system", "dataModel"],
            "size": 5
        }

        response = es.search(index=INDEX_NAME, body=query_body)
        hits = response['hits']['hits']
        max_score = response['hits'].get('max_score', 1)
        results = [
            {
                "layer": hit["_source"].get("layer"),
                "system": hit["_source"].get("system"),
                "dataModel": hit["_source"].get("dataModel"),
                "score": hit["_score"] / max_score if max_score else 0
            }
            for hit in hits
        ]
    return results

def save_user_selection(query, post_data, type_result):

    result_count = int(post_data.get("result_count", 0))

    for i in range(1, result_count + 1):
        selected_layer = post_data.get(f"layer_{i}")
        selected_organization = post_data.get(f"system_{i}")
        try:
            score = float(post_data.get(f"score_{i}", 0))
        except ValueError:
            score = 0.0

        selection_value = post_data.get(f"selection_{i}", None)
        if selection_value is None:
            relevance = 0
        elif selection_value == "related":
            relevance = 1
        elif selection_value == "unrelated":
            relevance = -1
        else:
            relevance = 0

        log_entry = LogUserSearch(
            query=query,
            selected_layer=selected_layer,
            selected_organization=selected_organization,
            relevance=relevance,
            score=score,
            type_result= type_result,
            timestamp=now()
        )
        log_entry.save()


def save_log_entry_from_json(data):

    entries = data if isinstance(data, list) else [data]

    for item in entries:
        query = item.get('query')
        layer = item.get('layer')
        organization = item.get('organization')
        relevance = int(item.get('relevance', 0))
        score = float(item.get('score', 0))
        type_result = int(item.get('type_result', 0))

        log_entry = LogUserSearch(
            query=query,
            selected_layer=layer,
            selected_organization=organization,
            relevance=relevance,
            score=score,
            type_result=type_result,
            timestamp=now()
        )
        log_entry.save()


def delete_all_embeddings_data():

    DocumentEmbedding.objects.all().delete()
    with connection.cursor() as cursor:
        cursor.execute("ALTER SEQUENCE embeddings_id_seq RESTART WITH 1;")
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)

    es.delete_by_query(index=INDEX_NAME, body={
        "query": {
            "match_all": {}
        }
    })



def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    return obj


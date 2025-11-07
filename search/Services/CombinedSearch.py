from django.utils.timezone import now

from search.Services.ElasticServices import search_elastic
from search.Services.SbertServices import sbert_service
from search.models.postgres.models import LogUserSearch


def get_combined_results(query, top_k=5):

    # دریافت نتایج از SBERT و Elasticsearch
    sbert_results = sbert_service.search(query, top_k=top_k)
    elastic_results = search_elastic(query)

    combined_dict = {}
    for res in elastic_results:
        layer = res.get('layer')
        if layer:
            combined_dict[layer] = res.copy()
    for res in sbert_results:
        layer = res.get('layer')
        if layer:
            if layer in combined_dict:
                combined_dict[layer]['score'] += res.get('score', 0)
            else:
                combined_dict[layer] = res.copy()

    combined_results = sorted(combined_dict.values(), key=lambda x: x.get('score', 0), reverse=True)[:top_k]

    if combined_results:
        max_score = max(res.get('score', 0) for res in combined_results)
        for res in combined_results:
            res['score'] = res.get('score', 0) / max_score if max_score else 0.0

    return combined_results

def log_user_search(query, result_count, post_data, LogUserSearch, now):
    for i in range(1, result_count + 1):
        selected_layer = post_data.get(f'layer_{i}')
        selected_system = post_data.get(f'system_{i}')
        try:
            score = float(post_data.get(f'score_{i}', 0))
        except ValueError:
            score = 0.0

        selection_value = post_data.get(f'selection_{i}', None)
        if selection_value is None:
            relevance = 0
        elif selection_value == 'related':
            relevance = 1
        elif selection_value == 'unrelated':
            relevance = -1
        else:
            relevance = 0

        LogUserSearch.objects.create(
            query=query,
            selected_layer=selected_layer,
            selected_organization=selected_system,
            relevance=relevance,
            score=score,
            type_result=0,
            timestamp=now()
        )

def save_log_entry_from_json_combined(data):

    entries = data if isinstance(data, list) else [data]

    for item in entries:
        query = item.get('query')
        layer = item.get('layer')
        organization = item.get('organization')
        relevance = int(item.get('relevance', 0))
        score = float(item.get('score', 0))
        type_result = int(item.get('type_result', 0))

        LogUserSearch.objects.create(
            query=query,
            selected_layer=layer,
            selected_organization=organization,
            relevance=relevance,
            score=score,
            type_result=type_result,
            timestamp=now()
        )

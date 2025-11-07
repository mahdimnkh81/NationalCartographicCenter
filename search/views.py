import json
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.utils.timezone import now
from search.Services.CombinedSearch import get_combined_results, log_user_search, save_log_entry_from_json_combined
from search.Services.ElasticServices import search_elastic, save_user_selection, convert_to_json_serializable, \
    save_log_entry_from_json, delete_all_embeddings_data
from search.Services.SbertServices import sbert_service, save_log_entry_from_json_sbert
from search.Services.TrainModel import build_sts_from_db, train_sbert_model, train_service
from search.models.postgres.models import LogUserSearch
from django.views.decorators.http import require_http_methods
import io
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_GET


# --------------------
# Elastic Related Views
# --------------------
# def search_elastic_view(request):
#     query = request.GET.get("q", "")
#     results = search_elastic(query) if query else []
#     return render(request, 'search_results_elastic.html', {'query': query, 'results': results})
def search_elastic_view(request):
    query = request.GET.get("query", "") or request.GET.get("q", "")
    results = search_elastic(query) if query else []

    # چک کردن اینکه درخواست JSON هست یا نه
    if request.headers.get('Accept') == 'application/json' or request.GET.get("format") == "json":
        return JsonResponse({
            "query": query,
            "results": results
        }, json_dumps_params={'ensure_ascii': False})

    # اگر JSON نبود، صفحه HTML نمایش داده میشه
    return render(request, 'search_results_elastic.html', {
        'query': query,
        'results': results
    })

# @csrf_exempt
# def log_user_selection_elastic_view(request):
#     if request.method == 'POST':
#         query = request.POST.get('query')
#         save_user_selection(query, request.POST, type_result=2)
#     return redirect('search_elastic_view')

@csrf_exempt
def log_user_selection_elastic_view(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            try:
                data = json.loads(request.body)
                save_log_entry_from_json(data)
                return JsonResponse({"status": "success"}, status=201)
            except Exception as e:
                return JsonResponse({"status": "error", "message": str(e)}, status=400)

        query = request.POST.get('query')
        save_user_selection(query, request.POST, type_result=2)
        return redirect('search_elastic_view')

    return JsonResponse({"error": "Only POST method allowed"}, status=405)


# --------------------
# SBERT Related Views
# --------------------
# def search_sbert_view(request):
#     query = request.GET.get('query', '')
#     results = sbert_service.search(query, top_k=5) if query else []
#     return render(request, 'search_results_sbert.html', {'query': query, 'results': results})
def search_sbert_view(request):
    query = request.GET.get('query', '') or request.GET.get("q", "")
    results = sbert_service.search(query, top_k=5) if query else []

    wants_json = (
        request.headers.get('Accept') == 'application/json' or
        request.GET.get("format") == "json" or
        request.path.endswith('.json')
    )

    if wants_json:
        safe_results = convert_to_json_serializable(results)
        return JsonResponse({
            "query": query,
            "results": safe_results
        }, json_dumps_params={'ensure_ascii': False})

    return render(request, 'search_results_sbert.html', {
        'query': query,
        'results': results
    })

def search_sbert_train_model_view(request):
    query = request.GET.get('query', '') or request.GET.get("q", "")
    results = train_service.search(query, top_k=5) if query else []

    wants_json = (
        request.headers.get('Accept') == 'application/json' or
        request.GET.get("format") == "json" or
        request.path.endswith('.json')
    )

    if wants_json:
        safe_results = convert_to_json_serializable(results)
        return JsonResponse({
            "query": query,
            "results": safe_results
        }, json_dumps_params={'ensure_ascii': False})

    return render(request, 'search_results_sbert_train_model.html', {
        'query': query,
        'results': results
    })


@csrf_exempt
def log_user_selection_sbert_view(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            try:
                data = json.loads(request.body)
                save_log_entry_from_json_sbert(data)
                return JsonResponse({"status": "success"}, status=201)
            except Exception as e:
                return JsonResponse({"status": "error", "message": str(e)}, status=400)

        # حالت فرم HTML معمولی
        query = request.POST.get('query')
        save_user_selection(query, request.POST, type_result=1)
        return redirect('search_sbert_view')

    return JsonResponse({"error": "Only POST method allowed"}, status=405)


def combined_search_view(request):
    query = request.GET.get('query', '').strip() or request.GET.get("q", "").strip()
    if not query:
        results = []
        wants_json = (
            request.headers.get('Accept') == 'application/json' or
            request.GET.get("format") == "json" or
            request.path.endswith('.json')
        )
        if wants_json:
            return JsonResponse({
                "query": "",
                "results": []
            }, json_dumps_params={'ensure_ascii': False})
        return render(request, 'combined_search_results.html', {'query': '', 'results': []})

    combined_results = get_combined_results(query)
    safe_results = convert_to_json_serializable(combined_results)

    wants_json = (
        request.headers.get('Accept') == 'application/json' or
        request.GET.get("format") == "json" or
        request.path.endswith('.json')
    )

    if wants_json:
        return JsonResponse({
            "query": query,
            "results": safe_results
        }, json_dumps_params={'ensure_ascii': False})

    return render(request, 'combined_search_results.html', {
        'query': query,
        'results': combined_results
    })

@csrf_exempt
def log_search_view(request):
    if request.method == "POST":
        if request.content_type == "application/json":
            try:
                data = json.loads(request.body)
                save_log_entry_from_json_combined(data)
                return JsonResponse({"status": "success"}, status=201)
            except Exception as e:
                return JsonResponse({"status": "error", "message": str(e)}, status=400)

        # حالت فرم HTML
        query = request.POST.get('query')
        try:
            result_count = int(request.POST.get('result_count', 0))
        except ValueError:
            result_count = 0

        log_user_search(query, result_count, request.POST, LogUserSearch, now)
        return redirect('combined_search_view')

    return JsonResponse({"error": "Only POST method allowed"}, status=405)


def success_view(request):
    return render(request, 'success.html')


def upload_excel_view(request):
    if request.method == 'POST':
        excel_file = request.FILES.get('excel_file')
        if not excel_file:
            return JsonResponse({'error': 'No file selected'}, status=400)

        try:
            # messages = sbert_service.process_excel_file(excel_file, min_row=2, max_row=21)
            messages = sbert_service.process_excel_file(excel_file, min_row=2)
        except Exception as e:
            print(e)
            return JsonResponse({'error': str(e)}, status=400)

        return render(request, 'success.html', {'messages': messages})

    return render(request, 'upload_excel.html')

def upload_excel_trainModel_view(request):
    if request.method == 'POST':
        excel_file = request.FILES.get('excel_file')
        if not excel_file:
            return JsonResponse({'error': 'No file selected'}, status=400)

        try:
            # messages = sbert_service.process_excel_file(excel_file, min_row=2, max_row=21)
            messages = train_service.process_excel_file(excel_file, min_row=2)
        except Exception as e:
            print(e)
            return JsonResponse({'error': str(e)}, status=400)

        return render(request, 'success.html', {'messages': messages})

    return render(request, 'upload_excel.html')



@require_http_methods(["GET", "POST"])
def delete_all_data(request):
    if request.method == "GET":
        return render(request, 'delete_confirmation.html')

    elif request.method == "POST":
        try:
            delete_all_embeddings_data()
            return redirect('delete_success')
        except Exception as e:
            return render(request, 'delete_confirmation.html', {
                'error': f"خطا در حذف اطلاعات: {str(e)}"
            })

def delete_success(request):
    return render(request, 'delete_success.html')

@require_GET
def build_sts_dataset_view(request):

        try:
            test_size = float(request.GET.get("test_size", 0.2))
            val_size = float(request.GET.get("val_size", 0.2))
            seed = int(request.GET.get("seed", 42))
            min_len = int(request.GET.get("min_len", 2))
            include_unlabeled = request.GET.get("include_unlabeled", "false").lower() in ["true", "1", "yes"]
            sample_n = int(request.GET.get("sample", 3))

            # ساخت دیتاست از دیتابیس
            train_ds, eval_ds, test_ds = build_sts_from_db(
                include_unlabeled=include_unlabeled,
                min_len=min_len,
                seed=seed
            )

            # اگر درخواست CSV است
            fmt = request.GET.get("format")
            if fmt == "csv":
                which = (request.GET.get("split") or "train").lower()
                if which not in ("train", "eval", "test"):
                    return JsonResponse({"error": "split نامعتبر. یکی از train/eval/test"}, status=400)

                ds = {"train": train_ds, "eval": eval_ds, "test": test_ds}[which]
                # تبدیل به pandas و سپس به CSV
                df_csv = ds.to_pandas()
                buf = io.StringIO()
                df_csv.to_csv(buf, index=False)
                buf.seek(0)
                filename = f"{which}_sts.csv"
                resp = HttpResponse(buf.getvalue(), content_type="text/csv; charset=utf-8")
                resp["Content-Disposition"] = f'attachment; filename="{filename}"'
                return resp

            # خروجی پیش‌فرض: JSON شامل سایزها و نمونه‌ها
            def sample(ds, k):
                k = min(k, len(ds))
                return [
                    {"sentence1": ds[i]["sentence1"], "sentence2": ds[i]["sentence2"], "score": float(ds[i]["score"])}
                    for i in range(k)
                ]

            payload = {
                "sizes": {
                    "train": len(train_ds),
                    "eval": len(eval_ds),
                    "test": len(test_ds),
                },
                "params": {
                    "test_size": test_size,
                    "val_size": val_size,
                    "seed": seed,
                    "min_len": min_len,
                    "include_unlabeled": include_unlabeled,
                },
                "samples": {
                    "train": sample(train_ds, sample_n),
                    "eval": sample(eval_ds, sample_n),
                    "test": sample(test_ds, sample_n),
                },
            }
            return JsonResponse(payload, json_dumps_params={"ensure_ascii": False})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


@require_http_methods(["GET", "POST"])
def train_and_evaluate_model(request):
    """نمایش تأیید آموزش مدل و اجرای آموزش در صورت تأیید"""
    if request.method == "POST":
        try:
            # دریافت پارامترها از فرم یا مقدار پیش‌فرض
            SEED = int(request.POST.get("seed", 42))
            num_train_epochs = int(request.POST.get("num_train_epochs", 1))
            batch_size = int(request.POST.get("batch_size", 64))
            learning_rate = float(request.POST.get("learning_rate", 2e-5))

            # اجرای آموزش
            result = train_sbert_model(
                SEED=SEED,
                num_train_epochs=num_train_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            # ارسال نتیجه به قالب success.html
            return render(request, "train_success.html", {
                "train_size": result["train_size"],
                "eval_size": result["eval_size"],
                "test_size": result["test_size"],
                "pre_metrics": result["pre_train_metrics"],
                "post_metrics": result["post_train_metrics"],
                "model_dir": result["final_dir"],
            })

        except Exception as e:
            return render(request, "train.html", {"error": str(e)})

    # حالت GET → نمایش فرم تأیید
    return render(request, "train.html")
# @require_GET
# def train_and_evaluate_model(request):
#     """ویو: آموزش + ارزیابی + بازگشت نتیجه به‌صورت JSON (بدون traceback)"""
#     try:
#         SEED = int(request.GET.get("seed", 42))
#         num_train_epochs = int(request.GET.get("num_train_epochs", 1))
#         batch_size = int(request.GET.get("batch_size", 64))
#         learning_rate = float(request.GET.get("learning_rate", 2e-5))
#
#         result = train_sbert_model(
#             SEED=SEED,
#             num_train_epochs=num_train_epochs,
#             batch_size=batch_size,
#             learning_rate=learning_rate
#         )
#         return JsonResponse(
#             {"ok": True, "result": result},
#             status=200,
#             json_dumps_params={"ensure_ascii": False}
#         )
#
#     except Exception as e:
#         msg = str(e) or repr(e) or "Unhandled error"
#         return JsonResponse(
#             {"ok": False, "error": msg},
#             status=400,
#             json_dumps_params={"ensure_ascii": False}
#         )

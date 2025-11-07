
from django.urls import path

from search import views
from search.views import search_sbert_view, search_elastic_view, success_view, upload_excel_view, \
    log_user_selection_elastic_view, combined_search_view, delete_all_data, delete_success, build_sts_dataset_view, \
    train_and_evaluate_model, upload_excel_trainModel_view, search_sbert_train_model_view

urlpatterns = [
    path('elastic/', views.search_elastic_view, name='search_elastic_view'),
    path('sbert/', views.search_sbert_view, name='search_sbert_view'),
    path('sbert-train-model/', views.search_sbert_train_model_view, name='search_sbert_train_model_view'),
    path('success/', views.success_view, name='success_url'),
    path('upload-excel/', upload_excel_view, name='upload_excel'),
    path('upload-excel-trainModel/', upload_excel_trainModel_view, name='upload_trainModel_excel'),
    path('log-user-selection/', views.log_user_selection_sbert_view, name='log_user_selection_sbert_view'),
    path('log_user_selection_elastic_view/', log_user_selection_elastic_view, name='log_user_selection_elastic_view'),
    path('combined/', combined_search_view, name='combined_search_view'),
    path('log-search/', views.log_search_view, name='log_search_view'),
    path('delete-all-data/', delete_all_data, name='delete_all_data'),
    path('delete-success/', delete_success, name='delete_success'),
    path('build_dataset/', build_sts_dataset_view, name='build_sts_dataset'),
    path('train_evaluate/', train_and_evaluate_model, name='train_and_evaluate_model'),

]


from django.db import models
import json

from django.utils.timezone import now


class DocumentEmbedding(models.Model):
    layer = models.CharField(max_length=255)
    system = models.CharField(max_length=255)
    dataModel = models.IntegerField()
    embeddingLayer = models.BinaryField()

    class Meta:
        db_table = 'embeddings'
        verbose_name = 'Document Embedding'
        verbose_name_plural = 'Document Embeddings'
        ordering = ['id']
        indexes = [
            models.Index(fields=['layer'], name='layer_idx'),
            models.Index(fields=['system'], name='system_idx'),
        ]

    def __str__(self):
        return f"{self.layer} - {self.system}"

class DocumentEmbeddingTrainModel(models.Model):
    layer = models.CharField(max_length=255)
    system = models.CharField(max_length=255)
    dataModel = models.IntegerField()
    # embeddingLayer = models.BinaryField()

    class Meta:
        db_table = 'embeddings_train_model'
        verbose_name = 'Document Embedding Train Model'
        verbose_name_plural = 'Document Embeddings Train Models'
        ordering = ['id']
        indexes = [
            models.Index(fields=['layer'], name='train_model_layer_idx'),
            models.Index(fields=['system'], name='train_model_system_idx'),
        ]

    def __str__(self):
        return f"{self.layer} - {self.system}"


class LogUserSearch(models.Model):
    query = models.CharField(max_length=255, verbose_name="کوئری کاربر")
    selected_layer = models.CharField(max_length=255, verbose_name="لایه انتخاب شده")
    selected_organization = models.CharField(max_length=255, verbose_name="سازمان انتخاب شده")
    relevance = models.IntegerField(verbose_name="مرتبط بودن با کوئری")  # 1, -1 یا 0
    score = models.FloatField(default=0.0, verbose_name="امتیاز")
    type_result = models.IntegerField(default=0, verbose_name="نوع نتایج")
    timestamp = models.DateTimeField(default=now, verbose_name="زمان")

    class Meta:
        db_table = "log_user_search"
        verbose_name = "نتایج جستجوی کاربر"
        verbose_name_plural = "نتایج جستجوهای کاربران"
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['query'], name='query_idx'),
        ]

    def __str__(self):
        return f"{self.query} - {self.selected_layer} - {self.selected_organization} - {self.relevance} - {self.score} - {self.type_result} ({self.timestamp})"

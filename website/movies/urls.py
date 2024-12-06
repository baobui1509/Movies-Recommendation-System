from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('recommend/index/', views.recommend_index, name='recommend_index'),
    path('genres/whitelist/', views.genre_whitelist, name='genre_whitelist'),
]

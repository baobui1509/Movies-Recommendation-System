from django.urls import path
from . import views

urlpatterns = [
    path('recommend/index/', views.recommend_index, name='recommend_index'), 
    path('recommend/collaborative_filtering/', views.collaborative_filtering, name='collaborative_filtering'), 
    path('genres/whitelist/', views.genre_whitelist, name='genre_whitelist'), 
]

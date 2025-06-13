import pandas as pd
import joblib  # Để tải mô hình đã lưu
from django.shortcuts import render
from .forms import MovieSearchForm, MovieSearchFormCollaborative
import os
from django.http import JsonResponse
from .recommendation import *
import ast
from io import StringIO



BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/movie_data_processed.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw/movie_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data/processed/simple_recommendations.pkl")
processed_data = pd.read_csv(PROCESSED_DATA_PATH)
processed_data['wr'] = pd.to_numeric(processed_data['wr'], errors='coerce')

def get_genre_unique():
    data = pd.read_csv(PROCESSED_DATA_PATH)
    genres = data['Genre'].dropna().unique()  
    genre_string = ', '.join(genres)
    unique_genres = sorted(set([genre.strip() for genre in genre_string.split(',')]))
    return unique_genres

genres_unique = get_genre_unique()

def genre_whitelist(request):
    query = request.GET.get('query', '').lower()
    suggestions = [genre for genre in genres_unique if genre.lower().startswith(query)]
    return JsonResponse(suggestions, safe=False)

def recommend_index(request):
    form = MovieSearchForm()
    movies = []
    empty = False
    sliders = processed_data.sort_values(by='wr', ascending=False).head(10).to_dict(orient='records')

    if request.method == 'POST':
        form = MovieSearchForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            sorted_by = form.cleaned_data.get('sorted_by')  
            genre = form.cleaned_data.get('genre')
            print(name, sorted_by, genre)
            try:
                movies = get_content_based_recommendations(name, similarity)
                if genre:
                    try:
                        genre_values = [item['value'].lower() for item in ast.literal_eval(genre) if 'value' in item]
                        print('genre_values: ', genre_values)
                        movies = movies[movies['Genre'].apply(lambda x: any(genre in x for genre in genre_values))]

                    except Exception as e:
                        print(f"Error processing genre filter: {e}")
                        
                movies = movies.merge(processed_data[['imdbID', 'Poster']], on='imdbID', how='left')
                movies = movies.sort_values(by=sorted_by, ascending=False)
                # print('len(movies):', len(movies))
                if len(movies) == 0: empty = True
                print(movies)
            except KeyError:
                empty = True
        else:
            print(form.errors)
            
    if isinstance(movies, list):
        movies = pd.DataFrame(movies)

    movies = movies.to_dict(orient='records')
    
    

    return render(request, 'index.html', {
        'form': form,
        'movies': movies,
        'empty': empty,
        'sliders': sliders,
    })

def collaborative_filtering(request):
    print("collaborative_filtering")
    form = MovieSearchFormCollaborative()
    movies = []
    empty = False
    sliders = processed_data.sort_values(by='wr', ascending=False).head(10).to_dict(orient='records')

    if request.method == 'POST':
        print("POSTTTTTTTTTT")
        form = MovieSearchFormCollaborative(request.POST)
        if form.is_valid():
            ID = int(form.cleaned_data['ID'])
            print("ID: ", ID)
            sorted_by = form.cleaned_data.get('sorted_by')  
            genre = form.cleaned_data.get('genre')
            print(ID, sorted_by, genre)
            try:
                movies = recommend_movies_df(ID, 10)
                print("len(movies): ", len(movies))
                if genre:
                    try:
                        genre_values = [item['value'].lower() for item in ast.literal_eval(genre) if 'value' in item]
                        print('genre_values: ', genre_values)
                        movies = movies[movies['Genre'].apply(lambda x: any(genre in x for genre in genre_values))]

                    except Exception as e:
                        print(f"Error processing genre filter: {e}")
                        
                movies = movies.merge(processed_data[['imdbID', 'Poster']], on='imdbID', how='left')
                movies = movies.sort_values(by=sorted_by, ascending=False)
                print('len(movies):', len(movies))
                if len(movies) == 0: empty = True
                print(movies)
            except KeyError:
                empty = True
        else:
            print("ERROR: ", form.errors)
            
    if isinstance(movies, list):
        movies = pd.DataFrame(movies)

    movies = movies.to_dict(orient='records')
    
    

    return render(request, 'collaborative.html', {
        'form': form,
        'movies': movies,
        'empty': empty,
        'sliders': sliders,
    })

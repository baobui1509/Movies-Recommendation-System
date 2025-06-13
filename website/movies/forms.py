from django import forms
import pandas as pd
import os


def get_genre_choices():
    # Đọc dữ liệu từ file CSV
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, "data/processed/movie_data_processed.csv")
    data = pd.read_csv(DATA_PATH)

    # Lấy danh sách thể loại duy nhất
    genres = data['Genre'].dropna().unique()  # Bỏ các giá trị NaN nếu có
    genre_choices = [('', 'Tất cả')] + [(genre, genre) for genre in genres]
    return genre_choices

class MovieSearchForm(forms.Form):
    name = forms.CharField(max_length=100, required=True)
    genre = forms.CharField(max_length=100, required=False)
    sorted_by = forms.ChoiceField(
        choices=[('imdbRating', 'IMDB'), ('Year', 'Year')],
        initial='imdbRating'
    )

class MovieSearchFormCollaborative(forms.Form):
    ID = forms.CharField(max_length=100, required=True)
    genre = forms.CharField(max_length=100, required=False)
    sorted_by = forms.ChoiceField(
        choices=[('imdbRating', 'IMDB'), ('Year', 'Year')],
        initial='imdbRating'
    )


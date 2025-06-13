import requests

def get_movie_data(imdb_id, api_key):
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
    print(url)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Trả về dữ liệu JSON từ OMDB API
    else:
        print(response)
        return None

print(get_movie_data("tt28995566", "431d9c38"))

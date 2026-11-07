import pandas as pd

# Настраиваем Pandas, чтобы таблицы не переносились на новые строки в узком терминале
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

# Функция для точного формирования таблиц с рамками в стиле задания
def print_custom_table(df_to_print):
    # Сбрасываем индекс, чтобы он стал обычной колоночкой слева (как на скриншотах задания)
    df_reset = df_to_print.reset_index()
    
    # Переводим в строковый вид
    table_str = df_reset.to_string(index=False)
    lines = table_str.split('\n')
    
    # Рисуем рамки
    border = '+' + '-' * (len(lines[0])) + '+'
    print(border)
    for line in lines:
        print(f"|{line}|")
    print(border)

# 1. Загрузка датасета[cite: 1]
df = pd.read_csv('netflix_data.csv')

# 2. Приведение числовых колонок к корректным типам[cite: 1]
df['releaseYear'] = pd.to_numeric(df['releaseYear'], errors='coerce')
df['imdbAverageRating'] = pd.to_numeric(df['imdbAverageRating'], errors='coerce')
df['imdbNumVotes'] = pd.to_numeric(df['imdbNumVotes'], errors='coerce')

# 3. Удаление строк с пропущенным IMDb рейтингом и жанром[cite: 1]
df = df.dropna(subset=['imdbAverageRating', 'genres'])

# 4. Удаление дубликатов по колонкам: title, releaseYear и imdbId[cite: 1]
df = df.drop_duplicates(subset=['title', 'releaseYear', 'imdbId'])

# Подготовка данных с разбивкой жанров
df_exploded = df.assign(genre=df['genres'].str.split(', ')).explode('genre')
df_exploded['genre'] = df_exploded['genre'].str.strip()


# --- ЗАДАНИЕ 1 ---
print("\n--- Результат задания 1 (По годам) ---")
recent_years = df.dropna(subset=['releaseYear']).copy()
recent_years['releaseYear'] = recent_years['releaseYear'].astype(int)
recent_years = recent_years[recent_years['releaseYear'] >= 2016]

pivot_years = recent_years.pivot_table(index='releaseYear', columns='type', values='title', aggfunc='count').fillna(0)
pivot_years = pivot_years.astype(int)
pivot_years = pivot_years.sort_index(ascending=False)
print_custom_table(pivot_years)


print("\n--- Результат задания 1 (Топ-15 жанров) ---")
pivot_genres = df_exploded.pivot_table(index='genre', columns='type', values='title', aggfunc='count').fillna(0)
pivot_genres = pivot_genres.astype(int)
pivot_genres['total'] = pivot_genres.sum(axis=1)
top_15_genres = pivot_genres.sort_values(by='total', ascending=False).head(15).drop(columns=['total'])
print_custom_table(top_15_genres)


# --- ЗАДАНИЕ 2 ---
print("\n--- Результат задания 2 ---")
genre_popularity = df_exploded.groupby('genre').agg(
    title_count=('title', 'count'),
    average_rating=('imdbAverageRating', 'mean'),
    total_votes=('imdbNumVotes', 'sum')
).reset_index()

genre_popularity['average_rating'] = genre_popularity['average_rating'].round(2)
genre_popularity['total_votes'] = genre_popularity['total_votes'].astype(int)
top_10_rating_genres = genre_popularity.sort_values(by='average_rating', ascending=False).head(10)
print_custom_table(top_10_rating_genres)


# --- ЗАДАНИЕ 3 ---
print("\n--- Результат задания 3 ---")
df_valid_years = df.dropna(subset=['releaseYear']).copy()
df_valid_years['releaseYear'] = df_valid_years['releaseYear'].astype(int)

yearly_dynamics = df_valid_years.groupby('releaseYear').agg(
    total_titles=('title', 'count'),
    average_rating=('imdbAverageRating', 'mean')
).reset_index()

yearly_dynamics['average_rating'] = yearly_dynamics['average_rating'].round(2)
recent_dynamics = yearly_dynamics[yearly_dynamics['releaseYear'] >= 2016].sort_values(by='releaseYear', ascending=False)
print_custom_table(recent_dynamics)


# --- ЗАДАНИЕ 4 ---
print("\n--- Результат задания 4 ---")
movies_only = df[(df['type'] == 'movie') & (df['releaseYear'].notna())].copy()
movies_only['releaseYear'] = movies_only['releaseYear'].astype(int)

yearly_movies = movies_only.groupby('releaseYear').size().reset_index(name='current_year_title_count')
yearly_movies = yearly_movies.sort_values('releaseYear')
yearly_movies['previous_year_title_count'] = yearly_movies['current_year_title_count'].shift(1)
yearly_movies['title_count_growth'] = yearly_movies['current_year_title_count'] - yearly_movies['previous_year_title_count']

growth_top = yearly_movies.dropna().sort_values(by='title_count_growth', ascending=False).head(10).copy()

growth_top['releaseYear'] = growth_top['releaseYear'].astype(int)
growth_top['current_year_title_count'] = growth_top['current_year_title_count'].astype(int)
growth_top['previous_year_title_count'] = growth_top['previous_year_title_count'].astype(int)
growth_top['title_count_growth'] = growth_top['title_count_growth'].astype(int)

print_custom_table(growth_top)


# --- ЗАДАНИЕ 5 ---
print("\n--- Результат задания 5 ---")
hidden_gems = df[
    (df['imdbAverageRating'] > 8.0) & 
    (df['imdbNumVotes'] >= 300) & 
    (df['imdbNumVotes'] <= 10000)
]

top_gems = hidden_gems.sort_values(by='imdbAverageRating', ascending=False).head(10)
result_gems = top_gems[['title', 'type', 'genres', 'releaseYear', 'imdbId', 'imdbAverageRating', 'imdbNumVotes', 'availableCountries']].copy()
result_gems['releaseYear'] = result_gems['releaseYear'].fillna(0).astype(int)
result_gems['imdbNumVotes'] = result_gems['imdbNumVotes'].astype(int)


result_gems['availableCountries'] = result_gems['availableCountries'].fillna('NULL')

print_custom_table(result_gems)
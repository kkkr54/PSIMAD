from datetime import date
import os
import sys

# Системные фиксы для корректной работы PySpark на Windows
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# Инициализация SparkSession
spark = (
    SparkSession.builder.master("local[*]")
    .appName("netflix_pyspark_analysis")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")


# Функция для красивого вывода таблиц с рамками (использует Pandas DataFrame на выходе)
def print_custom_table(df_spark):
  df_pd = df_spark.toPandas()
  df_reset = df_pd.reset_index(drop=True)
  table_str = df_reset.to_string(index=False)
  lines = table_str.split("\n")
  border = "+" + "-" * (len(lines[0])) + "+"
  print(border)
  for line in lines:
    print(f"|{line}|")
  print(border)


from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

# 1. Загрузка датасета с принудительным чтением всех колонок как строк, чтобы избежать ошибок парсинга
file_path = "netflix_data.csv"
df_raw = spark.read.option("header", "true").option("inferSchema", "false").csv(file_path)

# 2. Очистка и корректное приведение типов с проверкой на валидность года
df_cleaned = (
    df_raw
    # Оставляем только те строки, где releaseYear состоит из 4 цифр (отсекаем смещенные строки из-за запятых в CSV)
    .filter(F.col("releaseYear").rlike("^[0-9]{4}$"))
    .withColumn("releaseYear", F.col("releaseYear").cast("integer"))
    .withColumn("imdbAverageRating", F.col("imdbAverageRating").cast("double"))
    .withColumn("imdbNumVotes", F.col("imdbNumVotes").cast("integer"))
    .filter(F.col("imdbAverageRating").isNotNull() & F.col("genres").isNotNull())
    .dropDuplicates(["title", "releaseYear", "imdbId"])
)

# Подготовка данных с разбивкой жанров
df_exploded = (
    df_cleaned
    .withColumn("genre", F.explode(F.split(F.col("genres"), ", ")))
    .withColumn("genre", F.trim(F.col("genre")))
)

df_exploded.cache()

# Подготовка данных с разбивкой жанров (сначала разворачиваем массив, затем убираем пробелы)
df_exploded = (
    df_cleaned
    .withColumn("genre", F.explode(F.split(F.col("genres"), ", ")))
    .withColumn("genre", F.trim(F.col("genre")))
)

# 캐ширование для ускорения повторных вызовов
df_exploded.cache()


# --- ЗАДАНИЕ 1 ---
print("\n--- Результат задания 1 (По годам) ---")
recent_years = df_cleaned.filter(
    F.col("releaseYear").isNotNull() & (F.col("releaseYear") >= 2016)
)

pivot_years = (
    recent_years.groupBy("releaseYear")
    .pivot("type", ["movie", "tv"])
    .agg(F.count("title"))
    .na.fill(0)
)

# Приводим типы к int и сортируем по убыванию года
for col_name in pivot_years.columns:
  if col_name != "releaseYear":
    pivot_years = pivot_years.withColumn(
        col_name, F.col(col_name).cast("integer")
    )

pivot_years = pivot_years.orderBy(F.col("releaseYear").desc())
print_custom_table(pivot_years)


print("\n--- Результат задания 1 (Топ-15 жанров) ---")
pivot_genres = (
    df_exploded.groupBy("genre")
    .pivot("type", ["movie", "tv"])
    .agg(F.count("title"))
    .na.fill(0)
)

for col_name in pivot_genres.columns:
  if col_name != "genre":
    pivot_genres = pivot_genres.withColumn(
        col_name, F.col(col_name).cast("integer")
    )

pivot_genres = pivot_genres.withColumn(
    "total", F.col("movie") + F.col("tv")
)
top_15_genres = (
    pivot_genres.orderBy(F.col("total").desc())
    .limit(15)
    .drop("total")
)
print_custom_table(top_15_genres)


# --- ЗАДАНИЕ 2 ---
print("\n--- Результат задания 2 ---")
genre_popularity = (
    df_exploded.groupBy("genre")
    .agg(
        F.count("title").alias("title_count"),
        F.round(F.avg("imdbAverageRating"), 2).alias("average_rating"),
        F.sum("imdbNumVotes").cast("integer").alias("total_votes"),
    )
    .orderBy(F.col("average_rating").desc())
    .limit(10)
    .select("genre", "average_rating", "total_votes")
)
print_custom_table(genre_popularity)


# --- ЗАДАНИЕ 3 ---
print("\n--- Результат задания 3 ---")
yearly_dynamics = (
    df_cleaned.filter(F.col("releaseYear").isNotNull())
    .groupBy("releaseYear")
    .agg(
        F.count("title").alias("total_titles"),
        F.round(F.avg("imdbAverageRating"), 2).alias("average_rating"),
    )
    .filter(F.col("releaseYear") >= 2016)
    .orderBy(F.col("releaseYear").desc())
)
print_custom_table(yearly_dynamics)


# --- ЗАДАНИЕ 4 ---
print("\n--- Результат задания 4 ---")
movies_only = df_cleaned.filter(
    (F.col("type") == "movie") & F.col("releaseYear").isNotNull()
)

yearly_movies = (
    movies_only.groupBy("releaseYear")
    .agg(F.count("title").alias("current_year_title_count"))
    .orderBy("releaseYear")
)

window_spec = Window.orderBy("releaseYear")
yearly_movies_lag = yearly_movies.withColumn(
    "previous_year_title_count", F.lag("current_year_title_count", 1).over(window_spec)
)

growth_df = yearly_movies_lag.withColumn(
    "title_count_growth",
    F.col("current_year_title_count") - F.col("previous_year_title_count"),
).filter(F.col("previous_year_title_count").isNotNull())

growth_top = growth_df.orderBy(F.col("title_count_growth").desc()).limit(10)
print_custom_table(growth_top)


# --- ЗАДАНИЕ 5 ---
print("\n--- Результат задания 5 ---")
hidden_gems = df_cleaned.filter(
    (F.col("imdbAverageRating") > 8.0)
    & (F.col("imdbNumVotes") >= 300)
    & (F.col("imdbNumVotes") <= 10000)
)

top_gems = (
    hidden_gems.orderBy(F.col("imdbAverageRating").desc())
    .limit(10)
    .withColumn(
        "availableCountries",
        F.coalesce(F.col("availableCountries"), F.lit("NULL")),
    )
    .select(
        "title",
        "type",
        "genres",
        "releaseYear",
        "imdbId",
        "imdbAverageRating",
        "imdbNumVotes",
        "availableCountries",
    )
)
print_custom_table(top_gems)

# Завершение сессии Spark
spark.stop()
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *

input_path = sys.argv[1]
output_path = sys.argv[2]

def init_spark():
    spark = SparkSession \
        .builder \
        .enableHiveSupport() \
        .config("spark.sql.session.state.builder",
                "org.apache.spark.sql.hive.UQueryHiveACLSessionStateBuilder") \
        .config("spark.sql.catalog.class",
                "org.apache.spark.sql.hive.UQueryHiveACLExternalCatalog") \
        .config("spark.sql.extensions",
                ','.join(["org.apache.spark.sql.CarbonInternalExtensions", "org.apache.spark.sql.DliSparkExtension"])) \
        .config("spark.sql.parquet.compression.codec", "gzip") \
        .appName("feature engineering Task") \
        .getOrCreate()
    return spark

def main():
    spark = init_spark()
    spark.sql(f"""
    create table if not exists etl_mlops_cicd.ml_1m_train_df
    (
        userId string,
        movieId string,
        rating int,
        Movie_names string,
        Genres string,
        release_year string,
        gender string,
        age int,
        occupation string
    )
    stored as
        parquet
    LOCATION
        '{output_path}'
    """)
    
    
    # movie_dat_schema = StructType([StructField('movieId', StringType(), True), StructField('Movie_names', StringType(), True), StructField('Genres', StringType(), True)])
    # user_dat_schema = StructType([StructField('userId', StringType(), True), StructField('gender', StringType(), True), StructField('age', StringType(), True), StructField('occupation', StringType(), True), StructField('zip', StringType(), True)])
    # rating_dat_schema = StructType([StructField('userId', StringType(), True), StructField('movieId', StringType(), True), StructField('rating', LongType(), True), StructField('time', StringType(), True)])
    
    movies = spark.read.csv(f"{input_path}/movies.csv", header=True).toPandas()
    users = spark.read.csv(f"{input_path}/users.csv", header=True).toPandas()
    reviews = spark.read.csv(f"{input_path}/reviews.csv", header=True).toPandas()
    
    reviews.drop(['time'], axis=1, inplace=True)
    users.drop(['zip'], axis=1, inplace=True)
    movies['release_year'] = movies['Movie_names'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)
    final_df = spark.createDataFrame(reviews.merge(movies, on='movieId', how='left').merge(users, on='userId', how='left'))
    final_df.createOrReplaceTempView("final_df")
    
    spark.sql(f"""
    insert overwrite table etl_mlops_cicd.ml_1m_train_df
    select * from final_df
    """)
    
    print("数据量：", spark.sql("""select * from etl_mlops_cicd.ml_1m_train_df""").count())

main()
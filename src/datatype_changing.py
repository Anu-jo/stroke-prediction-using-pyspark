from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType,DoubleType
spark = SparkSession.builder.appName("dataPreprocessing").getOrCreate()
df = spark.read.csv('C:/Users/anujo/Desktop/M.Sc data science/big data/healthcare-dataset-stroke-data.csv', header=True, inferSchema=True)
df = df.withColumn("id", df["id"].cast(IntegerType()))\
     .withColumn("age", df["age"].cast(IntegerType()))\
     .withColumn("hypertension", df["hypertension"].cast(IntegerType()))\
     .withColumn("heart_disease", df["heart_disease"].cast(IntegerType()))\
     .withColumn("avg_glucose_level", df["avg_glucose_level"].cast(DoubleType()))\
     .withColumn("bmi", df["bmi"].cast(DoubleType()))\
     .withColumn("stroke", df["stroke"].cast(IntegerType()))
     

df.printSchema()
#data.write.mode("overwrite").parquet("C:/Users/anujo/Desktop/M.Sc data science/big data/healthcare-dataset-stroke-data1.csv")
df=df.toPandas()
df.to_csv("C:/Users/anujo/Desktop/M.Sc data science/big data/healthcare-dataset-stroke-data.csv",index=False)

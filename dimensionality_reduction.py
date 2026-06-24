from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorAssembler
spark = SparkSession.builder.appName("dimensionalityReduction").getOrCreate()
from pyspark.ml.linalg import Vectors
import pandas
from pyspark.sql.types import StructType, StructField, DoubleType
import pyspark.sql.types as T
# Load the dataset into a PySpark DataFrame
data = spark.read.csv("C:/Users/anujo/Desktop/M.Sc data science/big data/stroke_dataset.csv", header=True, inferSchema=True)
gender_indexer = StringIndexer(inputCol='gender', outputCol='gender_index')
ever_married_indexer = StringIndexer(inputCol='ever_married', outputCol='ever_married_index')
work_type_indexer = StringIndexer(inputCol='work_type', outputCol='work_type_index')
residence_type_indexer = StringIndexer(inputCol='Residence_type', outputCol='residence_type_index')
smoking_status_indexer = StringIndexer(inputCol='smoking_status', outputCol='smoking_status_index')
gender_encoder = OneHotEncoder(inputCol='gender_index', outputCol='gender_encoded')
ever_married_encoder = OneHotEncoder(inputCol='ever_married_index', outputCol='ever_married_encoded')
work_type_encoder = OneHotEncoder(inputCol='work_type_index', outputCol='work_type_encoded')
residence_type_encoder = OneHotEncoder(inputCol='residence_type_index', outputCol='residence_type_encoded')
smoking_status_encoder = OneHotEncoder(inputCol='smoking_status_index', outputCol='smoking_status_encoded')
# Assemble the features
assembler = VectorAssembler(
    inputCols=['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
               'bmi', 'gender_encoded', 'ever_married_encoded', 'work_type_encoded',
               'residence_type_encoded', 'smoking_status_encoded'],
    outputCol='features')

# Apply the transformations
data = gender_indexer.fit(data).transform(data)
data = ever_married_indexer.fit(data).transform(data)
data = work_type_indexer.fit(data).transform(data)
data = residence_type_indexer.fit(data).transform(data)
data = smoking_status_indexer.fit(data).transform(data)

gender_encoder_new = gender_encoder.fit(data)
ever_married_encoder_new = ever_married_encoder.fit(data)
work_type_encoder_new = work_type_encoder.fit(data)
residence_type_encoder_new = residence_type_encoder.fit(data)
smoking_status_encoder_new = smoking_status_encoder.fit(data)

data = gender_encoder_new.transform(data)
data = ever_married_encoder_new.transform(data)
data = work_type_encoder_new.transform(data)
data = residence_type_encoder_new.transform(data)
data = smoking_status_encoder_new.transform(data)

data = assembler.transform(data).select("features")

# Perform PCA on the data

pca = PCA(k=4, inputCol='features', outputCol='pca_features')
pca_model = pca.fit(data)
pca_result = pca_model.transform(data).select("pca_features")
#pca_result.select("pca_features").show()
#pca_result.show()
explained_variance = pca_model.explainedVariance
cumulative_variance = [sum(explained_variance[:i+1]) for i in range(len(explained_variance))]
print(cumulative_variance)
from pyspark.sql.functions import split
from pyspark.sql.functions import concat_ws

pca_result = pca_result.withColumn('pca_features_str', concat_ws(',', pca_result.pca_features.cast('array<string>')))

# Assuming 'df' is your PySpark dataframe
pca_result = pca_result.withColumn('PC1', split(pca_result.pca_features_str, ',')[0])
pca_result = pca_result.withColumn('PC2', split(pca_result.pca_features_str, ',')[1])
pca_result = pca_result.withColumn('PC3', split(pca_result.pca_features_str, ',')[2])
pca_result = pca_result.withColumn('PC4', split(pca_result.pca_features_str, ',')[3])
pca_result_table=pca_result.toPandas()
pca_result_table.to_csv("C:/Users/anujo/Desktop/M.Sc data science/big data/pca_result.csv",index=False)
#schema = T.StructType([T.StructField("cumulative_variance", T.FloatType(), True)])

# Convert numpy.float64 to regular Python float
#cumulative_variance = [float(x) for x in cumulative_variance]

#df = spark.createDataFrame([(x,) for x in cumulative_variance], schema)
#df.toPandas()
#df.show()

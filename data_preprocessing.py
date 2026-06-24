# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:00:36 2023

@author: anujo
"""
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan,when,count
import warnings
warnings.filterwarnings("ignore")
spark = SparkSession.builder.appName("dataPreprocessing").getOrCreate()
df = spark.read.csv('data/healthcare-dataset-stroke-data.csv', header=True, inferSchema=True)
df.printSchema()
df.show()
df_column=["bmi","smoking_status"]
df2 = df.select([count(when(col(c).contains('Unknown') | \
                            col(c).contains('N/A') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in df_column])
df2.show()
df_cleaned=df.where("smoking_status <> 'Unknown'")
df_cleaned.show()
df_new1=df_cleaned.toPandas()
df_new1.to_csv("data/stroke_dataset.csv",index=False)
#----data sampling
major_class = df_cleaned.filter(col("stroke") == 0)
minor_class = df_cleaned.filter(col("stroke") == 1)
ratio = int(major_class.count()/minor_class.count())
# #-----data oversmpling
# a = range(ratio)
# # duplicate the minority rows
# oversampled_df = minor_class.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
# # combine both oversampled minority rows and previous majority rows 
# oversampled_data = major_class.union(oversampled_df)
# oversampled_data.show()
counts = df_cleaned.groupBy("stroke").count()
# Get the count of the majority class
minority_count = counts.filter(counts.stroke == 1).select("count").collect()[0]["count"]
majority_count = counts.filter(counts.stroke == 0).select("count").collect()[0]["count"]

# Oversample the minority class
minority_df = df_cleaned.filter(df_cleaned.stroke == 1)
oversampled_minority_df = minority_df.sample(True, majority_count/minority_count)


# Combine the oversampled minority class and majority class
oversampled_data = oversampled_minority_df.union(df_cleaned.filter(df_cleaned.stroke == 0))
# oversampled_heart_df.show()
print("The data set size is:",oversampled_data.count(),len(oversampled_data.columns))
oversampled_data1=oversampled_data.toPandas()
oversampled_data1.to_csv("outputs/oversampled_dataset.csv",index=False)
#----data undersmpling
sampled_majority_df = major_class.sample(True, 1/ratio)
undersampled_data = sampled_majority_df.union(minor_class)
undersampled_data.show()
print("The data set size is:",undersampled_data.count(),len(undersampled_data.columns))
undersampled_data1=undersampled_data.toPandas()
undersampled_data1.to_csv("outputs/undersampled_dataset.csv",index=False)
        
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# create a SparkSession object
spark = SparkSession.builder.appName("MyApp").getOrCreate()

# set the logging level to ERROR
spark.sparkContext.setLogLevel("ERROR")


#spark = SparkSession.builder.appName("machineLearningModels").getOrCreate()
#data = spark.read.format("csv").option("header", "true").load("C:/Users/anujo/Desktop/M.Sc data science/big data/undersampled_dataset.csv")
string_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
indexers = [StringIndexer(inputCol=col, outputCol=col+'_index').fit(oversampled_data) for col in string_columns]
pipeline = Pipeline(stages=indexers)
oversampled_data = pipeline.fit(oversampled_data).transform(oversampled_data)

encoder = OneHotEncoder(inputCols=[col+'_index' for col in string_columns],
                        outputCols=[col+'_vec' for col in string_columns])
data = encoder.fit(oversampled_data).transform(oversampled_data)

# Drop the original string columns
data = data.drop(*string_columns)
# create a vector assembler to combine the input features into a single vector column
assembler = VectorAssembler(
    inputCols=[ 'gender_vec', 'ever_married_vec', 'work_type_vec',
               'Residence_type_vec', 'smoking_status_vec'],
    outputCol='features')

# apply the vector assembler to the input data
data = assembler.transform(data)

# split the data into training and test sets
training, test = data.randomSplit([0.7, 0.3])

# logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="stroke")
model = lr.fit(training)
predictions_log = model.transform(test)
evaluator_log = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
accuracy_log = evaluator_log.evaluate(predictions_log)
print("Accuracy of logistic regression classifier model= %g" % accuracy_log)
print("Test Error = %g" % (1.0 - accuracy_log))
predictions_log=predictions_log.toPandas()
predictions_log.to_csv("outputs/logistic_regression.csv",index=False)

#decision tree model
dt = DecisionTreeClassifier(labelCol="stroke", featuresCol="features",predictionCol="prediction_dec")
model = dt.fit(training)
predictions_decision = model.transform(test)
evaluator_decision = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction_dec", metricName="accuracy")
accuracy_decision = evaluator_decision.evaluate(predictions_decision)
print("Accuracy of decision tree classifier model= %g" % accuracy_decision)
print("Test Error = %g" % (1.0 - accuracy_decision))
predictions_decision=predictions_decision.toPandas()
predictions_decision.to_csv("outputs/decision_tree.csv",index=False)

#random forest model
rf = RandomForestClassifier(labelCol="stroke", featuresCol="features",predictionCol="prediction_ran", numTrees=10)
model = rf.fit(training)
predictions_random = model.transform(test)
evaluator_random = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction_ran", metricName="accuracy")
accuracy_random = evaluator_random.evaluate(predictions_random)
print("Accuracy random forest classifier model= %g" % accuracy_random)
print("Test Error = %g" % (1.0 - accuracy_random))
predictions_random=predictions_random.toPandas()
predictions_random.to_csv("outputs/random_forest.csv",index=False)

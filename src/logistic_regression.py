from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# create a SparkSession object
spark = SparkSession.builder.appName("MyApp").getOrCreate()

# set the logging level to ERROR
spark.sparkContext.setLogLevel("ERROR")


spark = SparkSession.builder.appName("machineLearningModels").getOrCreate()
data = spark.read.format("csv").option("header", "true").load("C:/Users/anujo/Desktop/M.Sc data science/big data/undersampled_dataset.csv")
string_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
indexers = [StringIndexer(inputCol=col, outputCol=col+'_index').fit(data) for col in string_columns]
pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)

encoder = OneHotEncoder(inputCols=[col+'_index' for col in string_columns],
                        outputCols=[col+'_vec' for col in string_columns])
data = encoder.fit(data).transform(data)

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

#decision tree model
dt = DecisionTreeClassifier(labelCol="stroke", featuresCol="features")
model = dt.fit(training)
predictions_decision = model.transform(test)
evaluator_decision = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
accuracy_decision = evaluator_decision.evaluate(predictions_decision)
print("Accuracy of decision tree classifier model= %g" % accuracy_decision)
print("Test Error = %g" % (1.0 - accuracy_decision))

#random forest model
rf = RandomForestClassifier(labelCol="stroke", featuresCol="features", numTrees=10)
model = rf.fit(training)
predictions_random = model.transform(test)
evaluator_random = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
accuracy_random = evaluator_random.evaluate(predictions_random)
print("Accuracy random forest classifier model= %g" % accuracy_random)
print("Test Error = %g" % (1.0 - accuracy_random))





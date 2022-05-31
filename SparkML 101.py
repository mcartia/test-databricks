# Databricks notebook source
# MAGIC %fs ls /training/ml/example/iris

# COMMAND ----------

# MAGIC %fs head dbfs:/training/ml/example/iris/iris.data

# COMMAND ----------

iris_df = spark.read.format("csv").option("inferSchema","true").load("/training/ml/example/iris/iris.data")

# COMMAND ----------

display(iris_df)

# COMMAND ----------

iris_colnames = ["sepal_length","sepal_width","petal_length","petal_width","label"]

iris_df = iris_df.toDF(*iris_colnames)

iris_df.printSchema()

# COMMAND ----------

display(iris_df)

# COMMAND ----------

import mlflow
from pyspark.ml.feature import VectorAssembler

mlflow.end_run()
mlflow.start_run()

assembler = VectorAssembler(inputCols=iris_colnames[:-1], outputCol="features")

display(output)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="label", outputCol="labelIdx")
indexed = indexer.fit(output).transform(output)

display(indexed)

# COMMAND ----------

from pyspark.ml.feature import MinMaxScaler

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

mlflow.log_metric("trainLen",0.8)
mlflow.log_metric("trainSamples",trainingData.count())

rf = RandomForestClassifier(labelCol="labelIdx", featuresCol="scaledFeatures", numTrees=10)
#model = rf.fit(trainingData)

# COMMAND ----------

predictions = model.transform(testData)

display(predictions)

# COMMAND ----------

from pyspark.ml import Pipeline

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[assembler, scaler, indexer, rf])

model = pipeline.fit()
mlflow.spark.log_model(model, artifact_path="training/ml/example/rf002")

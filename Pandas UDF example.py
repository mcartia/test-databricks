# Databricks notebook source
# MAGIC %sh
# MAGIC 
# MAGIC curl https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -o /tmp/iris.data

# COMMAND ----------

# MAGIC %fs mkdirs /training/ml/example/iris

# COMMAND ----------

# MAGIC %fs cp file:///tmp/iris.data dbfs:/training/ml/example/iris

# COMMAND ----------

iris_df = spark.read.format("csv").option("inferSchema","true").load("/training/ml/example/iris/iris.data")

iris_colnames = ["sepal_length","sepal_width","petal_length","petal_width","label"]
iris_df = iris_df.toDF(*iris_colnames)

display(iris_df)

# COMMAND ----------

from pyspark.sql.functions import lit

df1 = iris_df.withColumn("group_id",lit(1))
df2 = iris_df.withColumn("group_id",lit(2))

df_full = df1.union(df2)

# COMMAND ----------

from pyspark.sql.types import *
 
# define schema for what the pandas udf will return
schema = StructType([
StructField('group_id', IntegerType()),
StructField('num_instances_trained_with', IntegerType()),
StructField('model_str', StringType())
])

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from sklearn import preprocessing
import base64
 
@pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
def train_model(df_pandas):
    '''
    Trains a RandomForestRegressor model on training instances
    in df_pandas.
 
    Assumes: df_pandas has the columns:
                 ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
 
    Returns: a single row pandas DataFrame with columns:
               ['group_id', 'num_instances_trained_with', 
                'model_str']
    '''
 
    # get the value of this group id
    group_id = df_pandas['group_id'].iloc[0]
    print("Group ID: {}".format(group_id))
    
    # get the number of training instances for this group
    num_instances = df_pandas.shape[0]
 
    # get features and label for all training instances in this group
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    label = 'label';
    X = df_pandas[feature_columns]
    Y = df_pandas[label]
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y_encoded = encoder.transform(Y)
 
    # train this model
    model = RandomForestRegressor()
    model.fit(X,Y_encoded)
 
    # get a string representation of our trained model to store
    model_str = base64.b64encode(pickle.dumps(model))
 
    # build the DataFrame to return
    df_to_return = pd.DataFrame({'group_id': [group_id], 'num_instances_trained_with': num_instances, 'model_str': model_str})
 
    return df_to_return

# COMMAND ----------

df_trained_models = df_full.groupBy('group_id').apply(train_model)

# COMMAND ----------

df_trained_models.select("group_id","num_instances_trained_with").show()

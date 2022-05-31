# Databricks notebook source
# MAGIC %sh
# MAGIC curl https://raw.githubusercontent.com/apache/spark/master/examples/src/main/python/pi.py -o /tmp/pi.py

# COMMAND ----------

# MAGIC %sh 
# MAGIC cat /tmp/pi.py

# COMMAND ----------

# MAGIC %fs ls /training

# COMMAND ----------

# MAGIC %fs mkdirs /training/examples

# COMMAND ----------

# MAGIC %fs cp file:///tmp/pi.py dbfs:/training/examples

# COMMAND ----------

# MAGIC %fs ls /training/examples

# COMMAND ----------



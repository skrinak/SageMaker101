import boto3
import numpy as np
import pandas as pd
import io
import os
import sagemaker.amazon.common as smac
from sagemaker import get_execution_role

role = get_execution_role()
print(role)

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

# specify columns extracted from wbdc.names
data.columns = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
                "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
                "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                "concave points_worst","symmetry_worst","fractal_dimension_worst"] 


# save the data
data.to_csv("data.csv", sep=',', index=False)

# print the shape of the data file
print(data.shape)

# show the top few rows
display(data.head())

rand_split = np.random.rand(len(data))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

data_train = data[train_list]
data_val = data[val_list]
data_test = data[test_list]

train_y = ((data_train.iloc[:,1] == 'M') +0).as_matrix();
train_X = data_train.iloc[:,2:].as_matrix();

val_y = ((data_val.iloc[:,1] == 'M') +0).as_matrix();
val_X = data_val.iloc[:,2:].as_matrix();

test_y = ((data_test.iloc[:,1] == 'M') +0).as_matrix();
test_X = data_test.iloc[:,2:].as_matrix();

bucket = '2017-sagemaker'# enter your s3 bucket where you will copy data and model artifacts
prefix = 'sagemaker/breast_cancer_prediction' # place to upload training files within the bucket

train_file = 'linear_train.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_fileobj(f)

validation_file = 'linear_validation.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', train_file)).upload_fileobj(f)

test_file = 'test_data.csv'

data_test.to_csv(test_file, index=False)
with open(test_file, 'rb') as data:
    boto3.Session().resource('s3').Bucket(bucket).upload_fileobj(data, os.path.join(prefix, 'test', test_file))

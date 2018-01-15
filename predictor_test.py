import os
import io
import boto3
import json
import csv

# grab environment variables
bucketname = os.environ['BUCKET_NAME']
dataprefix = os.environ['DATA_PREFIX']
datafile = os.environ['DATA_FILE']
endpointname = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')
s3 = boto3.client('s3')

def load_test_data(bucketname, dataprefix, datafile, numrecords):
    print(bucketname+' - '+dataprefix+'/'+datafile)
    s3.download_file(bucketname, dataprefix+'/'+datafile, '/tmp/' + datafile)
    labels = []
    observations = []

    with open('/tmp/' + datafile) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        linecount = 0
        for row in csvReader:
            if row[0].isnumeric() and (numrecords <= 0 or linecount <= numrecords):
                observation = []
                for index in range(len(row)):
                    if index == 1:
                        if row[index] == "M":
                            labels.append('1')
                        else:
                            labels.append('0')
                    if index > 1:
                        observation.append(row[index])
                #observations.append(observation) 
                observations.append(','.join(observation))  
            linecount = linecount + 1   
    data = {'X': observations, 'y': labels}    
    return data
    
def lambda_handler(event, context):
    recordstotest = "0"
    if 'recordstotest' in event:
        recordstotest = event['recordstotest']
    try:
        numrecords = int(recordstotest)
    except ValueError:
        numrecords = 0 
    data = load_test_data(bucketname, dataprefix, datafile, numrecords)
    index = 0
    match=0
    mismatch=0
    for payload in data['X']:
        given_label = 'M' if data['y'][index] == '1' else 'B'
        print("Data Sample-{}: {}".format(index+1,payload))
        runtime= boto3.client('runtime.sagemaker')
        response = runtime.invoke_endpoint(EndpointName=endpointname,
                                           ContentType='text/csv',
                                           Body=payload)
        result = json.loads(response['Body'].read().decode())
        print(result)
        pred = int(result['predictions'][0]['predicted_label'])
        predicted_label = 'M' if pred == 1 else 'B'
        print("Given Label : {}, Predicted Label : {} - {}".format(given_label, predicted_label, 'Match' if given_label == predicted_label else 'Mismatch'))
        if given_label == predicted_label:
            match = match + 1
        else:
            mismatch = mismatch + 1
        index = index + 1
    print("Total Samples Tested : {}".format(index))
    print("Matches - {}({}%), Mismatches - {}({}%)".format(match,int(match*100/index),mismatch,100-int(match*100/index)))  
    result = {'SamplesTested' : index, 'TotalMatch' : match, 'TotalMismatch' : mismatch, 'Accuracy': int(match*100/index)}
    return result
# **Build End-to-End ML Pipeline for Truck Delay Classification**


The project addresses a critical challenge faced by the logistics industry. Delayed truck shipments not only result in increased operational costs but also impact customer satisfaction. Timely delivery of goods is essential to meet customer expectations and maintain the competitiveness of logistics companies.
By accurately predicting truck delays, logistics companies can:
* Improve operational efficiency by allocating resources more effectively
* Enhance customer satisfaction by providing more reliable delivery schedules
* Optimize route planning to reduce delays caused by traffic or adverse weather conditions
* Reduce costs associated with delayed shipments, such as penalties or compensation to customers

Build an End-to-End Machine Learning Pipeline - AWS RDS for data storage, setting up an AWS Sagemaker Notebook, performing data retrieval, conducting exploratory data analysis, and creating feature groups with Hopsworks. 

Machine-learning pipeline. Focusing on data retrieval from the feature store, train-validation-test split, one-hot encoding, scaling numerical features.  Build pipeline for model building with logistic regression, random forest, and XGBoost models. Explore hyperparameter tnning, grid and random search, and, ultimately, the deployment of a Streamlit application on AWS. 


**Note:  AWS Usage Charges**
This project leverages the AWS cloud platform to build the end-to-end machine learning pipeline. While using AWS services, it's important to note that certain activities may incur charges. We recommend exploring the AWS Free Tier, which provides limited access to a wide range of AWS services for 12 months. Please refer to the AWS Free Tier page for detailed information, including eligible services and usage limitations.

## **Approach**

* Data Retrieval from Hopsworks:
    * Connecting Hopsworks with Python.
    * Retrieving data directly from the feature store.


* Train-Validation-Test Split

* One-Hot Encoding

* Scaling Numerical Features

* Model Building
    * Logistic Regression
    * Random Forest
    * XGBoost


* Hyperparameter Tuning


* Streamlit Application Development and Fetching the Best Model


* Deployment on AWS EC2 Instance

## Folder Structure
```
# Execution Instructions

# Python version 3.10

To create a virtual environment and install requirements in Python 3.10 on different operating systems, follow the instructions below:




# Streamlit Application Deployment on AWS EC2

## Overview

This guide provides step-by-step instructions for deploying a Streamlit application on an AWS EC2 instance. 

## Prerequisites

- AWS Account
- Basic knowledge of AWS EC2, SSH, and Streamlit


## Deployment Steps

### 1. Launching EC2 Instance

- Launch an EC2 instance on AWS with the following specifications:
  - Ubuntu 22.04 LTS
  - Instance Type: t2.medium (or your preferred type)
  - Security Group: Allow inbound traffic on port 8501 for Streamlit

- Create and download a PEM key for SSH access to the EC2 instance.

- Disable Inheritance and Restrict Access on PEM key For Windows Users:
    - Locate the downloaded PEM key file (e.g., your-key.pem) using File Explorer.

    - Right-click on the PEM key file and select "Properties."

    - In the "Properties" window, go to the "Security" tab.

    - Click on the "Advanced" button.

    - In the "Advanced Security Settings" window, you'll see an "Inheritance" section. Click on the "Disable inheritance" button.

    - A dialog box will appear; choose the option "Remove all inherited permissions from this object" and click "Convert inherited permissions into explicit permissions on this object."

    - Once inheritance is disabled, you will see a list of users/groups with permissions. Remove permissions for all users except for the user account you are using (typically an administrator account).

    - Click "Apply" and then "OK" to save the changes.


### 2. Accessing EC2 Instance

1. Use the following SSH command to connect to your EC2 instance:
  ```
  ssh -i "your-key.pem" ubuntu@your-ec2-instance-public-ip
  ```

2. Gain superuser access by running: `sudo su`

3. Updating and Verifying Python
  - Update the EC2 instance with the latest packages:
    `apt update`

  - Verify Python installation:
    `python3 --version`

4. Installing Python Packages
`apt install python3-pip`

5. Transferring Files to EC2
    Use SCP to transfer your Streamlit application code to the EC2 instance:

    ```scp -i "your-key.pem" -r path/to/your/app ubuntu@your-ec2-instance-public-ip:/path/to/remote/location```

6. Setting Up Streamlit Application
    Change the working directory to the deployment files location:

    `cd /path/to/remote/location`

    Install dependencies from your requirements file:

    `pip3 install -r requirements.txt`

7. Running the Streamlit Application
    Test your Streamlit application (Use external link):
    `streamlit run app.py`


    For a permanent run, use nohup:
    `nohup streamlit run app.py`


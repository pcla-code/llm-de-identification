# De-identification Project

## Objective
This research project focuses on utilizing Large Language Models (LLMs) to remove personally identifiable data (PII) from forum posts. It involves two primary components and the use of specific folders and files:

## Project Structure
original_files: These files contain the original data with PII.

human_redacted_files: These are the files that humans have de-identified.

Prompts.csv: This CSV file includes a list of prompts to be used during the de-identification process and for evaluation.

## Code Files
1. de_identified_csv_generator.py
This component takes as input a ground truth file and an original file.
It generates an OpenAI de-identified CSV file as output for every prompt, containing redacted data.
2. de_identified_csv_evaluator.py
This component evaluates the accuracy of the de-identification process.
It takes files from the OpenAI_coded_files folder (created by the De-identified CSV Generator) as input.
The evaluator creates/updates the Metrics.csv file with metrics for each dataset and each prompt.

## Prerequisites
Before getting started, ensure you have the following prerequisites:

Python Environment: Make sure you have Python installed on your system.

OpenAI API Key: Obtain an API key from OpenAI to access their GPT-4 model for text generation. You can sign up for an API key on the OpenAI platform.

Input Data: Prepare the following input data:

original_files: These files contain the original data with PII.
human_redacted_files: These are the files that humans have de-identified.

# Getting Started
Here's how to initiate the project:

Step 1: Organize Data
Place your original files with PII in the original_files folder and human redacted files in human_redacted_files folder. We have added some sample files for your reference in the folder. Please note that we will not be able to share the original dataset as it consists of sensitive data.

Step 2: cd to the repository and add OpenAI API Key to de_identified_csv_generator.py

Step 3: Run the de_identified_csv_generator.py
Execute the de_identified_csv_generator script, providing the necessary input files and the output folder (which will be automatically created):
This script will process the audio transcripts, remove PII, and generate an OpenAI de-identified CSV file within an OpenAI_redacted_files folder.

Step 4: Run the de_identified_csv_evaluator.py
To evaluate the accuracy of the de-identification process, run the De-identified CSV Evaluator script:
This script will analyze the de-identified CSV files in the OpenAI_coded_files folder and update the Prompts.csv file with accuracy metrics for each dataset and each prompt. It will also create a new csv (Kappas.csv) to report the Kappa values. 



# De-identification Project

## Objective
This research project focuses on utilizing Large Language Models (LLMs) to remove personally identifiable data (PII) from forum posts. It involves two primary components and the use of specific folders and files:

## Project Structure
Ground Truth Files: This folder should contain your original files that include PII.

Original Files: This folder should contain your raw files that you want to de-identify.

Prompts.csv: This CSV file includes a list of prompts to be used during the de-identification process and for evaluation.

## Components
1. De-identified CSV Generator
This component takes as input a ground truth file and an original file.
It generates an OpenAI de-identified CSV file as output for every prompt, containing redacted data.
2. De-identified CSV Evaluator
This component evaluates the accuracy of the de-identification process.
It takes files from the OpenAI_coded_files folder (created by the De-identified CSV Generator) as input.
The evaluator updates the Prompts.csv file with accuracy metrics for each dataset and each prompt.

## Prerequisites
Before getting started, ensure you have the following prerequisites:

Python Environment: Make sure you have Python installed on your system.

OpenAI API Key: Obtain an API key from OpenAI to access their GPT-4 model for text generation. You can sign up for an API key on the OpenAI platform.

Input Data: Prepare the following input data:

Ground Truth Files: These files contain the original data with PII.
Original Files: These are the raw files that you want to de-identify.

# Getting Started
Here's how to initiate the project:

Step 1: Organize Data
Place your original files in the Original files folder.

Step 2: Configure OpenAI API Key

Step 3: Run the De-identified CSV Generator
Execute the De-identified CSV Generator script, providing the necessary input files and the output folder (which will be automatically created):
This script will process the audio transcripts, remove PII, and generate an OpenAI de-identified CSV file within an OpenAI_coded_files folder.

Step 4: Run the De-identified CSV Evaluator
To evaluate the accuracy of the de-identification process, run the De-identified CSV Evaluator script:
This script will analyze the de-identified CSV files in the OpenAI_coded_files folder and update the Prompts.csv file with accuracy metrics for each dataset and each prompt. It will also create a new csv (Kappas.csv) to report the Kappa values. 

Note: It takes 30-35 mins to generate the results using a single prompt for one csv. Testing one prompt against all csvs will take approximately 10-11 hours. 

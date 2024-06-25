from bs4 import BeautifulSoup
import openai
import pandas as pd
import os

# Specify the relative folder paths for original_files, human_redacted_files and OpenAI_redacted_files
gt_coded_folder = 'human_redacted_files/'
original_folder = 'original_files/'
output_folder = 'OpenAI_redacted_files/'

# Set the API key for openai
openai.api_key = 'Enter your key here'  # Make sure to use your own API key

# Reads prompts from a CSV file
try:
    prompt_df = pd.read_csv('Prompts.csv')
except Exception as e:
    print("Error reading Prompts.csv:", str(e))
    exit()

# Creates the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Constructs full paths using the current working directory
gt_coded_folder_path = os.path.join(os.getcwd(), gt_coded_folder)
original_folder_path = os.path.join(os.getcwd(), original_folder)

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def api_call(post_text, prompt):
    try:
        message_content = f'{prompt}\n{post_text}'
        response_check = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message_content}
            ],
            max_tokens=1000
        )
        return response_check["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"API call error for post_text: {post_text[:50]}", str(e))
        return "API_ERROR"

try:
    gt_files = os.listdir(gt_coded_folder_path)
except Exception as e:
    print(f"Error reading from directory {gt_coded_folder_path}:", str(e))
    exit()

for gt_file in gt_files:
    if gt_file.endswith('.csv'):
        try:
            original_file = gt_file.replace('_CODED.csv', '.csv')
            gt_coded = pd.read_csv(os.path.join(gt_coded_folder_path, gt_file),encoding_errors='ignore')
            original = pd.read_csv(os.path.join(original_folder_path, original_file),encoding_errors='ignore')
            common_ids = original[original['id'].isin(gt_coded['id'])]
            if not common_ids.empty:
                merged = common_ids.merge(gt_coded, on='id', how='inner', suffixes=('_original', '_coded'))
                new_df = merged[['post_text_original', 'post_text_human_coded']].copy()
                new_df['post_text_original'] = new_df['post_text_original'].apply(remove_html_tags)

                # Iterates for each prompt
                for index, row in prompt_df.iterrows():
                    prompt = row['prompt']
                    prompt_df_temp = new_df.copy()  # Creates a copy of the DataFrame for the current prompt
                    for row_idx in prompt_df_temp.index:
                        post_text_response = api_call(prompt_df_temp.loc[row_idx, 'post_text_original'], prompt)
                        prompt_df_temp.loc[row_idx, 'post_text_OPENAI_coded'] = post_text_response
                        
                        print(f"Updated row {row_idx} for prompt {index} in temporary dataframe.")

                    # Saves the CSV for the current prompt
                    csv_filename = f'{gt_file.replace(".csv", "")}_prompt{index}_openai_gpt4.csv'
                    output_file_path = os.path.join(output_folder, csv_filename)
                    prompt_df_temp.to_csv(output_file_path, index=False)
                    print(f"Saved file {output_file_path} for prompt {index}")

        except Exception as e:
            print(f"Error processing file {gt_file}:", str(e))


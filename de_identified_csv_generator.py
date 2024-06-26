import openai
import pandas as pd
import os

# Specify the relative folder paths for original files and output
original_folder = 'original_files/'
output_folder = 'results/OpenAI_redacted_files/'

# Set the API key for openai
openai.api_key = ' Enter you API key'  

# Reads prompts from a CSV file
try:
    prompt_df = pd.read_csv('prompts.csv', encoding='utf-8')
except Exception as e:
    print("Error reading prompts.csv:", str(e))
    exit()

# Creates the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Constructs full path using the current working directory
original_folder_path = os.path.join(os.getcwd(), original_folder)

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
    original_files = os.listdir(original_folder_path)
except Exception as e:
    print(f"Error reading from directory {original_folder_path}:", str(e))
    exit()

for original_file in original_files:
    if original_file.endswith('.csv'):
        try:
            original_df = pd.read_csv(os.path.join(original_folder_path, original_file), encoding='utf-8', encoding_errors='replace')
            # Iterates for each prompt
            for index, row in prompt_df.iterrows():
                prompt = row['prompt']
                for row_idx in original_df.index:
                    post_text_response = api_call(original_df.loc[row_idx, 'post_text_original'], prompt)
                    original_df.loc[row_idx, 'post_text_OpenAI_redacted'] = post_text_response
                    
                    print(f"Updated row {row_idx+1} for prompt {index+1} in dataframe.")

                # Saves the CSV for the current prompt
                csv_filename = f'{original_file.replace(".csv", "")}_prompt{index+1}_openai_gpt4.csv'
                output_file_path = os.path.join(output_folder, csv_filename)
                original_df.to_csv(output_file_path, index=False)
                print(f"Saved file {output_file_path} for prompt {index+1}")

        except Exception as e:
            print(f"Error processing file {original_file}:", str(e))

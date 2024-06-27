import pandas as pd
import os
import string

# Specify the relative folder paths for OpenAI_redacted_files
output_folder = 'results/OpenAI_redacted_files/'
human_redacted_folder = 'human_redacted_files/'

def count_redacted(text):
    return text.count("[REDACTED]")

def clean_text(text):
    # Checks if text is a string, converts to string if not
    if not isinstance(text, str):
        text = str(text)

    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = text.split()
    words = [word for word in words if word.isalnum() and word.lower() not in ['a', 'an', 'the']]
    return words

def add_word_lists_to_df(df):
    df['word_list_redacted'] = df['post_text_human_redacted'].apply(clean_text)
    df['word_list_openai_redacted'] = df['post_text_OpenAI_redacted'].apply(clean_text)
    return df

def calculate_metrics(df):
    total_true_positives = 0
    total_true_negatives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_words = 0

    tp_counts = []
    fp_counts = []
    fn_counts = []
    tn_counts = []

    for _, row in df.iterrows():
        row_tp = 0
        row_fp = 0
        row_fn = 0
        row_tn = 0
   
        human_words = row['word_list_redacted']
        openai_words = row['word_list_openai_redacted']
        
        for hw, ow in zip(human_words, openai_words):
            total_words += 1
            if hw == ow and hw == 'REDACTED':
                total_true_positives += 1
                row_tp += 1
            elif hw != ow and hw != 'REDACTED' and ow == 'REDACTED':
                total_false_positives += 1
                row_fp += 1
            elif hw != ow and hw == 'REDACTED' and ow != 'REDACTED':
                total_false_negatives += 1
                row_fn += 1
            else:
                total_true_negatives += 1
                row_tn += 1

        tp_counts.append(row_tp)
        fp_counts.append(row_fp)
        fn_counts.append(row_fn)
        tn_counts.append(row_tn)
        df.at[_, 'true_positives'] = row_tp
        df.at[_, 'true_negatives'] = row_tn
        df.at[_, 'false_positives'] = row_fp
        df.at[_, 'false_negatives'] = row_fn
        
    precision = total_true_positives / (total_true_positives + total_false_positives)
    recall = total_true_positives / (total_true_positives + total_false_negatives)
    accuracy = (total_true_positives + total_true_negatives) / (
                total_false_negatives + total_true_negatives + total_true_positives + total_false_positives)

    observed_agreement = (total_true_positives + total_true_negatives) / total_words
    expected_agreement = ((total_true_positives + total_false_positives) * (total_true_positives + total_false_negatives) + (total_false_positives + total_true_negatives) * (total_false_negatives + total_true_negatives)) / (total_words ** 2)

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return accuracy, precision, recall, kappa, tp_counts, tn_counts, fp_counts, fn_counts

prompt_df = pd.read_csv('prompts.csv', encoding_errors='ignore')
precision_df = pd.DataFrame(columns=prompt_df.columns)
recall_df = pd.DataFrame(columns=prompt_df.columns)
accuracy_df = pd.DataFrame(columns=prompt_df.columns)
kappa_df = pd.DataFrame(columns=prompt_df.columns)

all_csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]
human_redacted_files = [f for f in os.listdir(human_redacted_folder) if f.endswith('.csv')]
all_metrics = []

for file in all_csv_files:
    file_path = os.path.join(output_folder, file)
    df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
    original_file_name = file.split("_prompt")[0] + ".csv"
    df_human = pd.read_csv(os.path.join(human_redacted_folder, original_file_name), encoding='utf-8', encoding_errors='ignore')
    if 'post_text_human_redacted' not in df.columns:
        df = pd.merge(df, df_human, on='id')
    
    # Adds word lists as new columns
    df = add_word_lists_to_df(df)

    # Calculates metrics and get counts
    accuracy, precision, recall, kappa, tp_counts, tn_counts, fp_counts, fn_counts = calculate_metrics(df)

    # Adds counts to DataFrame
    df['true_positives'] = tp_counts
    df['true_negatives'] = tn_counts
    df['false_positives'] = fp_counts
    df['false_negatives'] = fn_counts

    # Extracts prompt number and original file name from filename
    prompt_number_part = file.split('_')[-3]
    prompt_number = int(prompt_number_part.replace('prompt', ''))
    original_file_name = file.split("_prompt")[0] + ".csv"

    print(f"Updated Metrics for {original_file_name}, Prompt {prompt_number}: Accuracy:{accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, Kappa: {kappa:.2f}")

    df.to_csv(file_path, index=False)

    # Stores metrics in a list
    all_metrics.append({
        'Prompt': prompt_number,
        'File': original_file_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Kappa': kappa
    })

# Creates a DataFrame with all metrics
all_metrics_df = pd.DataFrame(all_metrics)

# Saves all metrics to a CSV file
all_metrics_df.to_csv('results\metrics.csv', index=False)

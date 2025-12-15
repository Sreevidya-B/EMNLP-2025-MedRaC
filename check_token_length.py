# Before running, ensure the necessary libraries are installed:
# pip install pandas transformers torch sentencepiece

import pandas as pd
from transformers import AutoTokenizer
import json

def analyze_token_lengths(file_path='data/test_data.csv', output_path='check_token_length_results.json', summary_path='token_analysis_summary.txt'):
    """
    Analyzes the token lengths of 'Patient Note' and 'Question' columns in a CSV file
    and saves the results to a JSON file and summary statistics to a text file.

    Args:
        file_path (str): The path to the CSV file.
        output_path (str): The path to the output JSON file.
        summary_path (str): The path to the summary statistics text file.
    """
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        print("Tokenizer Qwen/Qwen3-8B loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None

    print(f"Loading and analyzing data from {file_path}...")
    try:
        df = pd.read_csv(file_path)

        required_columns = ["Patient Note", "Question", "Calculator ID", "Calculator Name"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing columns: {', '.join(missing_columns)}")
            return None, None

        df["Patient Note"] = df["Patient Note"].apply(lambda x: x if isinstance(x, str) else "")
        df["Question"] = df["Question"].apply(lambda x: x if isinstance(x, str) else "")
        df["Calculator ID"] = df["Calculator ID"].apply(lambda x: x if pd.notna(x) else "")
        df["Calculator Name"] = df["Calculator Name"].apply(lambda x: x if isinstance(x, str) else "")

        # Calculate token lengths for individual components
        df['note_token_length'] = df['Patient Note'].apply(lambda x: len(tokenizer.encode(x)))
        df['question_token_length'] = df['Question'].apply(lambda x: len(tokenizer.encode(x)))
        
        # Calculate User Message Tokens based on actual prompt construction
        # This matches the structure in _gen_formula_and_extracted_values()
        def calculate_user_msg_tokens(row):
            user_msg = (
                f"Here is the patient note:\n"
                f"{row['Patient Note']}\n\n"
                f"Here is the task:\n"
                f"{row['Question']}\n\n"
                "Please reason through each step carefully, providing justifications before stating the formula and extracted values. Return the response in the specified JSON format."
            )
            return len(tokenizer.encode(user_msg))
        
        df['user_message_tokens'] = df.apply(calculate_user_msg_tokens, axis=1)

        results = []
        for index, row in df.iterrows():
            results.append({
                "Row Number": int(row.get('Row Number', index + 1)),
                "Calculator ID": str(row['Calculator ID']),
                "Calculator Name": str(row['Calculator Name']),
                "Patient Note Tokens": int(row['note_token_length']),
                "Question Tokens": int(row['question_token_length']),
                "User Message Tokens": int(row['user_message_tokens'])
            })

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Token Length Analysis complete. Results saved to {output_path}")

        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("SUMMARY STATISTICS")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Total rows analyzed: {len(df)}")
        summary_lines.append("")
        summary_lines.append("Patient Note Tokens:")
        summary_lines.append(f"  Min:     {df['note_token_length'].min()}")
        summary_lines.append(f"  Max:     {df['note_token_length'].max()}")
        summary_lines.append(f"  Mean:    {df['note_token_length'].mean():.2f}")
        summary_lines.append("")
        summary_lines.append("Question Tokens:")
        summary_lines.append(f"  Min:     {df['question_token_length'].min()}")
        summary_lines.append(f"  Max:     {df['question_token_length'].max()}")
        summary_lines.append(f"  Mean:    {df['question_token_length'].mean():.2f}")
        summary_lines.append("")
        summary_lines.append("User Message Tokens (Patient Note + Question + Template):")
        summary_lines.append(f"  Min:     {df['user_message_tokens'].min()}")
        summary_lines.append(f"  Max:     {df['user_message_tokens'].max()}")
        summary_lines.append(f"  Mean:    {df['user_message_tokens'].mean():.2f}")
        summary_lines.append("=" * 60)
        
        # Find rows that would exceed context length (System + User > 8000)
        system_tokens_no_rag = 210  # From calculate_system_message_tokens()
        df['total_input_tokens'] = system_tokens_no_rag + df['user_message_tokens']
        long_prompts = df[df['total_input_tokens'] > 8000]
        
        if len(long_prompts) > 0:
            summary_lines.append("")
            summary_lines.append(f"Found {len(long_prompts)} rows that exceed 8000 token limit:")
            for _, row in long_prompts.iterrows():
                summary_lines.append(
                    f"  - Row {row.get('Row Number', 'N/A')}: "
                    f"System({system_tokens_no_rag}) + User({row['user_message_tokens']}) "
                    f"= {row['total_input_tokens']} tokens "
                    f"(Calculator: {row['Calculator Name']})"
                )
            if len(long_prompts) > 10:
                summary_lines.append(f"  ... and {len(long_prompts) - 10} more")
        
        return tokenizer, summary_lines

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def calculate_system_message_tokens(tokenizer):
    """
    Calculates and returns the token counts for the two system messages.
    
    Args:
        tokenizer: The loaded tokenizer instance.
        
    Returns:
        List of strings containing the system message token information.
    """
    # System message for use_rag=False (from _gen_formula_and_extracted_values in medRaC.py)
    system_msg_no_rag = (
        "You are a reasoning assistant that follows a chain-of-thought approach to find important information in a given patient note. "
        "Follow these steps to extract necessary information:\n"
        "1. Reason about which formula(s) are applicable. Then identify the correct formula required for the calculation and state it explicitly.\n"
        "2. First reason about what values are needed then explain where these values appear in the text. Then, explicitly extract the values and map them to the formula variables.\n"
        "{\"formula_reason\": str, \"formula\": str, \"extracted_values_reason\": str, \"extracted_values\": dict}\n\n"
        "- `formula_reason`: The reasons that a formula is applicable.\n"
        "- `formula`: The explicit mathematical equation used for the calculation (e.g., `BMI = weight / height^2`).\n"
        "- `extracted_values_reason`: Justification for how each value was identified.\n"
        "- `extracted_values`: A dictionary mapping variable names to extracted values from the note (e.g., {\"weight\": \"70kg\", \"height\": \"1.75m\"}).\n"
    )

    # System message for use_rag=True (from _gen_extracted_values in medRaC.py)
    system_msg_rag = (
        "You are a reasoning assistant that follows a chain-of-thought approach to find important information in a given patient note. "
        "Follow these steps to extract necessary information:\n"
        "First reason about what values are needed then explain where these values appear in the text. Then, explicitly extract the values and map them to the formula variables.\n"
        "{\"extracted_values_reason\": str, \"extracted_values\": dict}\n\n"
        "- `extracted_values_reason`: Justification for how each value was identified.\n"
        "- `extracted_values`: A dictionary mapping variable names to extracted values from the note (e.g., {\"weight\": \"70kg\", \"height\": \"1.75m\"}).\n"
    )

    tokens_no_rag = len(tokenizer.encode(system_msg_no_rag))
    tokens_rag = len(tokenizer.encode(system_msg_rag))

    system_msg_lines = []
    system_msg_lines.append("-" * 60)
    system_msg_lines.append("SYSTEM MESSAGE TOKEN COUNTS")
    system_msg_lines.append("-" * 60)
    system_msg_lines.append(f"System Message Tokens (use_rag=False): {tokens_no_rag}")
    system_msg_lines.append(f"System Message Tokens (use_rag=True):  {tokens_rag}")
    system_msg_lines.append("-" * 60)
    
    return system_msg_lines


if __name__ == "__main__":
    tokenizer, summary_lines = analyze_token_lengths()
    
    if tokenizer is not None and summary_lines is not None:
        # Calculate system message token counts
        system_msg_lines = calculate_system_message_tokens(tokenizer)
        
        output_path = 'token_analysis_summary.txt'
        with open(output_path, 'w') as f:
            f.write('\n'.join(system_msg_lines))
            f.write('\n\n')
            f.write('\n'.join(summary_lines))

        print(f"Summary of token length analysis saved to {output_path}")
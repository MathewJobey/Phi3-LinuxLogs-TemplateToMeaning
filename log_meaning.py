import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm  # Changed from tqdm.notebook for terminal support

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DATA_FOLDER = "../data"       # Adjust if your folders are different
OUTPUT_FOLDER = "../outputs"  # Adjust if your folders are different

# ==========================================
# 2. MODEL LOADING
# ==========================================
def load_model():
    print(f"Loading {MODEL_ID}...")
    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load Model (Optimized for hardware)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Hardware detected: {device.upper()}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            device_map=device, 
            torch_dtype="auto", 
            trust_remote_code=False 
        )
        
        # Create Pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        print("SUCCESS: Model is loaded and ready!")
        return pipe
        
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

# ==========================================
# 3. PROMPT GENERATION
# ==========================================
def build_prompt(template):
    system_instruction = """You are a Linux Log Translator.

Your task is to convert a Linux log TEMPLATE into ONE clear, professional, and easily understandable human-readable English sentence.

### VARIABLE DEFINITIONS
- <TIMESTAMP>: Date and time when the event occurred
- <HOSTNAME>: Name of the server where the event occurred
- <RHOST> / <IP>: Remote host IP address
- <USERNAME> / <USER>: User account involved
- <UID>: User ID number (0 usually indicates root/admin)
- <PID>: Process ID identifying the software program
- <PORT>: Network port number
- <STATE>: Action or status of the event  
  (e.g., session started, session closed, connection opened, connection closed, failed)

### STRICT RULES (DO NOT VIOLATE)
1. Rewrite the template only; do not infer or invent any information.
2. Every placeholder enclosed in < > MUST appear exactly as written in the output.
3. If the same placeholder appears multiple times in the template, include it only once in the sentence.
4. Do NOT rename, modify, or merge placeholders.
5. The sentence must be grammatically correct and flow naturally for all valid <STATE> values.
6. Produce exactly ONE complete sentence.
7. Output ONLY the final sentence — no explanations, headings, or formatting."""
    
    # Phi-3 Chat Format Tags
    return f"<|user|>\n{system_instruction}\n\nInput:\n{template}\n<|end|>\n<|assistant|>"

# ==========================================
# 4. MAIN EXECUTION LOGIC
# ==========================================
def main():
    # Load the AI model
    pipe = load_model()

    # Locate Data
    print(f"\nLooking for files in: {os.path.abspath(DATA_FOLDER)}")
    
    # Ensure data folder exists
    current_data_folder = DATA_FOLDER
    if not os.path.exists(current_data_folder):
        print(f"WARNING: Data folder '{DATA_FOLDER}' not found.")
        current_data_folder = "."  # Fallback to current directory
        print(f"Searching in current directory: {os.path.abspath(current_data_folder)}")

    available_files = [f for f in os.listdir(current_data_folder) if f.endswith('.xlsx')]
    target_filename = None

    # File Selection Logic
    if not available_files:
        print("WARNING: No Excel files found automatically.")
        user_input = input("Please paste the full path to your file: ").strip().replace('"', '')
        if os.path.exists(user_input):
            target_filename = user_input
    else:
        print("\nAvailable Files:")
        for i, f in enumerate(available_files):
            print(f" [{i+1}] {f}")
        
        selection = input("\nEnter the file number (or filename): ").strip()

        # Handle number selection
        if selection.isdigit() and 1 <= int(selection) <= len(available_files):
            target_filename = os.path.join(current_data_folder, available_files[int(selection)-1])
        # Handle filename input
        elif selection in available_files:
            target_filename = os.path.join(current_data_folder, selection)
        else:
            target_filename = selection.replace('"', '')

    # Processing Loop
    if not target_filename or not os.path.exists(target_filename):
        print(f"ERROR: File not found: {target_filename}")
        return

    print(f"\nReading {target_filename}...")
    try:
        df_summary = pd.read_excel(target_filename, sheet_name="Template Summary")
        df_logs = pd.read_excel(target_filename, sheet_name="Log Analysis")
        
        templates = df_summary['Template Pattern'].tolist()
        prompts = [build_prompt(t) for t in templates]
        
        print(f"Processing {len(prompts)} templates...")
        explanations = []

        # Run Inference
        for output in tqdm(pipe(prompts, max_new_tokens=120, return_full_text=False, do_sample=False), total=len(prompts)):
            
            raw_result = output[0]['generated_text'].strip()
            
            # Clean Output
            clean_result = raw_result.split('\n')[0] # First line only
            clean_result = clean_result.replace('Output:', '').replace('Result:', '').strip()
            clean_result = clean_result.replace('"', '').replace("'", "")
            
            if clean_result and not clean_result.endswith('.'):
                clean_result += "."
                
            explanations.append(clean_result)

        # Save Output
        df_summary['Event Meaning'] = explanations
        
        # Ensure output folder exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        base_name = os.path.basename(target_filename)
        save_path = os.path.join(OUTPUT_FOLDER, base_name.replace(".xlsx", "_Mapped.xlsx"))
        
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df_logs.to_excel(writer, sheet_name='Log Analysis', index=False)
            df_summary.to_excel(writer, sheet_name='Template Summary', index=False)
            
        print("-" * 60)
        print("SAMPLE RESULTS:")
        for i in range(min(3, len(explanations))):
            print(f"TEMPLATE: {templates[i][:50]}...")
            print(f"MEANING:  {explanations[i]}")
        
        print("-" * 60)
        print(f"DONE! Saved to: {save_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
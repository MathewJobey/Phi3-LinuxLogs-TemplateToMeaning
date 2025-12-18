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
    system_instruction = """You are a Linux Log Template Expander.
Your goal is to turn a technical log pattern into a natural English sentence structure.

### VARIABLE DICTIONARY (What the placeholders mean)
- <TIMESTAMP>: Date/Time of the event.
- <HOSTNAME>: The server or machine name.
- <RHOST> / <IP>: Remote IP address (the attacker or connector).
- <USER> / <USERNAME>: The user account involved.
- <UID> / <PID>: User ID / Process ID numbers.
- <STATE>: A event or system status  (e.g. 'opened', failed', 'accepted', 'closed').

### CORE TASK
Create a sentence that describes the event while preserving ALL variable placeholders as **fixed data slots**. 
Do not interpret the variables (e.g., do not change "<STATE>" to "initiated" or "closed"). Treat them as proper nouns that must appear in the final output.

### UNIVERSAL RULES
1. **Preservation:** Every placeholder inside < > (e.g., <TIMESTAMP>, <PID>, <STATE>) MUST appear in the output exactly as written.
2. **Grammar:** Structure the sentence so it makes sense regardless of what value fills the placeholder. 
   - *Bad:* "The session was <STATE>." (grammatically risky if state is 'failure')
   - *Good:* "The session entered a state of <STATE>." (always works)
3. **Completeness:** Never summarize. If a template has 5 variables, your sentence must contain 5 variables.

### EXAMPLES
Input: <TIMESTAMP> <HOSTNAME> su: session <STATE> for user <USERNAME>
Output: At <TIMESTAMP>, on the server <HOSTNAME>, a 'su' session for user <USERNAME> was recorded with a status of <STATE>.

Input: <TIMESTAMP> <HOSTNAME> sshd[<PID>]: Failed password for <USERNAME> from <RHOST>
Output: At <TIMESTAMP>, the server <HOSTNAME> recorded a failed password attempt for user <USERNAME> coming from remote host <RHOST>, handled by process ID <PID>.
"""
    
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
        
        # --- CHANGED LINE BELOW ---
        # Renames file to: [OriginalName]_meaning.xlsx
        save_path = os.path.join(OUTPUT_FOLDER, base_name.replace(".xlsx", "_meaning.xlsx"))
        
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
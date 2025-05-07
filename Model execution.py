import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from google import genai
from google.genai import types
from google.colab import files
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm

def safe_execution(func, error_message="An error occurred", *args, **kwargs):
    """Execute a function safely with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: {error_message}")
        print(f"Details: {str(e)}")
        traceback.print_exc()
        return None

# Improved file upload with validation
def upload_and_validate(message, file_type, required_columns=None):
    """Upload a file with validation and proper error handling"""
    print(f"\n🔼 {message}")
    
    # Try uploading up to 3 times
    for attempt in range(3):
        try:
            uploaded = files.upload()
            if not uploaded:
                if attempt < 2:
                    print("No file uploaded. Please try again.")
                    continue
                else:
                    raise RuntimeError("No file uploaded after multiple attempts.")
            
            file_path = next(iter(uploaded))
            
            # Validate file extension
            if file_type == "pkl" and not file_path.endswith('.pkl'):
                raise ValueError(f"Expected .pkl file, got {file_path}")
            elif file_type == "csv" and not file_path.endswith('.csv'):
                raise ValueError(f"Expected .csv file, got {file_path}")
            
            # For CSV, validate columns if specified
            if file_type == "csv" and required_columns:
                df = pd.read_csv(file_path)
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
                return file_path, df
            
            return file_path, None
        
        except Exception as e:
            if attempt < 2:
                print(f"Error: {str(e)}. Please try again.")
            else:
                raise RuntimeError(f"Failed to upload and validate file: {str(e)}")
    
    raise RuntimeError("Failed to upload file after multiple attempts")

# 1. Load and initialize Copilot pipeline with proper error handling
try:
    PIPELINE_PATH, _ = upload_and_validate("Please upload your customer-support Copilot pipeline (.pkl) to get started…", "pkl")
    
    # 2. Configure API client with error checking
    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        API_KEY = "AIzaSyAPrTyFXKJbs_wLLZcaC1JIkkZOCpT9Ieo"  # Fallback to default key
        os.environ["API_KEY"] = API_KEY
    
    client = genai.Client(api_key=API_KEY)
    MODEL_NAME = "gemini-2.5-flash-preview-04-17"
    
    # 3. Load the pipeline with error handling
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        print("\n✅ Pipeline loaded. Ready to analyze customer data in real time.\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline from {PIPELINE_PATH}: {str(e)}")
    
    # 4. Upload new customer data (CSV without 'Churn')
    # Determine required columns based on pipeline
    try:
        pre = pipeline.named_steps.get("pre")
        if pre:
            num_cols = pre.transformers_[0][2] if len(pre.transformers_) > 0 else []
            cat_cols = pre.transformers_[1][2] if len(pre.transformers_) > 1 else []
            required_columns = list(num_cols) + list(cat_cols)
        else:
            required_columns = None
    except Exception:
        required_columns = None  # If we can't determine required columns, proceed without validation
    
    data_path, df_new = upload_and_validate("Please upload the new customer data CSV to generate live call insights…", 
                                         "csv", required_columns)
    
    print(f"\n✅ Loaded {df_new.shape[0]} customers from {data_path}. Processing insights…\n")
    
    # 5. Performance optimization: Use chunks for large datasets
    CHUNK_SIZE = 1000  # Process 1000 customers at a time for better performance
    total_customers = len(df_new)
    
    # Initialize results arrays
    all_preds = np.zeros(total_customers, dtype=int)
    all_probs = np.zeros(total_customers)
    all_names = [None] * total_customers
    all_imps = [None] * total_customers
    
    # Process in chunks for better performance and memory usage
    for i in range(0, total_customers, CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE, total_customers)
        chunk_df = df_new.iloc[i:chunk_end].copy()
        
        # 6. Predict churn and probabilities with error handling
        try:
            chunk_preds = pipeline.predict(chunk_df)
            chunk_probs = pipeline.predict_proba(chunk_df)[:, 1]
            
            # Store results
            all_preds[i:chunk_end] = chunk_preds
            all_probs[i:chunk_end] = chunk_probs
            
        except Exception as e:
            print(f"Error predicting churn for chunk {i}:{chunk_end}: {str(e)}")
            # Continue with other chunks
    
    # 7. Identify top-3 churn drivers with optimized implementation
    def get_top_drivers(raw_df, pipeline, cat_cols, num_cols, top_n=3):
        try:
            pre = pipeline.named_steps["pre"]
            booster = pipeline.named_steps["model"].best_estimator_
            enc = pre.named_transformers_["cat"].named_steps["encoder"]
            encoded = enc.get_feature_names_out(cat_cols)
            features = list(num_cols) + list(encoded)
            
            # Transform data with better memory management
            Xt = pre.transform(raw_df)
            
            # Create DMatrix efficiently (avoid redundant conversions)
            dmat = xgb.DMatrix(Xt, feature_names=features)
            
            # Get contributions in parallel for better performance
            contribs = booster.get_booster().predict(dmat, pred_contribs=True)
            
            # Process results efficiently
            chunk_size = min(1000, len(raw_df))
            names, imps = [], []
            
            for row in contribs:
                vals = row[:-1]  # Exclude bias term
                # Use partial sorting for better performance
                idx = np.argpartition(np.abs(vals), -top_n)[-top_n:]
                # Sort the top_n indices
                idx = idx[np.argsort(np.abs(vals[idx]))[::-1]]
                names.append([features[i] for i in idx])
                imps.append([float(vals[i]) for i in idx])
                
            return names, imps
        except Exception as e:
            print(f"Error getting top drivers: {str(e)}")
            # Return empty lists as fallback
            return [[] for _ in range(len(raw_df))], [[] for _ in range(len(raw_df))]
    
    # Get top drivers for chunks in parallel
    def process_chunk(start_idx, end_idx):
        chunk_df = df_new.iloc[start_idx:end_idx]
        chunk_names, chunk_imps = get_top_drivers(chunk_df, pipeline, cat_cols, num_cols, top_n=3)
        return start_idx, end_idx, chunk_names, chunk_imps
    
    # Use ThreadPoolExecutor for parallel processing
    chunk_indices = [(i, min(i + CHUNK_SIZE, total_customers)) for i in range(0, total_customers, CHUNK_SIZE)]
    
    with ThreadPoolExecutor(max_workers=min(4, len(chunk_indices))) as executor:
        futures = [executor.submit(process_chunk, start, end) for start, end in chunk_indices]
        
        for future in tqdm(futures, desc="Processing chunks"):
            start, end, chunk_names, chunk_imps = future.result()
            for i, (names, imps) in enumerate(zip(chunk_names, chunk_imps)):
                idx = start + i
                all_names[idx] = names
                all_imps[idx] = imps
    
    # 8. Build output table with performance optimizations
    out = df_new.copy()
    out["PredictedChurn"] = all_preds
    out["ChurnProbability"] = all_probs
    
    if "MonthlyCharges" in out.columns:
        # Vectorized operation instead of loop
        out["LifetimeValueRisk"] = np.where(all_preds == 1, out["MonthlyCharges"] * 12, np.nan)
    
    # Add driver columns efficiently
    for i in range(3):
        out[f"Driver{i+1}"] = [names[i] if names and len(names) > i else None for names in all_names]
        out[f"Impact{i+1}"] = [imps[i] if imps and len(imps) > i else None for imps in all_imps]
    
    # 9. Save & download results with error handling
    output_csv = "live_support_insights.csv"
    try:
        out.to_csv(output_csv, index=False)
        print(f"\n💾 Insights saved to {output_csv}.")
        files.download(output_csv)
        print("Preparing live call summary…\n")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
    
    # 10. Calculate key metrics for prompt
    total_customers = len(df_new)
    at_risk_customers = sum(all_preds == 1)
    churn_rate = (at_risk_customers / total_customers) * 100 if total_customers > 0 else 0
    avg_revenue_at_risk = 0
    if "MonthlyCharges" in df_new.columns and at_risk_customers > 0:
        avg_revenue_at_risk = df_new.loc[all_preds == 1, "MonthlyCharges"].mean() * 12
    
    # Find common churn patterns based on driver frequency
    driver_counts = {}
    for i in range(1, 4):  # For Driver1, Driver2, Driver3
        col_name = f"Driver{i}"
        if col_name in out.columns:
            for driver in out.loc[out["PredictedChurn"] == 1, col_name]:
                if driver:
                    driver_str = str(driver)  # Convert to string in case it's not
                    if driver_str in driver_counts:
                        driver_counts[driver_str] += 1
                    else:
                        driver_counts[driver_str] = 1
    
    top_drivers = sorted(driver_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_drivers_str = ", ".join([f"{driver} ({count})" for driver, count in top_drivers])
    
    # Build prompt for Gemini
    try:
        with open(output_csv, "r") as f:
            csv_data = f.read()
    except Exception as e:
        print(f"Error reading CSV data: {str(e)}")
        csv_data = str(out.head(20))  # Fallback to using dataframe head if file can't be read
    
    prompt = f"""
    You are a customer support co-pilot assisting a call center agent during live Telco support interactions.
    Analyze the customer churn predictions and provide a clear, structured summary.
    
    Your tone should be concise, friendly, and empathetic. Output must be formatted in plain text, suitable for display in the Google Colab terminal.
    Be precise. Only include customers where PredictedChurn = 1.
    
    1. OVERVIEW
    
    * Total customers analyzed: {total_customers}
    * Customers at risk of churn (PredictedChurn = 1): {at_risk_customers}
    * Overall churn rate: {churn_rate:.1f}%
    * Average estimated annual revenue loss per at-risk customer: €{avg_revenue_at_risk:.2f}
    * Common churn risk patterns: {top_drivers_str}
    
    2. CUSTOMER-LEVEL INSIGHTS (only for customers with PredictedChurn = 1)
    For each customer, provide the following:
    
    ---
    
    Customer Index: \[Row Number or Customer ID]
    Churn Probability: \[XX.X%]
    Estimated Annual Value at Risk: €\[amount]
    
    Top 3 Churn Drivers:
    
    1. \[Driver 1] – \[Concise explanation of what this feature means in this context, e.g., "High Monthly Charges"]
    2. \[Driver 2] – \[Explanation]
    3. \[Driver 3] – \[Explanation]
    
    Recommended Call Actions:
    
    * Suggest 1–2 targeted actions the agent should take to reduce the risk. Use info like tenure, support history, service type.
    
    ---
    
    Guidelines:
    * Use natural language and telecom business intuition for insights.
    * Avoid redundancy—each insight should feel personalized.
    * Do not include markdown or bullet formatting like `**` or `#`.
    * Make sure to always output until the end so you cover all of the customers, dont cut off otherwise you wont help the agent.
    
    DATA BELOW:
    {csv_data} """
    
    print("🤖 Generating live support summary…\n")
    
    # Improved generate_with_retry with better error handling and performance
    def generate_with_retry(prompt, max_attempts=3, max_retries_per_chunk=3):
        full_response = ""
        safety_counter = 0
        max_safety = 10  # Prevent infinite loops
        
        # Split the original prompt into chunks if needed
        current_prompt = prompt
        
        while safety_counter < max_safety:
            safety_counter += 1
            
            for attempt in range(max_attempts):
                try:
                    start_time = time.time()
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=current_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=4096,  # Maximum allowed by the model
                            top_p=0.95,
                            top_k=40
                        )
                    )
                    elapsed = time.time() - start_time
                    
                    chunk_text = response.text.strip()
                    full_response += chunk_text
                    
                    # Better completion detection
                    is_likely_complete = False
                    
                    # Check for typical completion markers
                    if (
                        "Customer Index" in chunk_text and 
                        not chunk_text.endswith("...") and 
                        not chunk_text.endswith("…") and
                        not chunk_text.endswith("--") and
                        not chunk_text.endswith("Recomm")
                    ):
                        # Check if we've covered all at-risk customers (rough heuristic)
                        customer_count = chunk_text.count("Customer Index")
                        if customer_count >= at_risk_customers:
                            is_likely_complete = True
                        elif at_risk_customers > 10 and customer_count >= at_risk_customers * 0.9:
                            # If we've covered at least 90% of customers for large datasets
                            is_likely_complete = True
                    
                    if is_likely_complete:
                        print(f"✅ Response generation complete in {safety_counter} iterations")
                        return full_response
                    
                    # If incomplete, create a smarter continuation prompt
                    last_chunk = chunk_text[-800:] if len(chunk_text) > 800 else chunk_text
                    # Find the last complete customer entry
                    last_customer_idx = last_chunk.rfind("Customer Index")
                    
                    if last_customer_idx >= 0:
                        # Extract context from the last complete customer entry
                        context = last_chunk[last_customer_idx:]
                    else:
                        context = last_chunk
                    
                    current_prompt = f"""
                    You were previously generating a customer support summary but were cut off.
                    Please continue from where you left off. The last part you generated was:
                    
                    {context}
                    
                    Continue the summary covering all remaining customers with PredictedChurn = 1.
                    Remember to follow the same format for each customer:
                    - Customer Index
                    - Churn Probability
                    - Estimated Annual Value at Risk
                    - Top 3 Churn Drivers with explanations
                    - Recommended Call Actions
                    
                    Make sure to complete the summary for all remaining at-risk customers.
                    """
                    print(f"📝 Continuing generation (iteration {safety_counter}, {len(full_response)} chars so far)")
                    break  # Break out of the retry loop if successful
                    
                except Exception as e:
                    print(f"⚠️ Attempt {attempt+1} failed: {str(e)}")
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                    if attempt == max_attempts - 1:
                        return full_response + f"\n[Error: Unable to complete summary generation after {max_attempts} attempts.]"
            
            # If we've made several continuations and still haven't finished, break to avoid infinite loops
            if safety_counter >= max_safety:
                full_response += f"\n[Warning: Response may be incomplete after {max_safety} continuation attempts.]"
                break
        
        return full_response
    
    summary = generate_with_retry(prompt)
    print("\n" + "="*50)
    print("🛎️ LIVE SUPPORT INSIGHTS")
    print("="*50 + "\n")
    print(summary)

except Exception as e:
    print("\n❌ ERROR: The process encountered an error.")
    print(f"Details: {str(e)}")
    traceback.print_exc()

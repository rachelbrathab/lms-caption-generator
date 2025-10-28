import os
import torch
import nltk
import pandas as pd
import requests
import zipfile
import glob
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from tqdm import tqdm

def download_file(url, filename):
    """
    Downloads a file from a URL with a progress bar.
    """
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Downloaded {filename}.")

def unzip_file(filename, extract_dir):
    """
    Unzips a file to a specified directory.
    """
    print(f"Unzipping {filename} to {extract_dir}...")
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Unzipped {filename}.")

def setup_data():
    """
    Downloads and unzips all required data.
    """
    # Define file names
    images_zip = "Flickr8k_Dataset.zip"
    text_zip = "Flickr8k_text.zip"
    images_dir = "flickr8k_images"
    text_dir = "flickr8k_text"

    # Download Images
    if not os.path.exists(images_zip):
        download_file("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", images_zip)
    else:
        print(f"{images_zip} already exists. Skipping download.")
        
    # Download Text
    if not os.path.exists(text_zip):
        download_file("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip", text_zip)
    else:
        print(f"{text_zip} already exists. Skipping download.")
        
    # Unzip Images
    if not os.path.exists(images_dir):
        unzip_file(images_zip, images_dir)
    else:
        print(f"{images_dir} already exists. Skipping unzip.")
        
    # Unzip Text
    if not os.path.exists(text_dir):
        unzip_file(text_zip, text_dir)
    else:
        print(f"{text_dir} already exists. Skipping unzip.")

def load_and_parse_data():
    """
    Reads the text file and creates DataFrames for training and validation.
    """
    print("Parsing caption files...")
    captions_file = "flickr8k_text/Flickr8k.token.txt"
    images_folder = "flickr8k_images/Flicker8k_Dataset"
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) != 2:
            continue
            
        image_name_with_index = parts[0]
        caption = parts[1]
        
        image_name = image_name_with_index.split('#')[0]
        image_path = os.path.join(images_folder, image_name)
        
        if os.path.exists(image_path): # Check if image actually exists
            data.append({ "image_path": image_path, "caption": caption })
        
    df = pd.DataFrame(data)
    
    def load_image_names(filename):
        with open(f"flickr8k_text/{filename}", 'r') as f:
            return set(f.read().splitlines())

    train_images = load_image_names("Flickr_8k.trainImages.txt")
    val_images = load_image_names("Flickr_8k.devImages.txt")
    
    df['base_name'] = df['image_path'].apply(os.path.basename)
    train_df = df[df['base_name'].isin(train_images)].drop(columns=['base_name'])
    val_df = df[df['base_name'].isin(val_images)].drop(columns=['base_name'])
    
    # --- MEMORY FIX ---
    # We are using a small sample to prevent RAM errors and finish quickly.
    print("--- REDUCING DATASET SIZE FOR LOCAL CPU TRAINING ---")
    train_df = train_df.sample(n=100, random_state=42)
    val_df = val_df.sample(n=50, random_state=42)
    print("--- Using 100 training samples and 50 validation samples ---")
    
    return train_df, val_df

def create_datasets(train_df, val_df):
    """
    Converts Pandas DataFrames to Hugging Face Dataset objects.
    """
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    print(f"Training samples loaded: {len(train_dataset)}")
    print(f"Validation samples loaded: {len(val_dataset)}")
    return train_dataset, val_dataset

def preprocess_function(examples, feature_extractor, tokenizer, max_length=128):
    """
    Pre-processes a batch of image paths and captions.
    """
    try:
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    except Exception as e:
        print(f"Error opening image: {e}")
        return {} # Return empty dict to skip batch on error
        
    captions = examples['caption']
    
    model_inputs = feature_extractor(images=images, return_tensors="pt")
    
    labels = tokenizer(
        captions,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    model_inputs['labels'] = labels
    return model_inputs

def main():
    # --- 1. SETUP DATA ---
    setup_data()
    
    # --- 2. LOAD & PARSE DATA ---
    train_df, val_df = load_and_parse_data()
    train_dataset, val_dataset = create_datasets(train_df, val_df)
    
    # --- 3. LOAD MODEL & TOKENIZER ---
    print("Loading pre-trained model...")
    nltk.download('punkt')
    
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Pre-trained model loaded.")

    # --- 4. PREPROCESS DATASETS ---
    print("Processing datasets... (This may take a while)")
    
    processed_train_dataset = train_dataset.map(
        function=preprocess_function,
        batched=True,
        batch_size=4, # Small batch size for low RAM
        remove_columns=train_dataset.column_names,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
        writer_batch_size=100 # Write to disk often
    )

    processed_val_dataset = val_dataset.map(
        function=preprocess_function,
        batched=True,
        batch_size=4, # Small batch size for low RAM
        remove_columns=val_dataset.column_names,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
        writer_batch_size=100 # Write to disk often
    )
    
    print("Datasets processed.")

    # --- 5. SET UP TRAINER ---
    output_dir = "my-fine-tuned-model"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        eval_strategy="steps", # Correct argument name
        eval_steps=10, # Evaluate more often on small dataset
        logging_steps=10, # Log more often on small dataset
        num_train_epochs=1,
        save_strategy="epoch",
        report_to="none",
        push_to_hub=False,
        fp16=False,
        use_cpu=True # Force CPU training
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_val_dataset,
        tokenizer=feature_extractor,
        data_collator=default_data_collator,
    )

    # --- 6. START TRAINING ---
    print("="*50)
    print("STARTING MODEL TRAINING ON CPU (SAMPLE DATASET).")
    print("This should be quick (5-10 minutes).")
    print("="*50)
    
    trainer.train()

    # --- 7. SAVE FINAL MODEL ---
    print("Training complete. Saving model...")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    print("You can now point your 'backend/model.py' to this folder.")

if __name__ == "__main__":
    main()
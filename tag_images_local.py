import os
import csv
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# --- ARTIST CREDIT ---
ARTIST_NAME = "Your Friend's Name"

# --- ADVANCED AI VOCABULARY ---
# We provide a broad "palette" of concepts for the AI to detect
VOCABULARY = {
    "Style": ["children's book illustration", "lush watercolor", "vibrant oil painting", "detailed pencil sketch", "folk art style"],
    "Flora": ["giant kapok tree", "emerald canopy", "hanging vines", "exotic bromeliads", "thick moss", "tropical flora"],
    "Fauna": ["jaguar", "howler monkey", "tree frog", "toucan", "macaw", "sloth", "vibrant butterflies"],
    "Mood": ["mystical", "serene", "humid", "teeming with life", "urgent environmental theme", "sun-dappled", "shadowy"]
}

# Flatten the dictionary into a single list for CLIP
ALL_LABELS = [label for sublist in VOCABULARY.values() for label in sublist]

def main():
    print("Loading AI model weights...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    image_dir = "kapok_tree_images"
    output_file = "kapok_tree_analysis.csv"
    
    # Pre-tokenize all labels using a prompt template (PROMPT ENGINEERING)
    # This helps CLIP recognize concepts better than just single words
    prompts = [f"a high-quality illustration of {label}" for label in ALL_LABELS]
    text_tokens = tokenizer(prompts)

    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "AI Tags", "Artist Credit"])

        for file in tqdm(images, desc="Analyzing Art"):
            path = os.path.join(image_dir, file)
            try:
                img = preprocess(Image.open(path)).unsqueeze(0)

                with torch.no_grad():
                    # Calculate image and text embeddings
                    image_features = model.encode_image(img)
                    text_features = model.encode_text(text_tokens)
                    
                    # Normalize features for accurate similarity comparison
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    # Calculate similarity scores
                    # Multiplied by 100 as per standard CLIP implementation
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(5) # Take top 5 most confident tags

                # Filter results (Confidence Threshold)
                detected = [ALL_LABELS[idx] for i, idx in enumerate(indices) if values[i] > 0.05]
                
                writer.writerow([file, "; ".join(detected), ARTIST_NAME])
            except Exception as e:
                print(f"Error on {file}: {e}")

    print(f"\nSuccess! Deep analysis saved to {output_file}")

if __name__ == "__main__":
    main()
import torch
import open_clip
from PIL import Image
import os
import csv

# 1. SETUP: Load a lightweight version of the CLIP model
print("Loading CLIP (lightweight model)...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 2. DEFINITION: The tags we want the AI to "look" for
CANDIDATE_TAGS = ["jungle", "man with ax", "butterflies", "rainforest", "painting", "animals", "sunny", "dark"]

def tag_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = tokenizer(CANDIDATE_TAGS)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # Calculate similarity (this is the "AI brain" part)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    # Get tags that score above 15%
    results = zip(CANDIDATE_TAGS, text_probs[0].tolist())
    detected = [tag for tag, prob in results if prob > 0.15]
    return "; ".join(detected)

def main():
    image_dir = "kapok_tree_images"
    with open("clip_tags.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ai_tags"])
        
        for file in os.listdir(image_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Analyzing {file}...")
                tags = tag_image(os.path.join(image_dir, file))
                writer.writerow([file, tags])

if __name__ == "__main__":
    main()
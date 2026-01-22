import os
import csv
import google.generativeai as genai
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGE_DIR = "kapok_tree_images"
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def get_orientation(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return "landscape" if width > height else "portrait"

def analyze_image(image_path):
    img = Image.open(image_path)
    prompt = "Describe this artwork in 3-4 sentences and provide a comma-separated list of tags."
    
    # We ask for a specific format to make parsing easier
    full_prompt = f"{prompt}\nFormat:\nDescription: [text]\nTags: [tag1, tag2]"
    
    response = model.generate_content([full_prompt, img])
    text = response.text
    
    # Simple extraction logic
    description = ""
    tags = ""
    for line in text.split('\n'):
        if "Description:" in line: description = line.replace("Description:", "").strip()
        if "Tags:" in line: tags = line.replace("Tags:", "").strip()
    
    return {"description": description, "tags": tags}

def main():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with open("artwork_descriptions_gemini.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'description', 'tags', 'orientation'])
        writer.writeheader()
        
        for file in tqdm(image_files, desc="Processing"):
            path = os.path.join(IMAGE_DIR, file)
            analysis = analyze_image(path)
            writer.writerow({
                'filename': file,
                'description': analysis['description'],
                'tags': analysis['tags'],
                'orientation': get_orientation(path)
            })

if __name__ == "__main__":
    main()
import os
import time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from io import BytesIO
from PIL import Image, ImageFile
from transformers import AutoModelForImageClassification, ViTImageProcessor
import psutil

# Configuration
MODEL_NAME = "Falconsai/nsfw_image_detection"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
CORS(app, resources={
    r"/classify": {"origins": ["chrome-extension://*", "http://localhost:*"]},
    r"/debug": {"origins": "*"}
})

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Debug endpoint
@app.route('/debug', methods=['GET'])
def debug():
    return jsonify({
        "status": "active",
        "device": DEVICE,
        "model": MODEL_NAME,
        "system": {
            "cpu_cores": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / (1024 ** 3),
            "python_version": os.sys.version
        }
    })

# Model loading
print(f"{Colors.OKBLUE}⚡ Loading model...{Colors.ENDC}")
start_time = time.time()
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
).to(DEVICE)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
print(f"{Colors.OKGREEN}✅ Model loaded in {time.time()-start_time:.2f}s on {DEVICE.upper()}{Colors.ENDC}")

def is_valid_url(url):
    return url.startswith(('http://', 'https://')) and not any(
        domain in url for domain in ['localhost', '127.0.0.1', '192.168.']
    )

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if not request.is_json:
            print(f"{Colors.FAIL}❌ Invalid content type{Colors.ENDC}")
            return jsonify({"error": "Invalid content type"}), 400
            
        data = request.get_json()
        urls = data.get('urls', [])
        
        print(f"\n{Colors.HEADER}📨 Received {len(urls)} image URLs:{Colors.ENDC}")
        for i, url in enumerate(urls[:5], 1):
            print(f"  {i}. {url[:80]}{'...' if len(url) > 80 else ''}")

        valid_urls = [url for url in urls if is_valid_url(url)][:20]
        print(f"{Colors.OKBLUE}🔍 Valid images to process: {len(valid_urls)}{Colors.ENDC}")

        results = []
        for url in valid_urls:
            try:
                print(f"\n{Colors.OKBLUE}🔄 Processing: {url[:60]}...{Colors.ENDC}")
                start_time = time.time()
                
                response = requests.get(url, stream=True, timeout=5)
                dl_time = time.time() - start_time
                print(f"  {Colors.OKBLUE}⬇️ Downloaded in {dl_time:.2f}s ({len(response.content)/1024:.1f} KB){Colors.ENDC}")
                
                img = Image.open(BytesIO(response.content)).convert('RGB')
                inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred = probs.argmax().item()
                
                result = {
                    "url": url,
                    "label": model.config.id2label[pred],
                    "confidence": float(probs[0][pred])
                }
                color = Colors.FAIL if "nsfw" in result['label'].lower() else Colors.OKGREEN
                print(f"  {color}🔍 Result: {result['label']} ({result['confidence']:.0%}){Colors.ENDC}")
                results.append(result)
                
            except Exception as e:
                print(f"  {Colors.FAIL}❌ Error: {str(e)}{Colors.ENDC}")
                results.append({"url": url, "error": str(e)})
        
        success = len([r for r in results if 'error' not in r])
        failed = len([r for r in results if 'error' in r])
        print(f"\n{Colors.OKGREEN if success > 0 else Colors.FAIL}✅ Completed: {success} ok, {failed} failed{Colors.ENDC}")
        return jsonify({"results": results})
        
    except Exception as e:
        print(f"{Colors.FAIL}🔥 Server error: {str(e)}{Colors.ENDC}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"{Colors.HEADER}🚀 Starting server at http://localhost:5000{Colors.ENDC}")
    app.run(port=5000, debug=True, threaded=True)
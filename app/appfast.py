from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import tensorflow as tf
import numpy as np
import io
import json
import torchvision.models as models
import torch.nn as nn
import time

app = FastAPI(
    title="Brain MRI Classification API",
    description="API for classifying brain MRI scans using VGG16 and ResNet18 models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load VGG16 model
try:
    print("Loading VGG16 model...")
    vgg_model = tf.keras.models.load_model('../models/vgg16_brain_mri_model.h5')
    print("VGG16 model loaded successfully!")
except Exception as e:
    print(f"Error loading VGG16 model: {str(e)}")
    raise

# Create ResNet18 model architecture first
def get_resnet_model(num_classes=4):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    return model

# Set up class mappings
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Set manually if files missing
try:
    with open('vgg16_class_indices.json', 'r') as f:
        vgg_class_indices = json.load(f)
        vgg_classes = {int(v): k for k, v in vgg_class_indices.items()}
except:
    print("Using default class mapping for VGG16")
    vgg_classes = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

try:
    with open('resnet18_class_names.json', 'r') as f:
        class_names = json.load(f)
except:
    print("Using default class names for ResNet18")

# Safe loading of PyTorch model
print("Loading PyTorch model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
resnet_model = get_resnet_model(num_classes=len(class_names))

try:
    # Method 1: First try using torch.serialization's safe_globals context manager
    from torch.serialization import safe_globals
    from torchvision.models.resnet import ResNet
    
    with safe_globals([ResNet]):
        loaded_model = torch.load('../models/resnet18_weights.pth', map_location=device)
        
        # Check if it's a state_dict or full model
        if isinstance(loaded_model, dict):
            resnet_model.load_state_dict(loaded_model)
        else:
            resnet_model = loaded_model
    
    print("PyTorch model loaded successfully with safe_globals!")
except Exception as e:
    print(f"First loading attempt failed: {str(e)}")
    try:
        # Method 2: Try with explicit weights_only=False and add_safe_globals
        from torch.serialization import add_safe_globals
        from torchvision.models.resnet import ResNet
        
        add_safe_globals([ResNet])
        loaded_model = torch.load('../models/resnet18_weights.pth', weights_only=False, map_location=device)
        
        # Check if it's a state_dict or full model
        if isinstance(loaded_model, dict):
            resnet_model.load_state_dict(loaded_model)
        else:
            resnet_model = loaded_model
        
        print("PyTorch model loaded successfully with weights_only=False!")
    except Exception as e:
        # Method 3: Try with resnet18_brain_mri_full.pth if available
        try:
            with safe_globals([ResNet]):
                loaded_model = torch.load('../models/resnet18_weights.pth', map_location=device)
                
                if isinstance(loaded_model, dict):
                    resnet_model.load_state_dict(loaded_model)
                else:
                    resnet_model = loaded_model
                
                print("PyTorch model loaded from resnet18_weights.pth!")
        except Exception as inner_e:
            print(f"Error loading PyTorch model: {str(e)}")
            print(f"Additional attempt also failed: {str(inner_e)}")
            print("Proceeding with uninitialized ResNet model")

# Ensure the model is in eval mode and on the correct device
resnet_model.eval()
resnet_model = resnet_model.to(device)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain MRI Classification API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #4285f4;
            }
            h2 {
                color: #34a853;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
            .endpoint {
                margin-bottom: 20px;
                border-left: 4px solid #4285f4;
                padding-left: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Brain MRI Classification API</h1>
        <p>This API provides endpoints for classifying brain MRI scans using two different models:</p>
        
        <h2>Available Models:</h2>
        <ul>
            <li><strong>VGG16</strong>: Deep learning model for brain tumor classification</li>
            <li><strong>ResNet18</strong>: Alternative architecture for brain tumor classification</li>
        </ul>
        
        <h2>API Endpoints:</h2>
        
        <div class="endpoint">
            <h3>POST /predict/vgg</h3>
            <p>Classify a brain MRI image using the VGG16 model</p>
            <pre>curl -X POST "http://localhost:8000/predict/vgg" -F "file=@your_image.jpg"</pre>
        </div>
        
        <div class="endpoint">
            <h3>POST /predict/resnet</h3>
            <p>Classify a brain MRI image using the ResNet18 model</p>
            <pre>curl -X POST "http://localhost:8000/predict/resnet" -F "file=@your_image.jpg"</pre>
        </div>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p>Check API health status</p>
            <pre>curl "http://localhost:8000/health"</pre>
        </div>
        
        <p>For more details, visit the <a href="/docs">API documentation</a> (Swagger UI)</p>
    </body>
    </html>
    """
    return html_content

@app.post("/predict/vgg")
async def predict_vgg(file: UploadFile = File(...)):
    """
    Classify a brain MRI image using VGG16 model
    
    - **file**: The image file to classify
    
    Returns classification result with class name and confidence
    """
    start_time = time.time()
    
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = vgg_model.predict(image_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        inference_time = time.time() - start_time
        
        return {
            "model": "VGG16",
            "class": vgg_classes[predicted_class],
            "confidence": float(confidence),
            "inference_time": round(inference_time, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/resnet")
async def predict_resnet(file: UploadFile = File(...)):
    """
    Classify a brain MRI image using ResNet18 model
    
    - **file**: The image file to classify
    
    Returns classification result with class name and confidence
    """
    start_time = time.time()
    
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(image).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = resnet_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = float(probabilities[predicted_class])
        
        inference_time = time.time() - start_time
        
        return {
            "model": "ResNet18",
            "class": class_names[predicted_class],
            "confidence": float(confidence),
            "inference_time": round(inference_time, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "models": {
            "vgg16": "loaded" if vgg_model is not None else "not loaded",
            "resnet18": "loaded" if resnet_model is not None else "not loaded"
        },
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("appfast:app", host="0.0.0.0", port=8000, reload=True)
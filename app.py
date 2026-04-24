import torch
import gradio as gr
import os
from src.model import DRModel
from torchvision import transforms as T

CHECKPOINT_PATH = "artifacts/checkpoints/run-2025-09-10-19-21-12/epoch=13-step=42-val_loss=0.99-val_acc=0.75-val_kappa=nan.ckpt"

# Check if model checkpoint exists
if os.path.exists(CHECKPOINT_PATH) and os.path.getsize(CHECKPOINT_PATH) > 100:
    try:
        model = DRModel.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu")
        model.eval()
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
else:
    print("Model checkpoint not found or is just a placeholder.")
    print("Please train the model first using train.py or download a pre-trained model.")
    model_loaded = False

labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    # The model was trained with only 3 classes (0, 1, 2)
    # 3: "Severe",
    # 4: "Proliferative DR",
}

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Debug function to check image format
def debug_image_info(img):
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size if hasattr(img, 'size') else 'No size attribute'}")
    return img


# Define the prediction function
def predict(input_img):
    if not model_loaded:
        return {"Error": 1.0, "Model not loaded": 0.0}
    
    # Convert PIL image to tensor and apply transformations
    try:
        # Debug the input image
        debug_image_info(input_img)
        
        # Apply transformations
        input_tensor = transform(input_img).unsqueeze(0)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            print(f"Model output shape: {outputs.shape}")
            prediction = torch.nn.functional.softmax(outputs[0], dim=0)
            print(f"Prediction tensor: {prediction}")
            confidences = {labels[i]: float(prediction[i]) for i in labels}
        
        print(f"Prediction made: {confidences}")
        return confidences
    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return {"Error": 1.0, "Prediction failed": 0.0}


# Set up the Gradio app interface
dr_app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Diabetic Retinopathy Detection App",
    description="Welcome to our Diabetic Retinopathy Detection App! \
        This app utilizes deep learning models to detect diabetic retinopathy in retinal images.\
        Diabetic retinopathy is a common complication of diabetes and early detection is crucial for effective treatment.",
    examples=[
        "data/sample/10_left.jpeg",
        "data/sample/10_right.jpeg",
        "data/sample/15_left.jpeg",
        "data/sample/16_right.jpeg",
    ],
)

# Run the Gradio app
if __name__ == "__main__":
    print("Starting Diabetic Retinopathy Detection App...")
    print("Loading model from:", CHECKPOINT_PATH)
    dr_app.launch(share=False)
    print("Application launched successfully!")

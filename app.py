import streamlit as st
import torch
import timm
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="DeepFake Detector", page_icon="🧠", layout="centered")

st.title("🧠 DeepFake Detection System")
st.write("Upload a face image and detect whether it's REAL or FAKE.")

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cpu")

# ---------------------------
# MODEL
# ---------------------------
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 1)

model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------------------
# TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        return "FAKE", prob
    else:
        return "REAL", prob

# ---------------------------
# UPLOAD IMAGE
# ---------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)

    label, prob = predict(image)

    st.markdown("---")
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {prob:.4f}")

    if label == "REAL":
        st.success("This appears to be a REAL face ✅")
    else:
        st.error("This appears to be FAKE ⚠️")
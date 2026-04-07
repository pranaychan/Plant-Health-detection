import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_class_name(name):
    return name.replace("___", " - ").replace("_", " ")

@st.cache_resource
def load_model():
    checkpoint = torch.load("best_model.pth", map_location=device)

    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint["classes"]

model, classes = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def generate_gradcam(model, image_tensor, class_idx):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    return cam

st.title("Plant Health Detection + Grad-CAM")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    pred_class = predicted.item()

    st.success(f"Prediction: {format_class_name(classes[pred_class])}")
    st.info(f"Confidence: {confidence.item()*100:.2f}%")

    cam = generate_gradcam(model, img, pred_class)

    original = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    st.subheader("Grad-CAM (Model Attention)")
    st.image(overlay)

    top3_prob, top3_idx = torch.topk(probs, 3)

    st.write("Top 3 Predictions:")
    for i in range(3):
        st.write(
            f"{format_class_name(classes[top3_idx[0][i]])}: {top3_prob[0][i].item()*100:.2f}%"
        )
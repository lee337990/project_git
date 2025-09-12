import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("saved_models/best_model_epoch5_acc0.9385.pth", map_location=device))  # 파일명 수정 필요
    model.eval()
    return model.to(device)

model = load_model()

# 라벨 이름
LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 전처리 함수
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 예측 함수
def predict(image):
    image = ImageOps.grayscale(image)  # 흑백 변환
    img_tensor = transform(image).unsqueeze(0).to(device)
    output = model(img_tensor)

    st.write("모델 raw 출력:", output)

    prob = torch.softmax(output, dim=1)
    top_prob, top_class = torch.max(prob, dim=1)
    return LABELS[top_class.item()], top_prob.item()

# Streamlit 앱 UI
st.title("👟 Fashion MNIST 예측 앱")
st.write("학습된 ResNet18 모델을 활용해 이미지가 어떤 의류인지 예측해요!")

uploaded_file = st.file_uploader("이미지를 업로드하세요 (28x28 PNG 권장)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    if st.button("👀 예측하기"):
        label, confidence = predict(image)
        st.success(f"✅ 예측 결과: **{label}** ({confidence*100:.2f}%)")

        # GradCAM 시각화
        class GradCAM:
            def __init__(self, model, target_layer):
                self.model = model.eval()
                self.target_layer = target_layer
                self.gradients = None
                self.activations = None
                self.hook()

            def hook(self):
                def forward_hook(module, input, output):
                    self.activations = output.detach()
                def backward_hook(module, grad_input, grad_output):
                    self.gradients = grad_output[0].detach()
                self.target_layer.register_forward_hook(forward_hook)
                self.target_layer.register_backward_hook(backward_hook)

            def generate(self, input_tensor, class_idx=None):
                output = self.model(input_tensor)
                if class_idx is None:
                    class_idx = torch.argmax(output)
                self.model.zero_grad()
                output[0, class_idx].backward()

                weights = self.gradients.mean(dim=[2, 3], keepdim=True)
                cam = (weights * self.activations).sum(dim=1, keepdim=True)
                cam = torch.nn.functional.relu(cam)
                cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
                cam = cam.squeeze().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min())
                return cam

        gradcam = GradCAM(model, model.layer4[1].conv2)
        input_tensor = transform(image).unsqueeze(0).to(device)
        cam = gradcam.generate(input_tensor)

        # 시각화 출력
        fig, ax = plt.subplots()
        ax.imshow(image.convert("L").resize((224, 224)), cmap='gray')
        ax.imshow(cam, cmap='jet', alpha=0.5)
        ax.axis("off")
        st.pyplot(fig)

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import string

# Define a class mapping for digits 0-9 and uppercase letters A-Z
class_mapping = {i: str(i) for i in range(10)}
class_mapping.update({i + 10: letter for i, letter in enumerate(string.ascii_uppercase)})

class VGG11(nn.Module):
    def __init__(self, num_classes=36):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the pre-trained model
model = VGG11()
model.load_state_dict(torch.load('charviku_dmandava_assignment2_part4.h5', map_location=torch.device('cpu')))
model.eval()

# Function to preprocess and make predictions with class mapping
def predict(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image)
    
    _, predicted = torch.max(outputs, 1)
    prediction_label = predicted.item()
    
    predicted_class = class_mapping.get(prediction_label, "Unknown")
    
    return predicted_class

# Streamlit app
def main():
    st.title('Image Classification with Custom VGG11-like Model')
    st.write('Upload an image for classification:')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Perform prediction on the uploaded image
        if st.button('Classify'):
            prediction_label = predict(uploaded_file)
            st.write('Predicted Class:')
            st.write(prediction_label)
            
if __name__ == '__main__':
    main()




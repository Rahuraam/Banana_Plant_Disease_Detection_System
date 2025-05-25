import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
from googletrans import Translator  # Import the Translator library

# Initialize the translator
translator = Translator()

# Function to translate text
def translate_text(text, target_language="ta"):
    return translator.translate(text, dest=target_language).text

# Load the first-level model that predicts Disease or Variety (using VGG16 as base model)
first_level_model = load_model('category_banana_vgg16_model.h5')  # This model predicts Disease or Variety

# Load the label encoder for the first-level model
with open('category_label_encoder.pkl', 'rb') as file:
    first_level_label_encoder = pickle.load(file)

# Language selector (Dropdown)
language = st.selectbox('Choose Language', ['English', 'Tamil'], index=0)

# Function to get text in the selected language
def get_text_in_language(text, language):
    if language == 'Tamil':
        return translate_text(text, 'ta')
    return text  # Default is English

# Streamlit app interface
st.title(get_text_in_language('Banana Image Classification', language))  # Title in English or Tamil

st.write(get_text_in_language("""
    Upload an image of a banana, and the model will first predict whether it is a Disease or Variety.
    Then, based on that prediction, it will further classify the image into its specific category.
""", language))  # Explanation in English or Tamil

# Function to preprocess the input image for VGG16
def preprocess_image_for_vgg16(img):
    img = cv2.resize(img, (128, 128))  # Resize to 224x224 for VGG16
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess for VGG16
    return img

# Upload image
uploaded_file = st.file_uploader(get_text_in_language("Choose an image...", language), type=["jpg", "jpeg", "png"])

# Disease and Variety information (for display)
info = {
    "RESIZED BANANA APHIDS": get_text_in_language("Banana Aphids are small insects that suck sap from banana plants, causing yellowing and stunted growth. Remedy: Apply insecticidal soap or neem oil.", language),
    "RESIZED BLACK SIGATOKA": get_text_in_language("Black Sigatoka is a fungal disease that causes black streaks on banana leaves. Remedy: Use fungicides like Mancozeb or Copper-based fungicides.", language),
    "RESIZED BANANA FRUIT- SCARRING BEETLE": get_text_in_language("This beetle causes scarring on banana fruit. Remedy: Use insecticides to control beetle population, and remove infested fruits.", language),
    "RESIZED PANAMA DISEASE": get_text_in_language("Panama disease is caused by a soil-borne fungus, leading to wilting and yellowing. Remedy: There is no known cure, but using resistant banana varieties is recommended.", language),
    "RESIZED BACTERIAL SOFT ROT": get_text_in_language("This bacterial infection causes the fruit to rot. Remedy: Cut off infected parts, use copper-based fungicides, and improve drainage.", language),
    "RESIZED YELLOW SIGATOKA": get_text_in_language("Yellow Sigatoka is another leaf disease caused by a fungus. Remedy: Use fungicides like Propiconazole or Copper-based solutions.", language),
    "RESIZED PSEUDOSTEM WEEVIL": get_text_in_language("The weevil attacks the pseudostem of the banana plant, causing internal damage. Remedy: Apply insecticides and remove infected plants.", language),
    "RESIZED JAHAJI FRUIT": get_text_in_language("Jahaji is a variety known for its sweet, soft fruit, often grown in tropical climates. It is used for eating fresh and in desserts.", language),
    "RESIZED BHIMKOL": get_text_in_language("Bhimgol is a hardy banana variety with a strong plant structure. Its fruit is medium-sized and commonly used in cooking.", language),
    "RESIZED JAHAJI STEM": get_text_in_language("The Jahaji variety is not only known for its fruit but also for its sturdy stem, used in fiber production.", language),
    "RESIZED KACHKOL FRUIT": get_text_in_language("Kachkol bananas are smaller and sweeter, commonly used for snacks or desserts in local markets.", language),
    "RESIZED MALBHOG FRUIT": get_text_in_language("Malbhog is a variety of banana with a unique taste and texture, often referred to as a 'king of bananas' in some regions.", language),
    "RESIZED JAHAJI LEAF": get_text_in_language("Jahaji banana leaves are used extensively in traditional cooking for wrapping foods, particularly in tropical cuisines.", language),
    "RESIZED MALBHOG LEAF": get_text_in_language("Malbhog leaves are known for their durability and are often used to wrap foods, lending a distinct aroma to the dish.", language),
}

if uploaded_file is not None:
    # Read and display the uploaded image
    image_data = uploaded_file.read()
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_resized = preprocess_image_for_vgg16(img)  # Resize and preprocess image for VGG16

    # Step 1: First-level model prediction (Disease or Variety)
    first_level_prediction = first_level_model.predict(img_resized)
    first_level_pred_idx = np.argmax(first_level_prediction, axis=1)
    
    # Decode the prediction
    try:
        first_level_pred = first_level_label_encoder.inverse_transform(first_level_pred_idx)[0]
    except ValueError as e:
        st.error(f"Unexpected prediction index: {first_level_pred_idx}. Try retraining the label encoder.")
        st.stop()

    # Step 2: Load the appropriate model based on the first-level prediction
    if first_level_pred == "Banana Disease":
        # Load the disease model and label encoder
        model = load_model('disease_banana_vgg16_model.h5')
        with open('disease_label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)
    else:
        # Load the variety model and label encoder
        model = load_model('variety_banana_vgg16_model.h5')
        with open('variety_label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)

    # Step 3: Predict with the selected model (Disease or Variety specific)
    prediction = model.predict(img_resized)
    predicted_class_idx = np.argmax(prediction, axis=1)

    try:
        predicted_class = label_encoder.inverse_transform(predicted_class_idx)[0]
    except ValueError as e:
        st.error(f"Unexpected class prediction: {predicted_class_idx}. Check label encoder consistency.")
        st.stop()

    # Display the results
    st.image(uploaded_file, caption=get_text_in_language('Uploaded Image.', language), use_container_width=True)
    st.write(f"{get_text_in_language('Disease or Variety: ', language)} {first_level_pred}")
    st.write(f"{get_text_in_language('Specific Prediction: ', language)} {predicted_class}")

    # Display additional information based on prediction
    st.write(f"{get_text_in_language('Information: ', language)} {info.get(predicted_class, get_text_in_language('Information not available for this variety.', language))}")

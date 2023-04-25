# importing the libraries

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import time
import lion_tf2

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def streamlit_load_model():
    model = load_model('./HAM10000_Xception_dropout015_lion.h5', compile=False)
    model.compile(optimizer=lion_tf2.Lion(learning_rate=0.00005),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


def predict_class(image_path, model):
    # test some picture here
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img)
    img = img.astype(np.float32) / 255
    # this is for general testing
    predictions = model.predict(img)
    pred = np.argmax(predictions, axis=1)
    class_names_long = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 'Dermatofibroma',
                        'Melanocytic nevi', 'Melanoma', 'Vascular lesion']
    acc = f"{(np.max(predictions)) * 100:.2f}"
    result = f"{class_names_long[pred[0]]}"

    return acc, result


# Designing the interface
model = streamlit_load_model()
st.title("Skin Doctor")
# For newline
st.write('\n')

image = Image.open('images/image.png')
show = st.image(image, use_column_width=True)

st.sidebar.title("Upload Image")

# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader("Upload a photo of mole.", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)
else:
    # For newline
    st.sidebar.write('\n')

    if st.sidebar.button("Click Here to Classify"):
        if uploaded_file is None:
            st.sidebar.write("Please upload an Image to Classify")
        else:
            with st.spinner('Classifying ...'):
                acc, result = predict_class('images/image.png')
                time.sleep(2)
                st.success('Done!')
            st.sidebar.header("Algorithm Predicts: ")
            # Formatted probability value to 3 decimal places

            # Classify cat being present in the picture if prediction > 0.5
            if result != 'Melanoma':
                st.sidebar.write("종양의 종류 : 양성", '\n')
                st.sidebar.write('**Probability: **', acc, '%', '\n')
                st.sidebar.write('양성 종양은 위험한 종양(암)이 아님을 의미합니다.')
            else:
                st.sidebar.write("종양의 종류: 악성", '\n')
                st.sidebar.write('**Probability: **', acc, '%')
                st.sidebar.write("악성 종양은 위험한 종양이며, 반드시 치료가 필요하니 의사와 상의하시기 바랍니다.")

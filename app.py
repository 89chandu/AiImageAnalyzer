import streamlit as st
from PIL import Image
import numpy as np
import os

from google import genai
from dotenv import load_dotenv


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

st.title("AI Image Analyzer")

uploaded_file = st.file_uploader(
    "Upload An Image",
    type=["jpg","jpeg","png"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Upload Image"
    )

    image_array = np.array(image)

    st.write("NumPy Array")
    # st.write(image_array)


    #1 image dimensions

    height , width = image_array.shape[:2]

    st.write("Height",height)
    st.write("Width",width)

    # 2 . Numbers of channels

    channels = image_array.shape[2]
    st.write("Channels :",channels)

    # 3. Data type 
    st.write("Data Type :",image_array.dtype)

    #4.  Pixel Values

    pixel = image_array[0,0]

    st.write("Pixel Values",pixel)

    # red 
    # green
    # blue

    # Image Statics

    st.write(
        "Minimum Pixel value",
        np.min(image_array)
    )

    st.write("Maximum Pixel Value",
             np.max(image_array)
             )

    st.write("Average Pixel Value",
             np.mean(image_array)
             )

    red = image_array[:,:,0]
    green = image_array[:,:,1]
    blue = image_array[:,:,2]

    red_mean  = np.mean(red)
    green_mean = np.mean(green)
    blue_mean = np.mean(blue)

    st.write("🔴 Red Average :",round(red_mean,2))
    st.write("🟢 Green Average :",round(green_mean,2))
    st.write("🔵 Blue Average :",round(blue_mean,2))


    col1 , col2 ,col3 = st.columns(3)


    col1.metric(
        "red",
        f"{red_mean:.2f}"
    )

    col2.metric(
        "green",
        f"{green_mean:.2f}"
    )

    col3.metric(
        "blue",
        f"{blue_mean:.2f}"
    )


st.subheader("AI Image Analyzer")

if st.button("Analyze with Gemini "):
    with st.spinner("Gemini is analyzing image......"):

        response = client.models.generate_content(
            model="gemini-3.8-flash",

            contents=[
                image,
                """
                Analyze this image.

                tell me :
                1.What is visible ?
                2.Main objects or subjects ? 
                3.scene description ? 
                4.important visual details ?
                """
            ]
        )

    st.write(response.text)    




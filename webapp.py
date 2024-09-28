import pandas as pd
import numpy as np
import streamlit as st
import tensorflow.keras as keras
from pandas.io.formats.style import Styler
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# ("freedraw", "line", "rect", "circle", "transform", "polygon", "point")
drawing_mode = 'freedraw'
realtime_update = True
# initial_drawing = json.load(open("initial_drawing.json"))


model = keras.models.load_model("model/CNN_model_TF_2_10_0.h5")

def main():
    st.title("Handwritten Digit Recognition app")
    st.markdown(
        """
    Draw a digit on the canvas, get the prediction from the model.
    """
    )


    st.sidebar.title("Sidebar")
    drawing_mode = st.sidebar.selectbox(
            "Drawing tool:",
            ("freedraw", "transform"),
        )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10)



    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        height=150,
        width=150,
        stroke_width=stroke_width,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        # initial_drawing=initial_drawing,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )


    if (canvas_result.image_data is not None) and (np.any(canvas_result.image_data)):
        # st.text(canvas_result.image_data.shape)    # (150, 150, 4)
        
        # prediction with directly canvas image
        with st.container(border=True, height=None) :
            st.write("Below the prediction performed with the original image")
            st.image(canvas_result.image_data)

            # resize image to 28x28
            image = Image.fromarray(canvas_result.image_data[:,:,-1])
            resized_image = image.resize((28, 28))

            # rescale image to shape and value required by the model
            image_array = np.array(resized_image) / 255.0
            black_image = image_array.reshape(1, 28, 28, 1)
            prediction = model.predict(black_image)

            # visualize the prediction result
            df = pd.DataFrame(prediction)
            df_styler = Styler(df)
            df_styler.highlight_max(color='lightgreen', axis=1)
            df_styler.format("{:.2%}")
            st.dataframe(df_styler, width=None)

        with st.container(border=True, height=None) :

            # centralize the image
            digit_pixels = canvas_result.image_data[:,:,-1] > 0
            pixel_coords = np.argwhere(digit_pixels)

            # find edge pixels coordinates
            leftest = np.min(pixel_coords[:,1])
            rightest = np.max(pixel_coords[:,1])
            topest = np.min(pixel_coords[:,0])
            bottomest = np.max(pixel_coords[:,0])
            width = rightest - leftest
            height = bottomest - topest
            # st.write(leftest, rightest, width)
            # st.write(topest, bottomest,height)
            higher_dimention = max(width, height)

            # crop image array according to the edge pixels
            centralized_image = canvas_result.image_data[:,:,-1][topest:bottomest, leftest:rightest]
            
            # pad the image to make it square
            padding_pixel = 15
            left_padding = (higher_dimention - width) // 2
            right_padding = higher_dimention - width - left_padding
            top_padding = (higher_dimention - height) // 2
            bottom_padding = higher_dimention - height - top_padding
            padded_image = np.pad(centralized_image, ((top_padding + padding_pixel, bottom_padding + padding_pixel),
                                                      (left_padding + padding_pixel, right_padding + padding_pixel)),
                                                        mode='constant', constant_values=0)
            # st.write(padded_image.shape) # (higher_dimention+padding_pixel, higher_dimention+padding_pixel)

            # visualize centralized image
            zeros = np.zeros((padded_image.shape[0], padded_image.shape[1],3)) # shape (n, n, 3)
            restored_image = np.concatenate((zeros, np.expand_dims(padded_image, axis=-1)), axis=-1)    # shape (n, n, 4)
            st.write("Below the prediction performed after centralizing the original image")        
            with st.container(border=True, height=None) :
                st.image(restored_image/255.0)


            # resize image to 28x28
            image = Image.fromarray(padded_image)
            resized_image = image.resize((28, 28))

            # rescale image to shape and value required by the model
            image_array = np.array(resized_image) / 255.0
            black_image = image_array.reshape(1, 28, 28, 1)
            prediction = model.predict(black_image)

            # visualize the prediction result
            df = pd.DataFrame(prediction)
            df_styler = Styler(df)
            df_styler.highlight_max(color='lightgreen', axis=1)
            df_styler.format("{:.2%}")
            st.dataframe(df_styler, width=None)
        


if __name__ == "__main__":
    main()
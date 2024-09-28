# MNIST-Draw

### [Try the Live Demo deployed on Streamlit Community](https://singledigitrecognizer-mnist.streamlit.app/)

This repository contains a single webpage that allow users to draw digits (0-9) by hand and have it classified in real-time by a simple Convolutional Neural Network model trained on [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.

### CNN model

<div>
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d_15 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">320</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_16 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_17 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)            │          <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)            │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)                │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)                 │         <span style="color: #00af00; text-decoration-color: #00af00">102,500</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">1,010</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>
</div>

### Centralizing images

Above CNN's simplicity does not allow to achieve the shift-invariance to images, thus a simple way to mitigate this problem has been implemented from image side by centralizing and croping to the region containing digit.
![example of shifted digit](https://github.com/user-attachments/assets/69507361-b7dc-4738-9327-639f9b73a343)

## Setup
Python 3.10:
- Tensorflow.keras is used for model training and inference.
- [Streamlit](https://github.com/streamlit/streamlit) is used for web application generation and deployment.
- [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) extension is used to achieve the drawing on page feature.
```bash
# Clone this repository
git clone https://github.com/yiz-wu/SingleDigitRecognizer-MNIST-st.git

# Change to the repo root directory
cd SingleDigitRecognizer-MNIST-st

# Install required modules
pip install -r requirements.txt
```

## Usage
To launch the website locally, execute
```bash
streamlit run webapp.py
```


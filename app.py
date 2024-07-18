import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = tf.keras.models.load_model('custom_model.h5')

# Define the class labels for the emotions
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise',
                'Neutral']


# Define the function for prediction
def predict_emotion(image):
    img = load_img(image, target_size=(48, 48), color_mode="grayscale")
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    emotion = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    return emotion, f"{confidence:.2f}"


# Create the Gradio interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=[gr.Textbox(label="Emotion"), gr.Textbox(label="Confidence")],
    title="Facial Emotion Recognition",
    description="Upload an image to classify the emotion."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()

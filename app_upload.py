# app_upload.py

import os
import tempfile

import streamlit as st
import imageio
import tensorflow as tf

from utils import load_video, num_to_char   # or from utils import load_video, num_to_char
from modelutil import load_model


# ---------- Streamlit Page Config ----------
st.set_page_config(layout="wide", page_title="LipBuddy – Lip Reading Demo")


# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("LipBuddy")
    st.info("Upload a short lip video and the model will try to read the lips.")


# ---------- Main Title ----------
st.title("Lip Reading Demo (Upload a Video)")


# ---------- Video Uploader ----------
uploaded_video = st.file_uploader(
    "Upload a video (ideally similar format/length to your training data)",
    type=["mp4", "avi", "mov", "mkv"]
)

# Cache the model so it's not reloaded on every interaction
@st.cache_resource
def get_model():
    return load_model()

model = get_model()


def preprocess_video(file_bytes) -> tf.Tensor:
    """
    Save uploaded file to a temp path, run the same preprocessing used in training.
    Returns a tensor of shape (T, 46, 140, 1).
    """
    # Write to a temporary file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    # Use your existing video loader
    video_tensor = load_video(temp_path)  # or load_video(temp_path) if you didn't add the helper

    # Clean up if you want (optional)
    # os.remove(temp_path)

    return video_tensor


if uploaded_video is not None:
    # Show original video
    st.subheader("1. Uploaded Video")
    st.video(uploaded_video)

    with st.spinner("Preprocessing video..."):
        video_tensor = preprocess_video(uploaded_video.read())

    # Sanity info
    st.write(f"Extracted frames shape: `{video_tensor.shape}`")

    # Optional: show what the model 'sees' (GIF of preprocessed frames)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Model Input (Grayscale Crop)")
        frames_np = video_tensor.numpy()  # expected (T, 46, 140, 1) or (T, 46, 140)

        # Handle different possible shapes safely
        if frames_np.ndim == 4 and frames_np.shape[-1] == 1:
            # (T, H, W, 1) -> (T, H, W)
            frames_vis = frames_np[..., 0]
        elif frames_np.ndim == 3:
            # already (T, H, W)
            frames_vis = frames_np
        else:
            frames_vis = None

        gif_path = "model_view.gif"

        try:
            if frames_vis is not None and frames_vis.size > 0:
                # Normalize to 0–255 for visualization
                f_min = frames_vis.min()
                f_max = frames_vis.max()
                if f_max > f_min:
                    frames_vis = (frames_vis - f_min) / (f_max - f_min)
                else:
                    frames_vis = frames_vis * 0  # all zeros if constant

                frames_vis = (frames_vis * 255).astype("uint8")  # (T, H, W)

                # imageio expects list/array of 2D images
                imageio.mimsave(gif_path, frames_vis, fps=10)
                st.image(gif_path)
            else:
                st.warning("Could not generate GIF from frames; showing the first frame instead.")
                if frames_np.size > 0:
                    first = frames_np[0]
                    if first.ndim == 3 and first.shape[-1] == 1:
                        first = first[..., 0]
                    st.image(first, clamp=True)
        except Exception as e:
            st.warning(f"Could not generate GIF: {e}. Showing first frame instead.")
            if frames_np.size > 0:
                first = frames_np[0]
                if first.ndim == 3 and first.shape[-1] == 1:
                    first = first[..., 0]
                st.image(first, clamp=True)


    with col2:
        st.subheader("3. Model Prediction")

        # Add batch dimension: (1, T, 46, 140, 1)
        video_batch = tf.expand_dims(video_tensor, axis=0)

        with st.spinner("Running lip reading model..."):
            yhat = model.predict(video_batch)

        # yhat shape: (batch, time, vocab_size)
        time_steps = yhat.shape[1]

        # CTC decode
        decoded, log_prob = tf.keras.backend.ctc_decode(
            yhat,
            input_length=[time_steps],
            greedy=True
        )
        decoder = decoded[0].numpy()[0]   # (sequence_len,)

        # Show raw token indices (optional)
        st.write("Raw decoded token IDs:")
        st.text(decoder)

        # Convert prediction to characters and then string
        text_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")

        st.subheader("4. Decoded Text")
        st.success(text_prediction)

else:
    st.info("👆 Upload a video file to get started.")

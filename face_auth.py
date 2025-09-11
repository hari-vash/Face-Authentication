from deepface import DeepFace
import json
from pathlib import Path
import streamlit as st
import cv2 as cv

def create_student_img_embedding(image_path):
    embedding_obj = DeepFace.represent(img_path=image_path)
    embedding_values = embedding_obj[0]['embedding']
    return embedding_values

def verify_student(embedding_1,embedding_2):
    result = DeepFace.verify(img1_path=embedding_1,img2_path=embedding_2)
    return result['verified']

def stored_embedding_retrieval(student_id):
    path = Path('database.json')
    with path.open('r',encoding='utf-8') as f:
        data=json.load(f)
    idx = data["Student_ID"].index(student_id)
    retrieved_embedding = data["img_embeddings"][idx]
    return retrieved_embedding

st.set_page_config(page_title="Verify Student")
st.title("Verify Student - Capture Selfie")
st.write("Enter the Student ID and capture a selfie using your webcam.")

student_id = st.text_input("Enter your Student ID")

if st.button("Take Selfie"):
    if student_id is None:
        st.warning("Please provide a Studnt ID")
    else:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Make sure your camera is available and not used by another app.")
        else:
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                st.error("Failed to capture image from camera")
            else:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                if verify_student(create_student_img_embedding(frame_rgb), stored_embedding_retrieval(student_id)):
                    print("Face matched")
                else:
                    print("Face not recognized")

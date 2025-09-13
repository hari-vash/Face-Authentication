from deepface import DeepFace
import json
import streamlit as st
from pathlib import Path

path = Path('database.json')

with path.open('r',encoding='utf-8') as f:
    data=json.load(f)

def create_student_id(data):
    id_initial = "SID_"
    id_end = len(data["Student_ID"]) + 1
    student_id = id_initial + str(id_end)
    return student_id
    
def create_student_img_embedding(image_path):
    embedding_obj = DeepFace.represent(img_path=image_path)
    embedding_values = embedding_obj[0]['embedding']
    return embedding_values

ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]


st.set_page_config(page_title="Enroll Student")
st.title("Enroll Student")
st.subheader("Your student ID")
student_id = create_student_id(data)
st.text_input("Student_ID",value=student_id,disabled=True)

uploaded_file = st.file_uploader("Upload a clear face photo (jpg/jpeg/png)",type=ALLOWED_IMAGE_TYPES)


enroll_btn = st.button("Enroll")

if enroll_btn:
    if uploaded_file is None:
        st.warning("Please upload a photo before clicking enroll.")
    else:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext not in ALLOWED_IMAGE_TYPES:
            st.error("Unsupported file type. Please upload jpg or png.")
        else:
            img_embedding = create_student_img_embedding(uploaded_file)
            data["Student_ID"].append(student_id)
            data["img_embedding"].append(img_embedding)
            
            with path.open("w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False)
            
            st.success("Student Enrolled")
            st.write("**Student ID:**",student_id)
            

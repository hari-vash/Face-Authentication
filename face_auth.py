from deepface import DeepFace
import json
from pathlib import Path
import streamlit as st
import cv2 as cv
import threading

face_match = False

def check_face(frame,img_embedding,):
    global face_match
    try:
        if DeepFace.verify(frame,img_embedding.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

def stored_embedding_retrieval(student_id):
    path = Path('database.json')
    with path.open('r',encoding='utf-8') as f:
        data=json.load(f)
    idx = data["Student_ID"].index(student_id)
    retrieved_embedding = data["img_embedding"][idx]
    return retrieved_embedding

st.set_page_config(page_title="Verify Student")
st.title("Verify Student - Capture Selfie")
st.write("Enter the Student ID and capture a selfie using your webcam.")

counter = 0

student_id = st.text_input("Enter your Student ID")

mark_attendance_btn = st.button("Mark Attendance")

if mark_attendance_btn:
    if student_id is None:
        st.warning("Please provide a Student ID")
    else:
        frame_placeholder = st.empty()
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)        
        if not cap.isOpened():
            st.error("Could not open webcam. Make sure your camera is available and not used by another app.")
        else:
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_placeholder.image(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) 
                    if counter%30 == 0:
                        try:
                            retrieved_embedding = stored_embedding_retrieval(student_id)
                            threading.Thread(target=check_face, args=(frame.copy(),retrieved_embedding.copy())).start()
                        except ValueError:
                            pass
                    counter+=1
                    if face_match:
                        cv.putText(frame, "MATCH!", (20, 450), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        st.success("Face Matched")
                        cap.release()
                        frame_placeholder.empty()
                    else:
                        cv.putText(frame, "NO MATCH!", (20, 450), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    
                    
                    

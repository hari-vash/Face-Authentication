from deepface import DeepFace

embedding_objs_1 = DeepFace.represent(img_path="Hariansh_2025.jpg")
print(embedding_objs_1[0]['embedding'])
embedding_objs_2 = DeepFace.represent(img_path="fiyanshu_vashisht.jpeg")

result = DeepFace.verify(img1_path = embedding_objs_1[0]['embedding'], img2_path = embedding_objs_2[0]['embedding'])

print(result)
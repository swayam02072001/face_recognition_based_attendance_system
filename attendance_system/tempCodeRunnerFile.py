def faceEncodings(images):
    encodeList = []
    for img in images:
        #cv2 shows images in BGR format 
        #so we convert it to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
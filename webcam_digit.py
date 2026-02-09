import cv2
import numpy as np
import tensorflow as tf

#loading trained model
model = tf.keras.models.load_model("mnist_digit_model.keras")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    #defining box size where prediction will be taking place
    x1, y1 = int(width*0.4), int(height*0.2)
    x2, y2 = int(width*0.6), int(height*0.4)

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    #cropping the roi frame
    roi = frame[y1:y2, x1:x2]

    #making the captured frame and editing colour etc to make it suitable for predicting
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #resize to mnist size
    digit = cv2.resize(thresh, (28, 28))
    digit = digit.astype("float32") / 255.0
    digit = np.expand_dims(digit, axis=0)

    #prediction
    pred = model.predict(digit, verbose=0)
    number = np.argmax(pred)

    cv2.putText(frame, f"Prediction: {number}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,255,0), 2)

    cv2.imshow("digit Recognition", frame)
    cv2.imshow("processed digit", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
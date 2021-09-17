import cv2
from speak import say

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

detector = cv2.CascadeClassifier('C:\\Program Files\\Python39\\Lib\\site-packages\\opencv-master\\data\\haarcascades_cuda\\haarcascade_frontalface_default.xml')

face_id = input("Enter your Numeric user ID here:  ")

say("Taking samples, look at camera.........")
say("Taking samples, look at camera.........")

print("Taking samples, look at camera..................")
print("Taking samples, look at camera..................")

count = 0
# C:\Program Files\Python39\Lib\site-packages\opencv-master\data\haarcascades_cuda
while True:
    ret, img = cam.read()
    converted_image = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
    faces = detector.detectMultiScale(converted_image, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (225,0,0), 2)
        count +=1

        cv2.imwrite("samples/face." + str(face_id) + ',' + str(count) + "jpg" , converted_image[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 50:
        break

print("Samples taken now closing the program.........")
print("Samples taken now closing the program.........")
print("Samples taken now closing the program.........")
print("Samples taken now closing the program.........")
print("Samples taken now closing the program.........")
print("Samples taken now closing the program.........")
print("Samples taken now closing the program.........")
cam.release()
cv2.destroyAllWindows()


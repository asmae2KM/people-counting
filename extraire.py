import cv2
video_path = 'vedio.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(f'image/image{i}.png', frame)
    print(f'frame {i} was saved')
    i+=1

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
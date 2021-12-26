import cv2, datetime, imutils, numpy as np

#Use the VideoCapture to access webcamera
cap = cv2.VideoCapture(0)
cap.release()
cap = cv2.VideoCapture(0)

#Use haarcascades to detect all faces and bodies
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Initialize the frame work
first_frame = 0
next_frame = 0

# Counters and font
font = cv2.FONT_HERSHEY_SIMPLEX
slower_movement = 0
detect_movement = 0

#Frames in motion count
frames_motion = 20

# How much motion is detected
movement_in_frame = 100

while True:

    # Set movement to false
    movement_motion = False

    # Read frame
    reading, frame = cap.read()
    text = "No movement detected."

    # If camera capture is not working
    if not reading:
        print("Camera is not capturing.")
        continue

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # GaussianBlur removes noise
    grayscale = cv2.GaussianBlur(grayscale, (25, 25), 0)

    # If the first frame is nothing, initialise it
    if first_frame is None: first_frame = grayscale

    slower_movement += 1

  #Detect smaller, slower motions in the frame of camera
    if slower_movement > frames_motion:
        slower_movement = 0
        first_frame = next_frame

    # Set the next frame to compare (the current frame)
    next_frame = grayscale

    #Take the two frames, then find the difference
    frame_delta = cv2.absdiff(first_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]

    # Fill in holes via dilate(), and find contours
    thresh = cv2.dilate(thresh, None, iterations = 4)
    (contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for contour in contours:

        # Save the coordinates of all found contours
        (x, y, w, h) = cv2.boundingRect(contour)

        # Check if the contour is large enough to be detected
        if cv2.contourArea(contour) > 1500:
            movement_motion = True

            # Draw a rectangle around the movements
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            frame_size = (int(cap.get(4)), int(cap.get(4)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # The moment something moves momentarily, reset the persistent
    # movement timer.
    if  movement_motion == True:
        movement_occur = True
        detect_movement = movement_in_frame

    # As long as there was a recent transient movement, say a movement
    # was detected
    if detect_movement > 0:
        text = "Movement detected on camera. " + str(detect_movement) + "%"
        detect_movement -= 1

    else:
        text = "No movement has been detected."

    # Print text on the screen and display
    cv2.putText(frame, str(text), (10, 20), font, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # Delta to color
    frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)

    # Stack the frames together vertically
    cv2.imshow("Camera Motion Detection Frame", np.vstack((frame_delta, frame)))

    # If press Q then quit the open CV program
    if cv2.waitKey(1) == ord('q'):
        break

    #Give a recording of the motion with date and time.
    current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    out = cv2.VideoWriter( f"{current_time}.mp4", fourcc, 20, frame_size)

# Cleanup when closed
cap.release()
out.release()
cv2.destroyAllWindows()


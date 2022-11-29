import cv2
import os
import numpy as np

# load network model:
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Use the dnn.blobFromImage function to construct an input blob by resizing the image to a fixed 300x300 pixels and then normalizing it.
def detectFaceViaCafeeNeuralNetwork(img):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    (h, w) = img.shape[:2]

    my_list = []

    # convert detections to usable coordinates
    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence threshold
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected = (startX, startY, endX, endY)
            print(detected)
            my_list.append(detected)

    return my_list

def load_picture(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def frames_to_video(inputpath,outputpath,fps):

    image_array = []
    files = [f for f in os.listdir(inputpath) if os.isfile(os.join(inputpath, f))]
    # files.sort()//todo:sort
    for i in range(len(files)):
        img = cv2.imread(inputpath + files[i])
        size =  (img.shape[1],img.shape[0])
        img = cv2.resize(img,size)
        image_array.append(img)
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(outputpath,fourcc, fps, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()

# Haar cascade face detector
haarcascade_frontalface_default = 'haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_frontalface_default)
haarcascade_profileface = 'haarcascades/haarcascade_profileface.xml'
profileface_cascade = cv2.CascadeClassifier(haarcascade_frontalface_default)
haarcascade_lowerbody = 'haarcascades/haarcascade_lowerbody.xml'
lowerbody_cascade = cv2.CascadeClassifier(haarcascade_frontalface_default)
haarcascade_upperbody = 'haarcascades/haarcascade_upperbody.xml'
upperbody_cascade = cv2.CascadeClassifier(haarcascade_frontalface_default)


def get_detected_persons_positions(image):
    faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    profile_faces_detected = profileface_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    # lowerbody_detected = lowerbody_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    upperbody_detected = upperbody_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

    rects = faces_detected
    if len(profile_faces_detected):
        np.concatenate((rects, profile_faces_detected), axis=0)
    if len(upperbody_detected):
        np.concatenate((rects, upperbody_detected), axis=0)
        # np.concatenate((faces_detected,
        #           profile_faces_detected,
        #           lowerbody_detected,
        #           upperbody_detected),axis=0)

    if len(rects) > 0:
        return True, rects
    else:
        return False, None


def anonimize_video(video_file_name, output_video=False):
    video = cv2.VideoCapture(video_file_name)
    fps = video.get(cv2.CAP_PROP_FPS)
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"{video_file_name} resolution: {video_width} x {video_height}, fps: {fps}")

    if output_video:
        # output video:
        outputpath = 'processed_' + video_file_name
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out_video = cv2.VideoWriter(filename=outputpath, fourcc=fourcc, fps=fps, size=(video_width, video_height))

    print(f'file:{video_file_name}')
    print('PRESS ESCAPE BUTTON TO SKIP')

    while True:
        ok, frame = video.read()
        if not ok:
            #end of file
            break

        bboxNN = detectFaceViaCafeeNeuralNetwork(frame)
        print(bboxNN)
        #     todo: do same face detection via neural network!
        find, bboxs = get_detected_persons_positions(frame)
        processed_frame = np.copy(frame)
        if not find:
            #same frame to output
            print('not detected')
        if find:
            for bbox in bboxs:
                (x, y, w, h) = [int(v) for v in bbox]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)

                # to_blur = np.zeros((h, w, 3))
                fragment = frame[y:y + h, x:x + w]
                x_n = 0

                blur_fragment = cv2.blur(fragment,(int(video_height/10),int(video_width/10)), 40000)


                print(f'h = {h}, w = {w}')

                for x__ in range(x, x + w):
                    y_n = 0
                    for y__ in range(y, y + h):
                        # print(f'x = {x__}, y = {y__}, color = {frame[x__][y__]}')
                        frame[y__][x__] = blur_fragment[y_n][x_n]
                        y_n = y_n + 1
                    x_n = x_n + 1
        else:
            print('no find face')

        if output_video:
            out_video.write(frame)
        cv2.imshow(f'file:{video_file_name}, press ESC to skip', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # if ESC pressed
            if output_video:
                out_video.release()
            break
    if output_video:
        out_video.release()


def detect_object_coordinates(video_file):
    video = cv2.VideoCapture(video_file)
    _, frame = video.read()
    bbox = cv2.selectROI(frame)
    print(f'{video_file} selected:{bbox}')

def main():
    # anonimize_video('video1.mp4')
    anonimize_video('video2.mp4')
    anonimize_video('video3.mp4')
    anonimize_video('video4.mp4')

if __name__ == "__main__":
    main()
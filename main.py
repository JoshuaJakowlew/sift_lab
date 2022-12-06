import cv2 as cv
import numpy as np

def filter_matches(knn_matches):
    ratio_thresh = 0.7

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    return good_matches

minHessian = 400

detector = cv.SIFT_create()
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

model = cv.imread("./200.png", cv.IMREAD_GRAYSCALE)
model_keypoints, model_descriptors = detector.detectAndCompute(model, None)

cap = cv.VideoCapture("./video.mp4")

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('sift.mp4', fourcc, 25.0, (1862, 1200))

logs = []

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    height, width, channel = frame.shape[:3]
    gray = np.zeros([height, width, 1], dtype=np.uint8)
    cv.cvtColor(frame, cv.COLOR_BGR2GRAY, gray)

    frame_keypoints, frame_descriptors = detector.detectAndCompute(gray, None)

    knn_matches = matcher.knnMatch(model_descriptors, frame_descriptors, 2)

    good_matches = filter_matches(knn_matches)

    img_matches = np.empty((max(model.shape[0], gray.shape[0]), model.shape[1] + gray.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(model, model_keypoints, gray, frame_keypoints, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    out.write(img_matches)

    cv.imshow('Good Matches', img_matches)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    log = f'{i} {len(frame_descriptors)} {len(model_descriptors)} {len(good_matches)}'
    logs.append(log)

    i += 1

with open("myfile.txt", "w") as file:
    file.write('\n'.join(logs))

cap.release()
out.release()
cv.destroyAllWindows()

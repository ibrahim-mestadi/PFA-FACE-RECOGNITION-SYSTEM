from imutils import paths
import face_recognition
import pickle
import cv2
import os

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
curr_path = os.getcwd() 
data_base_path = os.path.join(curr_path, 'database')
print(data_base_path)
imagePaths = []
for path, subdirs, files in os.walk(data_base_path):
    for name in files:
        imagePaths.append(os.path.join(path, name))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	print(imagePath)
	name = imagePath.split(os.path.sep)[-2]
	print(name)
	# load the input image and convert it from BGR (OpenCV ordering)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,model='hog')
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
#stock the dataset in file.pickle
f = open(encodings.pickle, "ab")
f.write(pickle.dumps(data))
f.close()




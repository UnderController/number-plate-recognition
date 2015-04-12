This project is based on Qt IDE with openCV2.4.3
Automatic Number Plate Recognization on Spain
First Step:
Plate Detection
	Method used includes 1)Sobel 2)Threshold 3)Morphography 4)Varify the candidate region size 
	5)The above steps obtain the region approximate by Rect, then use FloodFill to get the background region(Plate background is white) varify the size again
	6)The image must be set in uniform size 33*144 Use the SVM to detect the whether it is a plate
Charater Recognization
	Method used includes: 1)Threshold 2)Find the contours 3)Varify the size 
	4)Extract the feature, features includes:Horizontal histogram and veritical histogram, the image must be set in uniform size 20*20 and low sampling representaion
	5)ANN classification 3 layer, just one hidden layer
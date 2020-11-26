import cv2
from cv2 import dnn_superres
# Create an SR object - only function that differs from c++ code
sr = dnn_superres.DnnSuperResImpl_create()
# Read image
image = cv2.imread('./input.png')
# Read the desired model
path = "EDSR_x4.pb"
sr.readModel(path)
# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)
# Upscale the image
result = sr.upsample(image)
# Save the image
cv2.imwrite("./upscaled.png", result)
cv2.imshow( " Resized image (CNN) " , result)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
img = image 
 
print ( ' Original Dimensions : ' ,img.shape)
 

width = int(img.shape[1] *4 )
height = int(img.shape[0] * 4 )
dim = (width, height)
 # resize image 
resized = cv2.resize(img, dim) #default inter_linear
 
print ( ' Resized Dimensions : ' ,resized.shape)
 # Save the image
cv2.imwrite("./resized.png", resized)
cv2.imshow( " Resized image (interpolation) " , resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
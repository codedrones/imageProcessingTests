import cv2 as cv
import numpy as np
import argparse

def saveImageProcessingStep(imageName, suffix, image):
    cv.imwrite(imageName+suffix+'.jpg',image)

my_parser = argparse.ArgumentParser();
my_parser.add_argument('input_path')

args = my_parser.parse_args()

input_path = args.input_path

print("Reading image "+input_path)
# read image
img = cv.imread(input_path)

#convert to grayscale
bw_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
saveImageProcessingStep(input_path,'_bw',bw_img)

#gaussian blur
blur = cv.GaussianBlur(bw_img,(5,5),0)
saveImageProcessingStep(input_path,'_blur',blur)

#thresholding with Otsu's algorithm
ret,th = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
saveImageProcessingStep(input_path,'_binary',th)

#canny edge detection
edges = cv.Canny(th,100,400,apertureSize = 3)
saveImageProcessingStep(input_path,'_edges',edges)

#get all lines
lines = cv.HoughLinesP(edges,1,np.pi/180,30,minLineLength=40,maxLineGap=5)

fld = cv.ximgproc.createFastLineDetector()
fld_lines = fld.detect(edges)

fld_img = fld.drawSegments(img,fld_lines)
saveImageProcessingStep(input_path,'_fld_lines',fld_img)

#superimpose lines on original image
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),1)

saveImageProcessingStep(input_path,'_lines',img)






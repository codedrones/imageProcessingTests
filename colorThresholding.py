import cv2 as cv
import argparse

lower_color_bound = cv.Scalar(100,0,0)
upper_color_bound = cv.Scalar(225,80,80)

my_parser = argparse.ArgumentParser();
my_parser.add_argument('input_path')

args = my_parser.parse_args()

input_path = args.input_path

print("Reading image "+input_path)
# read image
img = cv.imread(input_path)


#create a mask with color threshold


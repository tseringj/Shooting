
import json
import cv2 as cv2
import numpy as np
import imutils
import ultralytics
from ultralytics import YOLO
ultralytics.checks()


def getScore(radius_list, HoleDist, count_score, range_list ): #function to assign a score to each hole

  for key in range_list.keys():
    # print(range_list[key][1], HoleDist, range_list[key][0])
    if range_list[key][1] <= HoleDist <= range_list[key][0]:
      count_score[key] += 1
  return count_score
  
def centroid(contour):
    M = cv2.moments(contour)
    cx = int(round(M['m10']/M['m00']))
    cy = int(round(M['m01']/M['m00']))
    centre = (cx, cy)
    return centre


def findcentre(thresh, image):
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Get the outer contour (largest area contour)
  outer_contour = max(contours, key=cv2.contourArea)

  # Calculate the center of the outer contour
  M = cv2.moments(outer_contour)
  center_x = int(M["m10"] / M["m00"])
  center_y = int(M["m01"] / M["m00"])
  print(center_x,center_y)

  (x, y), radius = cv2.minEnclosingCircle(outer_contour)
  radius = int(radius)
  print(f'{radius=}')

  # Draw the contour and its center on the original image
  cv2.drawContours(image, [outer_contour], -1, (0, 255, 0), 4)
  #cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), 4)
  # cv2.imshow('outer', image)
  # cv2.waitKey(0) 
  # cv2.destroyAllWindows()
  return center_x, center_y, radius

def detectring(default):  
  
  default = cv2.resize(default,(640,640),cv2.INTER_AREA)

  # Convert to grayscale.
  gray = cv2.cvtColor(default, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (9, 9), 0)
  # cv2.imshow('gray', gray)
  # cv2.waitKey(0) 
  # cv2.destroyAllWindows()

  # # Apply morphology to connect unconnected components
  # kernel = np.ones((5, 5), np.uint8)
  # erode = cv2.erode(blurred, kernel, iterations=1)
  # dilated = cv2.dilate(erode, kernel, iterations=1)

  # _, optimal_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  # print(optimal_threshold)
  # edges = cv2.Canny(gray, int(optimal_threshold * 0.5), optimal_threshold)

  edges = cv2.Canny(gray, 50, 60)
  # cv2.imshow('edges', edges)
  # cv2.waitKey(0) 
  # cv2.destroyAllWindows()

  # cv2_imshow(edges)
  # dilated = cv2.dilate(edges, kernel, iterations=1)
  # erode = cv2.erode(dilated, kernel, iterations=1)
  #cv2_imshow(edges)


  kernel = np.ones((5,5),np.uint8)
  closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
  # cv2.imshow('closing', closing)
  # cv2.waitKey(0) 
  # cv2.destroyAllWindows()
  binary = cv2.threshold(closing, 200, 255, cv2.THRESH_BINARY)[1]
  # cv2.imshow('binary', binary)
  # cv2.waitKey(0) 
  # cv2.destroyAllWindows()

  # # cv2_imshow(thresh)
  img1 = default.copy()
  center_x, center_y, radius = findcentre(binary, img1)
  print(f'{radius}')
  img2 = default.copy()
  cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  # cnts = sorted(cnts, key=cv2.arcLength(cnts), reverse=True)


  # # Draw contours on the original image
  contour_image = default.copy()
  img2 = default.copy()

  center_list = []
  axes_list = []
  angle_list = []
  for c in cnts:
      if cv2.contourArea(c) > 1000:
        print(cv2.contourArea(c))
        epsilon = 0.01 * cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, epsilon, True)
        #cv2.polylines(contour_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2) 
        # cv2.drawContours(contour_image, c, -1, (0, 255, 0), 4)

        ellipse = cv2.fitEllipse(polygon)
        print(ellipse)
        # center = tuple(map(int, ellipse[0]))
        # axes = tuple(map(int, ellipse[1]))
        center = ellipse[0]
        axes = ellipse[1]
        angle = ellipse[2]

        # Print the parameters
        print("Center:", center)
        print("Axes:", axes)
        print("Angle:", angle)
        # cv2.ellipse(contour_image, ellipse, (0, 255, 0), 2)
        # cv2.imshow('contour', contour_image)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        # center, r = cv2.minEnclosingCircle(polygon)
        # center = tuple(map(int, center))
        # r = int(r)
        

        if axes not in axes_list:


          if len(axes_list) != 0 and abs(axes_list[-1][0]-axes[0])>10 and abs(axes_list[-1][0]-axes[1])>10:
            print(axes)
            axes_list.append(axes)
            center_list.append(center)
            angle_list.append(angle)
            # cv2.ellipse(img2, (center, axes, angle), (0, 255, 0), 2)
            # cv2.imshow('ellipse', img2)
            # cv2.waitKey(0) 
            # cv2.destroyAllWindows()

          elif len(axes_list) == 0:
            print(axes)
            axes_list.append(axes)
            center_list.append(center)
            angle_list.append(angle)
            # cv2.ellipse(img2, (center, axes, angle), (0, 255, 0), 2)
            # cv2.imshow('ellipse', img2)
            # cv2.waitKey(0) 
            # cv2.destroyAllWindows()
  print(center)
  img3 = default.copy()
  print(axes_list)
  for axes in axes_list:
    cv2.ellipse(img3, (center, axes, angle),  (0, 255, 0), 2)
  # cv2.imshow('final ellipse', img3)
  # cv2.waitKey(0) 
  # cv2.destroyAllWindows()
  output = {}
  output['center'] = center_list
  output['axes_list'] = axes_list 
  output['angle'] = angle_list           
  return img3, output


if __name__ == "__main__":

  for i in range(37, 38):
   default = cv2.imread('input/bullseye'+ str(i) + '.jpg', cv2.IMREAD_COLOR)
   cv2.imwrite('reference/ref'+ str(i)+'.jpg', image)
   image, output = detectring(default)
   cv2.imwrite('Output/contour'+ str(i)+'.jpg', image)
   with open('train/param' + str(i)+ '.json', "w") as json_file:
           json.dump(output, json_file, indent=4)
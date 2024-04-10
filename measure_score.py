
import math
import json
import cv2 as cv2
import numpy as np
import imutils
import ultralytics
from ultralytics import YOLO
ultralytics.checks()

def point_inside_ellipse(x, y, center_x, center_y, a, b):
    distance_squared = ((math.pow((x - center_x), 2) / math.pow(a, 2)) +
         (math.pow((y - center_y), 2) / math.pow(b, 2)))
    print(f'{distance_squared=}')
    return distance_squared <= 1


def getScore(image, data, centers): #function to assign a score to each hole
  center_list = data['center']
  axes_list = data['axes_list']
  angle_list = data['angle']

  print(axes_list)


  n_ring = len(axes_list)
  count_score = {}
  for i in range(n_ring):
    count_score[i+1] = 0

  
  for c in centers: #plot bullet holes


      pt_x, pt_y = c
      cv2.circle(image, (int(pt_x), int(pt_y)), 5, (0, 0, 255), 4)
      # distance_hor_axis = abs(pt_x - center_x)
      # distance_ver_axis = abs(pt_y - pt_y)
      print(pt_x, pt_y)
      for i in range(len(axes_list)):

        if i < len(axes_list)-1:
          x_upper, y_upper = center_list[i]
          x_lower, y_lower = center_list[i+1]
          a_upper = axes_list[i][0]/2
          b_upper = axes_list[i][1]/2
          a_lower = axes_list[i+1][0]/2
          b_lower = axes_list[i+1][1]/2
          # result_upper = point_inside_ellipse(pt_x, pt_y, x_upper, y_upper, a_upper/2, b_upper/2)
          # result_lower = point_inside_ellipse(pt_x, pt_y, x_lower, y_lower, a_lower/2, b_lower/2)
          result_upper = point_inside_ellipse(pt_x, pt_y, x_upper, y_upper,  b_upper, a_upper,)
          result_lower = point_inside_ellipse(pt_x, pt_y, x_lower, y_lower,  b_lower, a_lower,)
          # result_upper = point_inside_ellipse(pt_x, pt_y, x_upper, y_upper,   a_upper, b_upper,)
          # result_lower = point_inside_ellipse(pt_x, pt_y, x_lower, y_lower,  a_lower, b_lower)
          print(a_upper, b_upper)
          print(a_lower, b_lower)
          print(result_upper)
          print(result_lower)
          if result_upper and  not result_lower:
            count_score[i+1] += 1
        else:
          x_upper, y_upper = center_list[i]
          a_upper = axes_list[i][0]/2
          b_upper = axes_list[i][1]/2

          # result_upper = point_inside_ellipse(pt_x, pt_y, x_upper, y_upper, a_upper/2, b_upper/2)
          #result_upper = point_inside_ellipse(pt_x, pt_y, x_upper, y_upper, a_upper, b_upper)
          result_upper = point_inside_ellipse(pt_x, pt_y, x_upper, y_upper, b_upper, a_upper)
          print(a_upper, b_upper)

          print(result_upper)
          if result_upper:
            count_score[i+1] += 1
        print(f'{count_score=}')      

  print(axes_list)
  for i, axes in enumerate(axes_list):
    cv2.ellipse(image, (center_list[i], axes, angle_list[i]),  (0, 255, 0), 2)

  return image, count_score
  
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
  cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), 4)
  cv2.imshow('outer', image)
  cv2.waitKey(0) 
  cv2.destroyAllWindows()
  return center_x, center_y, radius



def detect_bullet(model, image):
    results = model.predict(image) 
    result = results[0]
    centers = []
    for box in result.boxes:
        cords = box.xyxy[0].tolist()
        cords = [int(ele) for ele in cords]
        print(f'{cords=}')
        x_min, y_min, x_max, y_max = cords[:4]
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2

        centers.append([x,y])


        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2) 
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)  # Wait for user to press any key (or set a timeout in milliseconds)
    # cv2.destroyAllWindows()

    return image, centers

def detectring(binary, default):  
  cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  # cnts = sorted(cnts, key=cv2.arcLength(cnts), reverse=True)


  # # Draw contours on the original image
  contour_image = default.copy()
  img2 = default.copy()


  radius_list = []
  for c in cnts:
      if cv2.contourArea(c) > 1000:
        print(cv2.contourArea(c))
        epsilon = 0.01 * cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, epsilon, True)
        # cv2.polylines(contour_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2) 
        cv2.drawContours(contour_image, c, -1, (0, 255, 0), 4)
        cv2.imshow('contour', contour_image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        center, r = cv2.minEnclosingCircle(polygon)
        center = tuple(map(int, center))
        r = int(r)

        if r not in radius_list:


          if len(radius_list) != 0 and abs(radius_list[-1]-r)>20:
            print(r)
            radius_list.append(r)
            cv2.circle(img2, center, r, (0, 255, 0), thickness=2)
            cv2.imshow('ring', img2)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

          elif len(radius_list) == 0:
            print(r)
            radius_list.append(r)
            cv2.circle(img2, center, r, (0, 255, 0), thickness=2)
            cv2.imshow('ring', img2)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
            
  return len(radius_list), radius_list

def template_matching( template, image):
    """Perform template matching on the given image with the specified template."""

    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    result = cv2.matchTemplate( gray_template, gray_image, cv2.TM_CCOEFF_NORMED)


    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])


    matched_portion = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return matched_portion

def countscore(ref, default, data):

  default = template_matching(ref, default)

  default = cv2.resize(default,(640,640),cv2.INTER_AREA)

  # Convert to grayscale.
  gray = cv2.cvtColor(default, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (9, 9), 0)
  cv2.imshow('gray', gray)
  cv2.waitKey(0) 
  cv2.destroyAllWindows()

  edges = cv2.Canny(gray, 50, 60)
  cv2.imshow('edges', edges)
  cv2.waitKey(0) 
  cv2.destroyAllWindows()

  kernel = np.ones((5,5),np.uint8)
  closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
  cv2.imshow('closing', closing)
  cv2.waitKey(0) 
  cv2.destroyAllWindows()
  binary = cv2.threshold(closing, 200, 255, cv2.THRESH_BINARY)[1]
  cv2.imshow('binary', binary)
  cv2.waitKey(0) 
  cv2.destroyAllWindows()

  
  img2 = default.copy()
  model = YOLO('runs/detect/train10/weights/best.pt')
  img2, centers = detect_bullet(model, img2)
  cv2.imshow('Result_yolo', img2)
  cv2.waitKey(0) 
  cv2.destroyAllWindows()
  print(f'{centers=}')
  img3 = default.copy()
  image, count_score = getScore(img3, data, centers)

  # center_x, center_y = data['center']
  # for center in centers: #plot bullet holes


  #     pt_x, pt_y = center
  #     HoleDist = np.sqrt((pt_x-center_x)**2 +(pt_y - center_y)**2)

  #     print(HoleDist)
  #     count_score = getScore(axes_list, HoleDist, count_score, radius_range)


  # print(f'{count_score=}')



  return image, count_score

if __name__ == "__main__":

  for i in range(37, 38):
   ref = cv2.imread('reference/ref'+ str(i) + '.jpg', cv2.IMREAD_COLOR)
   default = cv2.imread('input/bullseye'+ str(i) + '.jpg', cv2.IMREAD_COLOR)
   with open('train/param' + str(i)+ '.json', "r") as json_file:
               data = json_file.read()
               json_data = json.loads(data)
   output, count = countscore(ref, default, json_data)
   cv2.imwrite('Output/output'+ str(i)+'.jpg', output)
   with open('results/result' + str(i)+ '.json', "w") as json_file:
           json.dump(count, json_file, indent=4)
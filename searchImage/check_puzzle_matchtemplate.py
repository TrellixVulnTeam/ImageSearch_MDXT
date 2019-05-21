import cv2
import os
import numpy as np
import collections
from matplotlib import pyplot as plt
PLOT = True

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scaled_image(image, pix_size):
    w, h = image.shape[::-1]
    if max(w, h) > pix_size:
        z = max(w, h)/pix_size
        w, h = int(w/z), int(h/z)
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

def search_within_folders(target_image, result_vec):
    for folder in os.listdir(targetpath):
        print("In folder ", folder)
        for picture in os.listdir(targetpath + folder):
            if '.jpg' not in picture:
                continue
            #if '165' not in folder:
            #    continue
            img = cv2.imread(targetpath + folder + "/" + picture, 0)
            wt, ht = target_image.shape[::-1]
            #img = cv2.Canny(img, wt, ht)
            #show_image(img)

            # Apply template Matching
            try:
                res = cv2.matchTemplate(img, target_image, method)
            except:
                continue

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            #print(min_val)
            if min_val < result_vec["global_min_val"]:
                result_vec["global_min_val"] = min_val
                print("folder = ", folder)
                print("path = ", targetpath + folder + "/" + picture)
                result_vec[folder]["path"] = targetpath + folder + "/" + picture
                result_vec[folder]["min val"] = min_val
                if "count" in result_vec[folder].keys():
                    result_vec[folder]["count"] += 1
                else:
                    result_vec[folder]["count"] = 1
                    print("found new one")
                result_vec[folder]["location"] = min_loc
    return result_vec

targetpath = '../get_images/Pictures/'
#puzzlepath = '../puzzle.jpg'
puzzlepath = '../coverraetsel_1705.jpg'
pixel_size_vec = np.linspace(30, 150, (150-30)/5)
puzzle = cv2.imread(puzzlepath, 0)
method = eval('cv2.TM_SQDIFF_NORMED')


#puzzle = cv2.Canny(puzzle, w, h)
show_image(puzzle)
result_vec = collections.defaultdict(dict)
result_vec["global_min_val"] = 100


"""
result_vec = {'global_min_val': 0.08379919081926346,
              '1': {'path': '../get_images/Pictures/1/11.jpg', 'min val': 0.11378612369298935, 'count': 2, 'location': (515, 769)},
              '10': {'path': '../get_images/Pictures/10/101.jpg', 'min val': 0.09999868273735046, 'count': 2, 'location': (322, 278)},
              '104': {'path': '../get_images/Pictures/104/1040.jpg', 'min val': 0.08929844200611115, 'count': 1, 'location': (343, 436)},
              '183': {'path': '../get_images/Pictures/183/1830.jpg', 'min val': 0.08782242983579636, 'count': 1, 'location': (272, 794)},
              '51': {'path': '../get_images/Pictures/51/511.jpg', 'min val': 0.08379919081926346, 'count': 1, 'location': (134, 78)}}
for key in result_vec.keys():
    try:
        print(result_vec[key]['path'])
        #print(result_vec[key].keys())
    except:
        print("ignoring ", key)
for key, value in result_vec.items():
    if key == "path":
        print(value)
        
        
solution:
defaultdict(<class 'dict'>, {'global_min_val': 0.08379919081926346, 
'1': {'path': '../get_images/Pictures/1/11.jpg', 'min val': 0.11378612369298935, 'count': 2, 'location': (515, 769)}, 
'10': {'path': '../get_images/Pictures/10/101.jpg', 'min val': 0.09999868273735046, 'count': 2, 'location': (322, 278)}, 
'104': {'path': '../get_images/Pictures/104/1040.jpg', 'min val': 0.08929844200611115, 'count': 1, 'location': (343, 436)}, 
'183': {'path': '../get_images/Pictures/183/1830.jpg', 'min val': 0.08782242983579636, 'count': 1, 'location': (272, 794)}, 
'51': {'path': '../get_images/Pictures/51/511.jpg', 'min val': 0.08379919081926346, 'count': 1, 'location': (134, 78)}})


"""
for pixel_size in pixel_size_vec:
    puzzle_scaled = scaled_image(puzzle, pixel_size)
    print("pixel_size: ", pixel_size)
    result_vec = search_within_folders(puzzle_scaled, result_vec)

print(result_vec)
for key in result_vec.keys():
    try:
        candidate_path = result_vec[key]['path']
        # print(result_vec[key].keys())
        if PLOT:
            img = cv2.imread(candidate_path, 0)
            wt, ht = puzzle.shape[::-1]
            # w_img, h_img = img.shape[::-1]
            # img = cv2.Canny(img, w_img, h_img)
            try:
                res = cv2.matchTemplate(img, puzzle, method)
            except:
                continue
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = min_loc
            bottom_right = (top_left[0] + wt, top_left[1] + ht)

            cv2.rectangle(img, top_left, bottom_right, 255, 2)

            plt.subplot(121), plt.imshow(puzzle, cmap='gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle('cv2.TM_SQDIFF_NORMED')

            plt.show()

    except:
        print("ignoring ", key)

import cv2
import os
import collections
from matplotlib import pyplot as plt
PLOT = True
VERBOSE = True


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


def search_within_folders(target_image, result_vec, global_min_val):
    for folder in os.listdir(targetpath):
        print("In folder ", folder)
        for picture in os.listdir(targetpath + folder):
            if '.jpg' not in picture:
                continue
            if folder not in ('1', '103', '47', '165'):
                continue
            img = cv2.imread(targetpath + folder + "/" + picture, 0)
            # wt, ht = target_image.shape[::-1]
            # img = cv2.Canny(img, wt, ht)
            # show_image(img)

            # Apply template Matching
            try:
                res = cv2.matchTemplate(img, target_image, method)
            except:
                continue

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val < global_min_val:
                global_min_val = min_val
                if VERBOSE:
                    print("folder = ", folder)
                    print("path = ", targetpath + folder + "/" + picture)
                result = collections.defaultdict(dict)#{}
                result["folder"] = folder
                result["path"] = targetpath + folder + "/" + picture
                result["min_val"] = min_val
                result["location"] = min_loc
                if folder in (entry["folder"] for entry in result_vec):
                    pass
                else:
                    result["count"] = 1
                    result_vec.append(result)
                for entry in result_vec:
                    if folder in entry["folder"]:
                        entry["count"] += 1
    return result_vec, global_min_val


targetpath = '../get_images/Pictures/'
puzzlepath = '../puzzle3.jpg'
# puzzlepath = '../coverraetsel_1705.jpg'
pixel_size_vec = [50] # np.linspace(50, 50, (150-30)/5)
puzzle = cv2.imread(puzzlepath, 0)
method = eval('cv2.TM_SQDIFF_NORMED')


# puzzle = cv2.Canny(puzzle, w, h)
show_image(puzzle)
result_vec = []
global_min_val = 100


for pixel_size in pixel_size_vec:
    puzzle_scaled = scaled_image(puzzle, pixel_size)
    print("pixel_size: ", pixel_size)
    result_vec, global_min_val = search_within_folders(puzzle_scaled, result_vec, global_min_val)

if VERBOSE:
    print(result_vec)
for guess in [result_vec[-1]]:# or ::-1
    print("guess = ", guess)
    print(guess["min_val"])
    try:
        candidate_path = guess["path"]
        if PLOT:
            img = cv2.imread(candidate_path, 0)
            wt, ht = puzzle_scaled.shape[::-1]
            print("wt = ", wt, ht)
            # w_img, h_img = img.shape[::-1]
            # img = cv2.Canny(img, w_img, h_img)
            try:
                res = cv2.matchTemplate(img, puzzle_scaled, method)
            except:
                continue
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = min_loc
            print("min loc = ", min_loc)
            bottom_right = (top_left[0] + wt, top_left[1] + ht)

            cv2.rectangle(img, top_left, bottom_right, 255, 2)

            plt.subplot(121), plt.imshow(puzzle_scaled, cmap='gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle('cv2.TM_SQDIFF_NORMED')

            plt.show()

    except:
        print("ignoring ", guess["folder"])


import cv2
import os
import collections
from matplotlib import pyplot as plt
import numpy as np
PLOT = True
VERBOSE = True
CANNY = False
CHEAT_MODE = False
BEST_MATCH_ONLY = True
CHEAT_FOLDERS = ['103', '47', '121', '165', '34', '158', '55']
HAY_SIZE_PIXELS = 200
NEEDLE_SIZE_PIXELS = 40
NEEDLE_MIN_PIXEL_SIZE = 30
NEEDLE_MAX_PIXEL_SIZE = 60


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()


def scaled_image(image, pix_size):
    w, h = image.shape[::-1]
    z = max(w, h)/pix_size
    w, h = int(w/z), int(h/z)
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def draw_rect(locations, delta_x, delta_y):
    x_init, y_init = locations
    plt.plot((x_init, x_init), (y_init, y_init + delta_y), 'r', linewidth=2.0)
    plt.plot((x_init, x_init + delta_x), (y_init, y_init), 'r', linewidth=2.0)
    plt.plot((x_init + delta_x, x_init + delta_x), (y_init, y_init + delta_y), 'r', linewidth=2.0)
    plt.plot((x_init, x_init + delta_x), (y_init + delta_y, y_init + delta_y), 'r', linewidth=2.0)


def search_within_folders(needle_image, haystack_path, disparity):
    result_list = []
    for folder in os.listdir(haystack_path):
        if VERBOSE:
            print("In folder ", folder)
        for picture in os.listdir(haystack_path + folder):
            if '.jpg' not in picture:
                continue
            if CHEAT_MODE:
                if folder not in CHEAT_FOLDERS:
                    continue
            haystack_image = cv2.imread(haystack_path + folder + "/" + picture, 0)
            haystack_image = scaled_image(haystack_image, HAY_SIZE_PIXELS)
            if CANNY:
                w_needle, h_needle = needle_image.shape[::-1]
                needle_image = cv2.Canny(needle_image, w_needle, h_needle)
                w_haystack, h_haystack = haystack_image.shape[::-1]
                haystack_image = cv2.Canny(haystack_image, w_haystack, w_haystack)

            # Apply template Matching
            res = cv2.matchTemplate(haystack_image, needle_image, method)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if min_val < disparity:
                disparity = min_val
                if VERBOSE:
                    print("folder = ", folder)
                    print("path = ", haystack_path + folder + "/" + picture)
                    print("min value = ", min_val)
                    print("min loc = ", min_loc)
                result = collections.defaultdict(dict)
                result["folder"] = folder
                result["path"] = haystack_path + folder + "/" + picture
                result["min_val"] = min_val
                result["location"] = min_loc
                result["pixel"] = needle_image.shape[::-1]
                if folder in (entry["folder"] for entry in result_list):
                    pass
                else:
                    result["count"] = 1
                    result_list.append(result)
                for entry in result_list:
                    if folder in entry["folder"]:
                        entry["count"] += 1
    return result_list, disparity


dataset_path = '../get_images/Pictures/'
puzzle_path = '../coverraetsel_0507.webp'#passed
puzzle_path = '../coverraetsel_2806.webp'#passed
puzzle_path = '../coverraetsel_2106.webp'#passed
puzzle_path = '../coverraetsel_1704.jpg'
puzzle_path = '../coverraetsel_0305.jpg'#passed
puzzle_path = '../coverraetsel_1406.webp'#passed
puzzle_path = '../coverraetsel_1705.jpg'#passed
puzzle_path = '../coverraetsel_3105.jpg'#passed
puzzle_path = '../coverraetsel_1207.webp'#partially passed
puzzle_path = '../coverraetsel_1907.webp'#passed
puzzle_path = '../coverraetsel_2607.webp'#partially passed

pixel_size_vec = np.linspace(NEEDLE_MIN_PIXEL_SIZE, NEEDLE_MAX_PIXEL_SIZE,
                             int((NEEDLE_MAX_PIXEL_SIZE-NEEDLE_MIN_PIXEL_SIZE)/5))
puzzle_image = cv2.imread(puzzle_path, 0)
method = cv2.TM_SQDIFF_NORMED


if VERBOSE:
    show_image(puzzle_image)
results = []
result_set = []


for pixel_size in pixel_size_vec:
    puzzle_image_scaled = scaled_image(puzzle_image, pixel_size)
    if VERBOSE:
        print("pixel_size: ", pixel_size)
    results, __ = search_within_folders(puzzle_image_scaled, dataset_path, 100)
    result_set = result_set + [results[-1]] if BEST_MATCH_ONLY else result_set + results[::-1]

if VERBOSE:
    print("Possible folder candidates are:")
    for result in results:
        print(result["folder"])
for candidate in result_set:
    if VERBOSE:
        print("guess = ", candidate["folder"])
        print("min value = ", candidate["min_val"])
    candidate_path = candidate["path"]
    if PLOT:
        dataset_image = cv2.imread(candidate_path, 0)
        dataset_image = scaled_image(dataset_image, 200)
        if CANNY:
            w_puzzle, h_puzzle = puzzle_image.shape[::-1]
            puzzle_image = cv2.Canny(puzzle_image, w_puzzle, h_puzzle)
            w_dataset, h_dataset = dataset_image.shape[::-1]
            dataset_image = cv2.Canny(dataset_image, w_dataset, h_dataset)
        w_puzzle, h_puzzle = candidate["pixel"]
        pixel_size = max(w_puzzle, h_puzzle)
        needle_image_scaled = scaled_image(puzzle_image, pixel_size)

        plt.subplot(121), plt.imshow(needle_image_scaled, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dataset_image, cmap='gray')
        draw_rect(candidate["location"], w_puzzle, h_puzzle)

        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle('cv2.TM_SQDIFF_NORMED')

        plt.show()

# ImageSearch
This project uses a dataset (haystack) to find a match with the given puzzle (needle). As a first approach, the OpenCV library in Python has been used.

The dataset is fetched from the first few google result pictures when searching for the three investigators ("Die drei Fragezeichen").
The script simply downloads a number of pictures for each episode and places it into the corresponding folder.

The second script applies an OpenCV function called matchTemplate to look for the puzzle image in a given image (dataset). matchTemplate seems to calculate a convolution between the puzzle and the dataset image to check the similarity. Due to this fact, it is important that the puzzle image and target image are roughly scale 1:1.

The goal of the whole project is to automate visual detection of imageparts within larger images and to state where it is assumed to be.

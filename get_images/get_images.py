#program downloaded from https://stackoverflow.com/questions/20716842/python-download-images-from-google-image-search

from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os
import cookielib
import json


def get_soup(url, header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url, headers=header)), 'html.parser')


for episode in range(1, 200):#make 200
    url = "https://www.google.co.in/search?q=die%20drei%20fragezeichen%20episode%20+" + str(episode) + "&source=lnms&tbm=isch"#die drei fragezeichen episode " + str(episode) + "
    #url = "https://www.google.co.in/search?q=1&source=lnms&tbm=isch"
    print (url)
    DIR = "Pictures/"+str(episode)
    header = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    soup = get_soup(url, header)

    ActualImages = []  # contains the link for Large original images, type of  image
    counter = 0
    for a in soup.find_all("div", {"class": "rg_meta"}):
        link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
        ActualImages.append((link, Type))
        counter = counter + 1
        if counter > 5:
            break

    print("there are total", len(ActualImages), "images for episode", episode)

    if not os.path.exists(DIR):
                os.mkdir(DIR)

    if not os.path.exists(DIR):
                os.mkdir(DIR)
    ###print images

    for j, (img, Type) in enumerate(ActualImages):
        print (j)
        if j < 3:
            try:
                req = urllib2.Request(img, headers={'User-Agent': header})
                raw_img = urllib2.urlopen(req).read()

                #cntr = len([i for i in os.listdir(DIR) if image_type in i]) + 1
                #print (cntr)
                if len(Type) == 0:
                    f = open(os.path.join(DIR, str(episode) + str(j) + ".jpg"), 'wb')
                else:
                    f = open(os.path.join(DIR, str(episode) + str(j) + "."+Type), 'wb')


                f.write(raw_img)
                f.close()
            except Exception as e:
                print ("could not load : " + img)
                print (e)
import argparse
import gmplot
import numpy as np
import os


# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--path", default="routes/default.npy", help="path to route file")
# args = vars(ap.parse_args())


latitude = {}
longitude = {}

route = np.load('routes/default.npy')

gmap = gmplot.GoogleMapPlotter(route[0, 0],
                               route[0, 1], 7)

for i in range(len(route)):
    latitude[i] = [route[i, 0], route[i, 3]]
    longitude[i] = [route[i, 1], route[i, 4]]

    # scatter method of map object
    # scatter points on the google map
    gmap.scatter(latitude[i], longitude[i], '# FF0000',
                 size=40, marker=False)

    # Plot method Draw a line in
    # between given coordinates
    gmap.plot(latitude[i], longitude[i],
              'cornflowerblue', edge_width=2.5)

gmap.draw('map.html')
os.system('open map.html')

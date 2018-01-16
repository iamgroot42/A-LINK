import os
import sys

users = {}

for filename in os.listdir(sys.argv[1]):
        person = filename.split('_')[0]
        if person not in users.keys():
                users[person] = []
        users[person].append(filename)

for dis in users.keys():
        finalpath = sys.argv[2] + dis
        #print finalpath
        os.mkdir(finalpath, 0755)
        for image in users[dis]:
                os.rename(sys.argv[1] + image, finalpath + "/" + image)

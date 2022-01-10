import random
import urllib.request
import csv
import cv2
import os
from statistics import multimode
import torch

HEIGHT = 128
WIDTH = 128


class datagen(object):
    def __init__(self):
        self.labels = {"1": [0, 1, 0, 0, 0],
                       "2": [0, 0, 1, 0, 0],
                       "3": [1, 0, 0, 0, 0],
                       "4": [0, 0, 0, 0, 1],
                       "5": [0, 0, 0, 1, 0]}

        self.inputs = None
        self.image_names = self.create_image_names()
        self.indexes = self.initialize_indexes()  # this is a list of valid indexes for which there are images
        self.targets = self.create_inputs_and_targets()

        return

    def create_image_names(self, images_folder):
        image_names = [f for f in os.listdir(images_folder) if ".DS" not in f]
        self.image_names = image_names[:30000]
        return self.image_names

    def initialize_indexes(self):
        self.indexes = []
        img_indexes1 = []
        for name in self.image_names[:30000]:
            img_indexes1.append(name[:-6])
        img_indexes = list(set(img_indexes1))
        for index in img_indexes:
            for name in self.image_names:
                if name[:-6] == index:
                    int_indexes = int(index)
                    self.indexes.append(int_indexes)

        return self.indexes

    def create_inputs_and_targets(self, input_folder, images_folder):
        img_indexes1 = []
        img_index = []
        self.targets = []
        targets = []
        self.inputs = []
        imp = []
        with open(input_folder, 'r') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                tgts = [row[17], row[19], row[21], row[23], row[25], row[27]]
                label = multimode(tgts)
                if len(label) == 1:
                    targets.append(str(label[0]))
                else:
                    if row[15] == "ONE_CLASS_TRIPLET":
                        targets.append("4")
                    elif row[15] == "TWO_CLASS_TRIPLET":
                        l = random.choice(label)
                        targets.append(str(l))
                    elif row[15] == "THREE_CLASS_TRIPLET":
                        targets.append("5")
                tgts = []
        # print(targets, len(targets))
        for index in self.indexes:
            for name in self.image_names[:30000]:
                if name[:-6] == str(index):
                    imp.append(os.path.join(images_folder, name))
                if len(imp) == 3:
                    self.targets.append(targets[index])
                    break
            self.inputs.append(imp)
            imp = []

        return self.targets

    def get_batch(self, images_folder, batch_size=1):
        inpts = []
        tgts = []
        all_origs = []
        # origs = []
        # images = []
        indexes = random.sample(self.indexes, batch_size)
        for index in indexes:
            origs = []
            images = []
            name0 = str(index) + "_0.jpg"
            name1 = str(index) + "_1.jpg"
            name2 = str(index) + "_2.jpg"
            names = [os.path.join(images_folder, name0), os.path.join(images_folder, name1), os.path.join(images_folder, name2)]
            # print("len targts = ", len(self.targets), index)
            tgts.append(int(self.targets[index]))
            for name in names:
                orig = cv2.imread(name)
                image = cv2.resize(orig, (WIDTH, HEIGHT))
                origs.append(orig)
                images.append(image/255)
            inpts.append(images)
            all_origs.append(origs)

        l = len(inpts)
        tgts = torch.tensor(tgts, dtype=torch.long)
        inpts = torch.tensor(inpts).permute(0, 1, 4, 2, 3)
        return inpts, tgts, origs

    def get_test_image(self, test_path):
        img_indexes1 = []
        im_names = []
        test_inputs = []
        origs = []
        inputs = []
        image_names = os.listdir(test_path)
        for name in image_names[:1000]:
            img_indexes1.append(name[:-6])
        img_indexes = list(set(img_indexes1))
        for index in img_indexes:
            int_index = int(index)
            for name in image_names:
                if name[:-6] == index:
                    im_names.append(os.path.join(test_path, name))
                if len(im_names) == 3:
                    break
            test_inputs.append(im_names)

        img_set = random.choice(test_inputs)
        for img in img_set:
            image = cv2.imread(img)
            image = cv2.resize(image, (WIDTH, HEIGHT))
            origs.append(image)
            image = image/255
            inputs.append(image)

        inputs = torch.tensor(inputs, dtype=torch.float)
        inputs = inputs.permute((0, 3, 1, 2))

        return inputs, origs


if __name__ == '__main__':
    dg = datagen()


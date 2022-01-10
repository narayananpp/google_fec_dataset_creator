import urllib
import numpy as np
import csv
import cv2
import os


class GoogleFECdatasetCreator(object):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        with open(self.csv_path, 'r') as f:
            self.csvreader = csv.reader(f)
            self.data = []
            for row in self.csvreader:
                self.data.append(row)
        # Following are the column numbers with respect to the Google FEC dataset in CSV format
        self.url_indexes = [0, 5, 10]
        self.bbox_indexes = [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]]
        return

    def fetch_url(self, img_url):
        resp = urllib.request.urlopen(img_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def create_images_repository(self, folder, start_index, end_index):
        """
        An index starts from 0. Index specified as 0 corresponds to the index of the first row in csv file
        start_index: The index from which images need to be downloaded
        end_index: The index until which images need to be downloaded(not included)

        """
        valid_indexes = []
        for i, row in enumerate(self.data[start_index:end_index]):
            images = []
            boxes = []
            for j in range(3):
                url = row[self.url_indexes[j]]
                bindex = self.bbox_indexes[j]
                box = [row[bindex[0]], row[bindex[1]], row[bindex[2]], row[bindex[3]]]
                try:
                    image = self.fetch_url(url)
                except:
                    break

                images.append(image)
                boxes.append(box)

            if len(images) == 3:
                valid_indexes.append(i)
                for k in range(3):
                    img_n = str(i) + "_" + str(k) + ".jpg"
                    f = os.path.join(folder, img_n)
                    cv2.imwrite(f, images[k])

        return valid_indexes

    def create_sliced_images(self, folder, start_index, end_index):
        valid_indexes = []
        for i, row in enumerate(self.data[start_index:end_index]):
            images = []
            boxes = []
            for j in range(3):
                url = row[self.url_indexes[j]]
                bindex = self.bbox_indexes[j]
                box = [row[bindex[0]], row[bindex[1]], row[bindex[2]], row[bindex[3]]]
                try:
                    image = self.fetch_url(url)
                except:
                    break

                images.append(image)
                boxes.append(box)

            if len(images) == 3:
                valid_indexes.append(i)
                for k in range(3):
                    img_n = str(i) + "_" + str(k) + ".jpg"
                    f = os.path.join(folder, img_n)
                    self.get_image_slice(images[k], boxes[k], f)

        return valid_indexes

    def create_sliced_images_from_folder(self, folder, out_folder):
        image_names = os.listdir(folder)
        for name in image_names:
            img_path = os.path.join(folder, name)
            index = int(name[:-6])
            img_num = int(name[-5: -4])
            bindex = self.bbox_indexes[img_num]
            box = [self.data[index][bindex[0]], self.data[index][bindex[1]], self.data[index][bindex[2]], self.data[index][bindex[3]]]
            image = cv2.imread(img_path)
            f = os.path.join(out_folder, name)
            img = self.get_image_slice(image, box, f)
            cv2.imwrite(f, img)
        return

    def get_image_slice(self, img, box, fn):
        height, width, depth = img.shape
        x1 = int(width * float(box[0]))
        x2 = int(width * float(box[1]))
        y1 = int(height * float(box[2]))
        y2 = int(height * float(box[3]))
        img = img[y1:y2, x1:x2, :]
        return img


if __name__ == '__main__':
    dg = GoogleFECdatasetCreator('/Users/navakrish/Desktop/Navneeth/internship/datasets/abridged_fec_train.csv')
    indexes = dg.create_sliced_images("/Users/navakrish/Desktop/tmp", 0, 3)
    print(indexes)
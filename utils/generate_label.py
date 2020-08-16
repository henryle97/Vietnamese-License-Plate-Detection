import cv2

from model_center import CENTER_MODEL
import os
from PIL import Image
import pascal_voc_writer
import tqdm


class GenerateLabelObjectDetection:
    """
        Create XML format label from center
    """

    def __init__(self, weight_model_path, img_src_dir, label_dst_dir):
        self.model = CENTER_MODEL(weight_path=weight_model_path)

        self.img_paths = [os.path.join(img_src_dir, x) for x in os.listdir(img_src_dir)]
        self.names = {1: 'corner'}
        self.label_dst_dir = label_dst_dir
        print("Number of images is ", len(self.img_paths), ". Demo path: ", self.img_paths[0])
        if not os.path.exists(label_dst_dir):
            os.mkdir(label_dst_dir)

    def process(self):
        count = 0
        error_img = []
        for img_path in tqdm.tqdm(self.img_paths):
            img_name = os.path.basename(img_path).split(".")[0]
            try:
                img = cv2.imread(img_path)
                h, w, c = img.shape

            except:
                print("Error open image!!!")
                error_img.append(img_path)
                continue

            # print("Predicting ...")
            list_center_label = self.model.detect_corner(img)
            # print(list_center_label)
            writer = pascal_voc_writer.Writer(img_path, w, h)
            for center_label in list_center_label:
                x_c, y_c = center_label[0], center_label[1]
                width_box, height_box = 10, 10  # pixel
                x_min, y_min, x_max, y_max = int(x_c - width_box / 2), int(y_c - height_box / 2), int(
                    x_c + width_box / 2), int(y_c + height_box / 2)
                writer.addObject(self.names[1], x_min, y_min, x_max, y_max)

            writer.save(os.path.join(self.label_dst_dir, img_name + ".xml"))
            count += 1

        print("Processed %d images" % (count))
        with open('error_img.txt', 'w') as f:
            for path in error_img:
                f.write(path + "\n")


if __name__ == "__main__":
    gen = GenerateLabelObjectDetection(weight_model_path="weights/model_last_check.pth"
                                       , img_src_dir="DATA/bienso_3820_images",
                                       label_dst_dir="DATA/bienso_3820_xml")
    gen.process()















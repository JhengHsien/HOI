from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch.nn as nn
import torch

class dino():
    def __init__(self):
        self.model = load_model("/home/nii/Desktop/sin/HOICLIP/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/nii/Desktop/sin/HOICLIP/GroundingDINO/weights/groundingdino_swint_ogc.pth")

    def catch(self, obj_classes, imgs_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # testing stage 
        # obj classes
        img_root = 'data/hico_20160224_det/images/test2015/'
        obj_prompt = ''
        for each_obj in obj_classes:
            obj_prompt = obj_prompt + each_obj + '. '
        human_bboxes = []
        obj_bboxes = []
        # load image

        for img_path in imgs_path:
            path = img_root + img_path
            image_source, image = load_image(path)   
            
            # for each detection
            image_human_detection = []
            image_obj_detection = []

            # detect
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=obj_prompt,
                box_threshold=0.4,
                text_threshold=0.25
            )
            if (len(boxes)>25): boxes, phrases = boxes[:25], phrases[:25]

            # classify human and obj
            for idx, det in enumerate(phrases):
                if det == "person":
                    image_human_detection.append(boxes[idx])
                elif det != "":
                    for each_bbox in boxes:
                        image_obj_detection.append(each_bbox)

            if (len(image_human_detection) == 0):
                human_bboxes.append(torch.tensor([]).to(device))
            else: human_bboxes.append(torch.stack(image_human_detection).to(device))

            if (len(image_obj_detection) == 0):
                obj_bboxes.append(torch.tensor([]).to(device))
            else:obj_bboxes.append(torch.stack(image_obj_detection).to(device))

        return human_bboxes, obj_bboxes
        


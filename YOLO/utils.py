import torch


def iou(box_predict, box_target):
    """
    // yolo version prediction and target box
    input:
    box_predict -- predicted box, (N, 4) 4 <- mid_point_position, height, width
    box_target -- target box, (N, 4) 4 <- mid_point_position, height, width

    output: intersection of union
    """
    #find corner point of intersection region
    box1_x1 = box_predict[...,0:1] - 0.5 * box_predict[...,2:3] #preserve last dimension shape(N,1)
    box1_y1 = box_predict[...,1:2] - 0.5 * box_predict[...,3:4]
    box1_x2 = box_predict[...,0:1] + 0.5 * box_predict[...,2:3]
    box1_y2 = box_predict[...,1:2] + 0.5 * box_predict[...,3:4]

    box2_x1 = box_target[...,0:1] - 0.5 * box_target[...,2:3]
    box2_y1 = box_target[...,1:2] - 0.5 * box_target[...,3:4]
    box2_x2 = box_target[...,0:1] + 0.5 * box_target[...,2:3]
    box2_y2 = box_target[...,1:2] + 0.5 * box_target[...,3:4]

    xi1 = torch.max(box1_x1, box2_x1)
    yi1 = torch.max(box1_y1, box2_y1)
    xi2 = torch.min(box1_x2, box2_x2)
    yi2 = torch.min(box1_y2, box2_y2)

    inter_area = (yi2 - yi1).clamp(0)* (xi2 - xi1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou

def non_max_suppression(pred_boxes, iou_threshold, probability_threshold):
    """"
    #
    imput:
    pred_boxes:list of element [class_idx, object_probability, x, y, w, h]
    iou_threshold: bounding_box predicting same class, whose iou with box with largest object_probability larger than this threshold will be discarded
    probability_threshold: bounding_box with object_probability les than this threshold will be discarded

    output: filtered list of bounding boxes
    """
    assert type(pred_boxes) == list

    #discard bounding boxes with low object_probability
    bboxes = [box for box in pred_boxes if box[1] > probability_threshold]
    bboxes.sort(key=take_Object_Probability, reverse=True)
    bboxes_after_nms = []

    while bboxes:
        first_box = bboxes.pop(0)
        #remain bounding_boxes with different class or same class with less IOU
        bboxes = [box for box in bboxes if box[0] != first_box[0] or iou(torch.tensor(first_box[2:]),torch.tensor(box[2:]))<iou_threshold]
        bboxes_after_nms.append(first_box)

    return bboxes_after_nms

# take second element for sort
def take_Object_Probability(elem):
    return elem[1]






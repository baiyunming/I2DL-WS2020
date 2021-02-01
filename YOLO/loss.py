import torch
import torch.nn as nn
from utils import iou

class YoloLoss(nn.Module):
    def __init__(self, S = 5, B = 2, C = 4):
        super(YoloLoss, self).__init__()
        self.loss_criteria = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, prediction, target):
        #prediction(output of YOLO): batch_size * (S(5) * S(5) * C+ B * 5 (4+2*5))
        prediction = prediction.reshape(prediction.shape[0], self.S, self.S, self.C+self.B*5)

        iou_box1 = iou(prediction[..., 5:9], target[...,5:9])
        iou_box2 = iou(prediction[..., 10:14], target[...,5:9])
        ious = torch.cat([iou_box1.unsqueeze(0),iou_box2.unsqueeze(0)], dim=0)
        iou_max, best_box = torch.max(ious, dim=0)
        exist_box = target[...,4].unsqueeze(3)

        box_predictions = exist_box * ((1 - best_box) * prediction[..., 5:9] + best_box * prediction[..., 10:14])
        box_targets  = exist_box * target[..., 5:9]

        #take sqrt of width and height of bounding boxes
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4] + 1e-6))
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4] + 1e-6)

        box_loss = self.loss_criteria(torch.flatten(box_predictions, end_dim= -2), torch.flatten(box_targets, end_dim= -2))

        #loss for confidence score
        pred_box = exist_box * ((1-best_box) *  prediction[...,4:5] + best_box * prediction[...,9:10])
        object_loss = self.loss_criteria(torch.flatten(pred_box), torch.flatten(exist_box * target[...,4:5]))

        #loss for nonobject
        no_object_loss = self.loss_criteria( torch.flatten((1 - exist_box) * (prediction[...,4:5]), start_dim=1),
                                             torch.flatten((1 - exist_box) * target[...,4:5], start_dim=1))

        no_object_loss += self.loss_criteria( torch.flatten((1 - exist_box) * (prediction[...,9:10]), start_dim=1),
                                             torch.flatten((1 - exist_box) * target[...,4:5], start_dim=1))

        #loss for class

        class_loss = self.loss_criteria(torch.flatten(exist_box * prediction[...,:4], end_dim= -2),
                                        torch.flatten(exist_box * target[...,:4], end_dim= -2))

        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss

        return loss
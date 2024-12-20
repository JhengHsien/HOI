import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcherHOI(nn.Module):

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, cost_hoi_class: float = 1):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_hoi_class = cost_hoi_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs = len(outputs['pred_sub_boxes'])
        # num_queries = len([i for i in j for j outputs['pred_sub_boxes']])
        # print(bs, num_queries)
        # exit()
        bs_hoi_indices = []
        if 'pred_hoi_logits' in outputs.keys():
            for bs_idx, bs_logits in enumerate(outputs['pred_hoi_logits']):
                out_hoi_prob = bs_logits.sigmoid()
                # print(out_hoi_prob)
                tgt_hoi_labels = torch.cat([targets[bs_idx]['hoi_labels']])
                tgt_hoi_labels_permute = tgt_hoi_labels.permute(1, 0)
                cost_hoi_class = -(out_hoi_prob.matmul(tgt_hoi_labels_permute) / \
                                (tgt_hoi_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                                (1 - out_hoi_prob).matmul(1 - tgt_hoi_labels_permute) / \
                                ((1 - tgt_hoi_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
                cost_hoi_class = self.cost_hoi_class * cost_hoi_class

                C_hoi = self.cost_hoi_class * cost_hoi_class
                C_hoi = C_hoi.view(bs_logits.shape[0], -1).cpu()
                hoi_sizes = len(targets[bs_idx]['hoi_labels'])
                hoi_indices = [linear_sum_assignment(c) for i, c in enumerate(C_hoi.split(hoi_sizes,-1))]
                bs_hoi_indices.append(hoi_indices[0])
        # elif 'pred_verb_logits' in outputs.keys():
        #     out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
        #     tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
        #     tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
        #     cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
        #                         (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
        #                         (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
        #                         ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
        #     cost_hoi_class = self.cost_verb_class * cost_verb_class
        # else:
        #     cost_hoi_class = 0

        # tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
        # out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
        # cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
        # out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
        # out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)
        
        # bs_sub_indices = []
        # bs_obj_indices = []
        # for i in range(bs):
        #     out_sub_bbox = outputs['pred_sub_boxes'][i]
        #     out_obj_bbox = outputs['pred_obj_boxes'][i]

        #     tgt_sub_boxes = torch.cat([targets[i]['sub_boxes']])
        #     tgt_obj_boxes = torch.cat([targets[i]['obj_boxes']])

        # # if out_sub_bbox.dtype == torch.float16:
        # #     out_sub_bbox = out_sub_bbox.type(torch.float32)
        # #     out_obj_bbox = out_obj_bbox.type(torch.float32)

        #     cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)

        #     cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)

        #     if cost_sub_bbox.shape[1] == 0:
        #         cost_bbox = cost_sub_bbox
        #     else:
        #         cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]


        #     cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        #     cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
        #                     cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        #     if cost_sub_giou.shape[1] == 0:
        #         cost_giou = cost_sub_giou
        #     else:
        #         cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

        #     C_sub = self.cost_bbox * cost_bbox + \
        #         self.cost_giou * cost_sub_giou

        #     C_obj = self.cost_bbox * cost_obj_bbox + \
        #         self.cost_giou * cost_obj_giou

        #     # C = self.cost_bbox * cost_sub_bbox + \
        #     #     self.cost_giou * cost_giou
        

        #     C_sub = C_sub.view(1, cost_sub_bbox.shape[0], -1).cpu()
        #     C_obj = C_obj.view(1, cost_obj_bbox.shape[0], -1).cpu()
        #     # C = C.view(1, num_queries, -1).cpu()

        #     sub_sizes = [len(targets[i]["sub_boxes"])]
        #     obj_sizes = [len(targets[i]["obj_boxes"])]

        #     sub_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_sub.split(sub_sizes, -1))]
        #     obj_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_obj.split(obj_sizes, -1))]
        #     bs_sub_indices.append(sub_indices)
        #     bs_obj_indices.append(obj_indices)
        
        
        return bs_hoi_indices
        
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, cost_verb_class=args.set_cost_verb_class,
                               cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
                               cost_hoi_class=args.set_cost_hoi)

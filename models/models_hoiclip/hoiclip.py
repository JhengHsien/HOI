import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
from ModifiedCLIP import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label
from datasets.static_hico import HOI_IDX_TO_ACT_IDX, OBJ_IDX_TO_OBJ_NAME

from ..backbone import build_backbone
from ..matcher import build_matcher
from .gen import build_gen
from .dino import dino
# from .ca import CrossAttention
from .softmax_focal import SoftmaxFocalLoss
from datasets.static_hico import OBJ_IDX_TO_OBJ_NAME
import math


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


class HOICLIP(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        # self.transformer = transformer
        # hidden_dim = transformer.d_model
        # self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        # self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        # self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        # self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.inter2verb = MLP(args.clip_embed_dim, args.clip_embed_dim // 2, args.clip_embed_dim, 3)
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_model, self.preprocess = clip.load(self.args.clip_model)

        # Dino
        self.dino = dino()
        # Cross Attention
        self.MHA = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        # self.CrossAttention = CrossAttention
        self.projector = nn.Linear(512, 512)

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
            unseen_index = None

        clip_label, obj_clip_label, v_linear_proj_weight, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_CLIP(hoi_text_label, obj_text_label, unseen_index, args.no_clip_cls_init)
        num_obj_classes = len(obj_text) - 1  # del nothing
        self.clip_visual_proj = v_linear_proj_weight

        self.text_embedding = train_clip_label

        self.hoi_class_fc = nn.Sequential(
            nn.Linear(512, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim),
        )

        if unseen_index:
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
        else:
            unseen_index_list = []

        if self.args.dataset_file == 'hico':
            verb2hoi_proj = torch.zeros(117, 600)
            select_idx = list(set([i for i in range(600)]) - set(unseen_index_list))
            for idx, v in enumerate(HOI_IDX_TO_ACT_IDX):
                verb2hoi_proj[v][idx] = 1.0
            # self.verb2hoi_proj = nn.Parameter(verb2hoi_proj[:, select_idx], requires_grad=False)
            # self.verb2hoi_proj_eval = nn.Parameter(verb2hoi_proj, requires_grad=False)

            # self.verb_projection = nn.Linear(args.clip_embed_dim, 117, bias=False)
            # self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            # self.verb_weight = args.verb_weight
        else:
            verb2hoi_proj = torch.zeros(29, 263)
            for i in vcoco_hoi_text_label.keys():
                verb2hoi_proj[i[0]][i[1]] = 1

            # self.verb2hoi_proj = nn.Parameter(verb2hoi_proj, requires_grad=False)
            # self.verb_projection = nn.Linear(args.clip_embed_dim, 29, bias=False)
            # self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            # self.verb_weight = args.verb_weight

        # if args.with_clip_label:
        #     if args.fix_clip_label:
        #         self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text), bias=False)
        #         self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
        #         for i in self.visual_projection.parameters():
        #             i.require_grads = False
        #     else:
        #         self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text))
        #         self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)

        #     if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
        #         self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600, bias=False)
        #         self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        # else:
        #     self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))

        # if args.with_obj_clip_label:
        #     self.obj_class_fc = nn.Sequential(
        #         nn.Linear(hidden_dim, args.clip_embed_dim),
        #         nn.LayerNorm(args.clip_embed_dim),
        #     )
        #     if args.fix_clip_label:
        #         self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1, bias=False)
        #         self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        #         for i in self.obj_visual_projection.parameters():
        #             i.require_grads = False
        #     else:
        #         self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
        #         self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        # else:
        #     self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        # self.transformer.hoi_cls = clip_label / clip_label.norm(dim=-1, keepdim=True)

        # self.hidden_dim = hidden_dim
        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.uniform_(self.pos_guided_embedd.weight)

    def init_classifier_with_CLIP(self, hoi_text_label, obj_text_label, unseen_index, no_clip_cls_init=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_inputs = torch.cat([clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        text_inputs_del = torch.cat(
            [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
        clip_model = self.clip_model
        clip_model.to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_inputs.to(device))
            text_embedding_del = clip_model.encode_text(text_inputs_del.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            v_linear_proj_weight = clip_model.visual.proj.detach()

        if not no_clip_cls_init:
            print('\nuse clip text encoder to init classifier weight\n')
            return text_embedding.float(), obj_text_embedding.float(), v_linear_proj_weight.float(), \
                   hoi_text_label_del, obj_text_inputs, text_embedding_del.float()
        else:
            print('\nnot use clip text encoder to init classifier weight\n')
            return torch.randn_like(text_embedding.float()), torch.randn_like(
                obj_text_embedding.float()), torch.randn_like(v_linear_proj_weight.float()), \
                   hoi_text_label_del, obj_text_inputs, torch.randn_like(text_embedding_del.float())

    def forward(self, samples: NestedTensor, is_training=True, clip_input=None, targets=None, imgs_path=None, human_bboxes=None, obj_bboxes=None, original_imgs=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # (decoder_layer, bs, quries, 256), (decoder_layer, bs, quries, 256), (decoder_layer, bs, quries, 512), (bs, quries, 512), (bs, 1 , 600), (bs, 50, 512)
        

        ## train stage ##
        if (type(human_bboxes[0]) == type(None)):
            obj_classes=OBJ_IDX_TO_OBJ_NAME
            dino_human_bboxes, dino_object_boxes = self.dino.catch(obj_classes=obj_classes, imgs_path=imgs_path)
            human_bboxes = dino_human_bboxes
            obj_bboxes = dino_object_boxes

        human_features = []

        for idx, boxes in enumerate(human_bboxes):
            actor = []
            if (len(boxes) == 0):
                wanted_features = torch.zeros(1,512).to(device)
            else:
                for person_bbox in boxes:
                    w,h = original_imgs[idx].size
                    person_xyxy = person_bbox * torch.tensor([w, h, w, h], dtype=torch.float32).to(device)
                    person_xyxy = box_cxcywh_to_xyxy(person_xyxy)
                    x1, y1, x2, y2 = round(person_xyxy[0].item()), round(person_xyxy[1].item()), round(person_xyxy[2].item()), round(person_xyxy[3].item())
                    
                    # just in case therer are some invalid bboxes
                    if (x1 >= x2 or y1 >= y2):
                        x1 = 508
                        x2 = 599
                        y1 = 422
                        y2 = 517

                    wanted = original_imgs[idx].crop((x1, y1, x2, y2))
                    actor.append(self.preprocess(wanted))
                    
                wanted_features = torch.tensor(np.stack(actor)).to(device)
                wanted_features = self.clip_model.encode_image(wanted_features)[0].float()

            human_features.append(wanted_features)

        
        obj_features = []
        for idx, boxes in enumerate(obj_bboxes):
            objs = []
            if (len(boxes) == 0):
                wanted_features = torch.zeros(1,512).to(device)
            else:    
                for obj_bbox in boxes:
                    w,h = original_imgs[idx].size
                    obj_xyxy = obj_bbox * torch.tensor([w, h, w, h], dtype=torch.float32).to(device)
                    obj_xyxy = box_cxcywh_to_xyxy(obj_xyxy)
                    x1, y1, x2, y2 = round(obj_xyxy[0].item()), round(obj_xyxy[1].item()), round(obj_xyxy[2].item()), round(obj_xyxy[3].item())
                    
                    # just in case therer are some invalid bboxes
                    if (x1 >= x2 or y1 >= y2):
                        x1 = 33
                        x2 = 538
                        y1 = 47
                        y2 = 414
                    
                    wanted = original_imgs[idx].crop((x1, y1, x2, y2))
                    objs.append(self.preprocess(wanted))
                wanted_features = torch.tensor(np.stack(objs)).to(device)
                wanted_features = self.clip_model.encode_image(wanted_features)[0].float()

            obj_features.append(wanted_features)
    

        ############################################

        # ##  test stage ##
        # if args.eval:
        #     human_bboxes, obj_bboxes = self.dino.catch(obj_classes=OBJ_IDX_TO_OBJ_NAME, imgs_path=imgs_path,)

        # ##################


        ######### Interaction features ##########
        # Strategy: CA<Person, object> then CA<PO, Scene>, dimention should be same as inter_hs
        # By without queries, we should have not followed <quries, dimention> rule

        # should consider the case about: 
        # if person are zero
        # if obj are zero

        pair_match = []
        scene = self.clip_model.encode_image(clip_input)[0].float()
        person_objects = []
        logits = []
        for idx, people in enumerate(human_features): # 4, 3
            if (len(people) == 0 or len(obj_features[idx]) == 0):
                logit.append(torch.zeros((1, 512), dtype=torch.float32))
            else:
                
                all_pair = []
                
                people_nums = people.shape[0]
                objects_nums = obj_features[idx].shape[0]

                all_pair.append(torch.range(0,objects_nums))

                po_gcd = math.lcm(people_nums, objects_nums)

                human = people.repeat(int(po_gcd/people_nums), 1)
                objects = obj_features[idx].repeat(int(po_gcd/objects_nums), 1)

                po_features = self.MHA(human, objects, objects)[0][:people_nums]
                po_features = torch.div(po_features, 2)
                po_features = torch.add(po_features, torch.div(scene[idx].repeat(po_features.shape[0],1),2))
                logits.append(po_features)
            pair_match.append(all_pair)

                # Do classify

        # So it is like there are <numbers> X <numbers> interaction features depends on how many people and how many objects in this frame 
        #########################################

        ####### Do object class fc #########
        # obj_logits = []
        # for each_object in obj_features:
        #     obj_logits.append(self.obj_logit_scale.exp() * self.obj_visual_projection(each_object))
        ####################################

        # ####### Do hoi class fc ###########


        action_logits = []
        text_embedding = self.text_embedding / self.text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = self.projector(text_embedding)
        for each_frame in logits:
            frame_eatures = self.projector(each_frame)
            action_logits.append(frame_eatures @ text_embedding.t())
        torch.cuda.empty_cache()
        # ####################################

        # if self.args.with_obj_clip_label:
        #     obj_logit_scale = self.obj_logit_scale.exp()
        #     o_hs = self.obj_class_fc(o_hs)
        #     o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
        #     outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
        # else:
        #     outputs_obj_class = self.obj_class_embed(o_hs)

        # if self.args.with_clip_label:
        #     logit_scale = self.logit_scale.exp()
        #     # inter_hs = self.hoi_class_fc(inter_hs)
        #     outputs_inter_hs = inter_hs.clone()
        #     verb_hs = self.inter2verb(inter_hs)
        #     inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
        #     verb_hs = verb_hs / verb_hs.norm(dim=-1, keepdim=True)
        #     if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' \
        #             and (self.args.eval or not is_training):
        #         outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
        #         outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj_eval
        #         outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
        #     else:
        #         outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
        #         outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj
        #         outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
        # else:
        #     inter_hs = self.hoi_class_fc(inter_hs)
        #     outputs_inter_hs = inter_hs.clone()
        #     outputs_hoi_class = self.hoi_class_embedding(inter_hs)
        
        # out = {'pred_hoi_logits': action_logits, 'pred_obj_logits': outputs_obj_class[-1],
        #        'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 'clip_visual': clip_visual,
        #        'clip_cls_feature': clip_cls_feature, 'hoi_feature': inter_hs[-1], 'clip_logits': clip_hoi_score}
        out = {'pred_hoi_logits': action_logits,
                'pred_pair_id': pair_match,
                # 'pred_obj_logits': obj_logits,
                'pred_obj_boxes': human_bboxes,
                'pred_sub_boxes': obj_bboxes,
               'hoi_feature': logits}       

        # if self.args.with_mimic:
        #     out['inter_memory'] = outputs_inter_hs[-1]
        # if self.aux_loss:
        #     if self.args.with_mimic:
        #         aux_mimic = outputs_inter_hs
        #     else:
        #         aux_mimic = None

        #     # out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_class,
        #     #                                                 outputs_sub_coord, outputs_obj_coord,
        #     #                                                 aux_mimic)
        #     out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, aux_mimic)                                                

        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord, outputs_inter_hs=None):

        if outputs_hoi_class.shape[0] == 1:
            outputs_hoi_class = outputs_hoi_class.repeat(self.dec_layers, 1, 1, 1)
        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                    #    'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                    #    'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                    #    'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1],
                       }
        if outputs_inter_hs is not None:
            aux_outputs['inter_memory'] = outputs_inter_hs[-self.dec_layers: -1]
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.with_mimic:
            self.clip_model, _ = clip.load(args.clip_model, device=device)
        else:
            self.clip_model = None
        self.alpha = args.alpha

        # softmax focal loss
        self.softmax_focal = SoftmaxFocalLoss()

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        # assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], 0,
        #                             dtype=torch.int64, device=src_logits.device)

        for tid, each_target in enumerate(targets):
            obj_labels_gt = each_target['obj_labels']
            pre = _sigmoid(src_logits[tid])
            pre = pre.topk(len(obj_labels_gt), 1, True, True)[0].float()
            loss_obj_ce = F.cross_entropy(pre, obj_labels_gt, self.empty_weight)


        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, topk=5):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits'] # I have N + M predicts
        # print(targets[0]['hoi_labels'].shape) # means there are 4 labels
        # dtype = src_logits.dtype
        # idx = self._get_src_permutation_idx(indices)

        # for each batch:
        # for i in src_logits:
        acc = 0.0
        loss = []
        for tid, each_target in enumerate(targets):
            # collect target_numbers perdictions
            acc_pred = 0.0
            # print(len(each_target["sub_boxes"]), len(each_target["obj_boxes"]))
            # print(torch.where(each_target['hoi_labels'] == 1))
            # exit()
            tgt_idx = torch.where(each_target['hoi_labels'] == 1)[1]
            # print(each_target["sub_boxes"], each_target["obj_boxes"])
            # print(tid, tgt_idx)
            # print(src_logits[tid].shape)

            gt_num = len(tgt_idx) # for each one sub which has gt_num interaction

            batch_pred = []
            for indices_id in indices[tid][0]:
                pred = src_logits[tid][indices_id]
                # acc loss
                acc_pre = torch.nn.functional.softmax(src_logits[tid][indices_id], dim=-1).topk(gt_num, 0, True, True)[1]
                # print(acc_pre, tgt_idx)
                
                for tgt_rel in tgt_idx:
                    acc_pred += (tgt_rel in acc_pre)
                
                batch_pred.append(pred)
            
            if (len(batch_pred) == 0): 
                batch_pred.append(torch.zeros(each_target['hoi_labels'].shape[1]))
                batch_pred = torch.stack(batch_pred)
                target_class_o = torch.zeros((1,each_target['hoi_labels'].shape[1]))
                loss_hoi_ce = self.softmax_focal(batch_pred, target_class_o, weights=None, gamma=self.alpha)
            else:
                batch_pred = torch.stack(batch_pred).to(device)
                loss_hoi_ce = self.softmax_focal(batch_pred, each_target['hoi_labels'], weights=None, gamma=self.alpha)
            # loss_hoi_ce = self._neg_loss(batch_pred, each_target['hoi_labels'], weights=None, alpha=self.alpha)
            loss.append(loss_hoi_ce)

            # print(acc_pred, len(tgt_idx))
            rel_labels_error = 100 - 100 * acc_pred / max(len(tgt_idx), 1)
            acc += rel_labels_error

        loss = torch.tensor(sum(loss)).to(device)
        losses = {'loss_hoi_labels': loss}
        
        # losses['hoi_class_error'] = torch.from_numpy(np.array(
        #     rel_labels_error)).to(device).float()
        
            # acc_pred = 0.0
            # for tgt_rel in tgt_idx:
            #     acc_pred += (tgt_rel in pre)
            # acc += acc_pred / len(tgt_idx)

        
        # rel_labels_error = 100 - 100 * acc / max(len(targets), 1)
        # losses = {}
        # losses['loss_hoi_labels'] = torch.from_numpy(np.array(
        #     rel_labels_error)).to(device).float()

        # target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)]).to(dtype)
        # target_classes = torch.zeros_like(src_logits)
        # target_classes[idx] = target_classes_o
        # src_logits = _sigmoid(src_logits)
        
        # loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        # losses = {'loss_hoi_labels': loss_hoi_ce}

        # print(topk)

        # _, pred = src_logits[idx].topk(topk, 1, True, True)

        # acc = 0.0
        # for tid, target in enumerate(target_classes_o):
        #     tgt_idx = torch.where(target == 1)[0]
        #     if len(tgt_idx) == 0:
        #         continue
        #     acc_pred = 0.0
        #     for tgt_rel in tgt_idx:
        #         print(tgt_rel)
        #         print(pred[tid])
        #         exit()
        #         acc_pred += (tgt_rel in pred[tid])
        #     acc += acc_pred / len(tgt_idx)
        # rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        # losses['hoi_class_error'] = torch.from_numpy(np.array(
        #     rel_labels_error)).to(src_logits.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def mimic_loss(self, outputs, targets, indices, num_interactions):
        src_feats = outputs['inter_memory']
        src_feats = torch.mean(src_feats, dim=1)

        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
        loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
        losses = {'loss_feat_mimic': loss_feat_mimic}
        return losses
    def reconstruction_loss(self, outputs, targets, indices, num_interactions):
        raw_feature = outputs['clip_cls_feature']
        hoi_feature = outputs['hoi_feature']

        loss_rec = F.l1_loss(raw_feature, hoi_feature)
        return {'loss_rec': loss_rec}

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                # 'obj_labels': self.loss_obj_labels,
                # 'sub_obj_boxes': self.loss_sub_obj_boxes,
                # 'feats_mimic': self.mimic_loss,
                # 'obj_labels': 0,
                # 'sub_obj_boxes': 0,
                # 'feats_mimic': 0,
                # 'rec_loss': self.reconstruction_loss
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # indices = [0,0]

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        # num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
        #                                    device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_interactions)
        # num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
   
                for loss in self.losses:
                    kwargs = {}
                    if loss =='rec_loss':
                        continue
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        # out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']
        pair = outputs['pred_pair_id']
        # clip_visual = outputs['clip_visual']
        # clip_logits = outputs['clip_logits']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits
        # obj_scores = out_obj_logits.sigmoid()
        # obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores[0].device)
        # sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        # sub_boxes = sub_boxes * scale_fct[:, None, :]
        # obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        # obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            # hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
            #     index]
            hs = hoi_scores[index]
            sb = out_sub_boxes[index]
            ob = out_obj_boxes[index]
            # sl = torch.full_like(ol, self.subject_category_id)
            # l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'boxes': b.to('cpu')})
            # results.append({'labels': l.to('cpu')})

            ids = torch.arange(b.shape[0])

            # results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'), 'clip_visual': clip_visual[index].to('cpu'),
            #                     'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:], 'clip_logits': clip_logits[index].to('cpu')})
            results[-1].update({'hoi_scores': hs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})
        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_gen(args)

    model = HOICLIP(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    if args.with_clip_label:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.with_mimic:
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef

    if args.with_rec_loss:
        weight_dict['loss_rec'] = args.rec_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['hoi_labels']
    # if args.with_mimic:
    #     losses.append('feats_mimic')

    # if args.with_rec_loss:
    #     losses.append('rec_loss')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors

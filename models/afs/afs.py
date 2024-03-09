import torch
import torch.nn as nn
from tqdm import tqdm
from utils.misc_helper import to_device
import torch.nn.functional as F
import torch.distributed as dist
import copy


class AFS(nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 structure,
                 init_bsn,
                 ):

        super(AFS, self).__init__()

        self.inplanes=inplanes
        self.instrides=instrides
        self.structure=structure
        self.init_bsn=init_bsn

        self.indexes=nn.ParameterDict()

        for block in self.structure:
            for layer in block['layers']:
                self.indexes["{}_{}".format(block['name'],layer['idx'])]=nn.Parameter(torch.zeros(layer['planes']).long(),requires_grad=False)
                self.add_module("{}_{}_upsample".format(block['name'],layer['idx']),
                                nn.UpsamplingBilinear2d(scale_factor=self.instrides[layer['idx']]/block['stride']))


    @torch.no_grad()
    def forward(self, inputs,train=False):
        block_feats = {}
        feats = inputs["feats"]
        for block in self.structure:
            block_feats[block['name']]=[]

            for layer in block['layers']:
                feat_c=torch.index_select(feats[layer['idx']]['feat'], 1, self.indexes["{}_{}".format(block['name'],layer['idx'])].data)
                feat_c=getattr(self,"{}_{}_upsample".format(block['name'],layer['idx']))(feat_c)
                block_feats[block['name']].append(feat_c)
            block_feats[block['name']]=torch.cat(block_feats[block['name']],dim=1)

        if train:
            gt_block_feats = {}
            gt_feats = inputs["gt_feats"]
            for block in self.structure:
                gt_block_feats[block['name']] = []
                for layer in block['layers']:
                    feat_c = torch.index_select(gt_feats[layer['idx']]['feat'], 1, self.indexes["{}_{}".format(block['name'], layer['idx'])].data)
                    feat_c = getattr(self, "{}_{}_upsample".format(block['name'], layer['idx']))(feat_c)
                    gt_block_feats[block['name']].append(feat_c)
                gt_block_feats[block['name']] = torch.cat(gt_block_feats[block['name']], dim=1)
            return {'block_feats':block_feats,"gt_block_feats":gt_block_feats}

        return {'block_feats':block_feats}



    def get_outplanes(self):
        return { block['name']:sum([layer['planes'] for layer in block['layers']])  for block in self.structure}

    def get_outstrides(self):
        return { block['name']:block['stride']  for block in self.structure}


    @torch.no_grad()
    def init_idxs(self, model, train_loader, distributed=True):
        anomaly_types = copy.deepcopy(train_loader.dataset.anomaly_types)

        if 'normal' in train_loader.dataset.anomaly_types:
            del train_loader.dataset.anomaly_types['normal']

        for key in train_loader.dataset.anomaly_types:
            train_loader.dataset.anomaly_types[key] = 1.0/len(list(train_loader.dataset.anomaly_types.keys()))

        model.eval()
        criterion = nn.MSELoss(reduce=False).to(model.device)
        for block in self.structure:
            self.init_block_idxs(block, model, train_loader, criterion,distributed=distributed)
        train_loader.dataset.anomaly_types = anomaly_types
        model.train()


    def init_block_idxs(self,block,model,train_loader,criterion,distributed=True):

        if distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            if rank == 0:
                tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))
            else:
                tq = range(self.init_bsn)
        else:
            tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))

        cri_sum_vec=[torch.zeros(self.inplanes[layer['idx']]).to(model.device) for layer in block['layers']]
        iterator = iter(train_loader)

        for bs_i in tq:
            try:
                input = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                input = next(iterator)

            bb_feats = model.backbone(to_device(input),train=True)

            ano_feats=bb_feats['feats']
            ori_feats=bb_feats['gt_feats']
            gt_mask = input['mask'].to(model.device)

            B= gt_mask.size(0)

            ori_layer_feats=[ori_feats[layer['idx']]['feat'] for layer in block['layers']]
            ano_layer_feats=[ano_feats[layer['idx']]['feat'] for layer in block['layers']]

            for i,(ano_layer_feat,ori_layer_feat) in enumerate(zip(ano_layer_feats,ori_layer_feats)):
                layer_name=block['layers'][i]['idx']

                C = ano_layer_feat.size(1)

                ano_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ano_layer_feat)
                ori_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ori_layer_feat)

                layer_pred = (ano_layer_feat - ori_layer_feat) ** 2

                _, _, H, W = layer_pred.size()

                layer_pred = layer_pred.permute(1, 0, 2, 3).contiguous().view(C, B * H * W)
                (min_v, _), (max_v, _) = torch.min(layer_pred, dim=1), torch.max(layer_pred, dim=1)
                layer_pred = (layer_pred - min_v.unsqueeze(1)) / (max_v.unsqueeze(1) - min_v.unsqueeze(1)+ 1e-4)

                label = F.interpolate(gt_mask, (H, W), mode='nearest')
                label = label.permute(1, 0, 2, 3).contiguous().view(1, B * H * W).repeat(C, 1)

                mse_loss = torch.mean(criterion(layer_pred, label), dim=1)

                if distributed:
                    mse_loss_list = [mse_loss for _ in range(world_size)]
                    dist.all_gather(mse_loss_list, mse_loss)
                    mse_loss = torch.mean(torch.stack(mse_loss_list,dim=0),dim=0,keepdim=False)

                cri_sum_vec[i] += mse_loss

        for i in range(len(cri_sum_vec)):
            cri_sum_vec[i][torch.isnan(cri_sum_vec[i])] = torch.max(cri_sum_vec[i][~torch.isnan(cri_sum_vec[i])])
            values, indices = torch.topk(cri_sum_vec[i], k=block['layers'][i]['planes'], dim=-1, largest=False)
            values, _ = torch.sort(indices)

            if distributed:
                tensor_list = [values for _ in range(world_size)]
                dist.all_gather(tensor_list, values)
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(tensor_list[0].long())
            else:
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(values.long())

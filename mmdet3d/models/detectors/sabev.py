import torch
import torch.nn.functional as F
import numpy as np
from mmcv.runner import force_fp32

from mmdet.models import DETECTORS
from .centerpoint import CenterPoint
from .bevdet import BEVDepth4D
from torch.cuda.amp import autocast
from ...datasets.pipelines.loading import LoadAnnotationsBEVDepth
from ..necks.litemono import LiteMono
from ..necks.litemono_decoder import DepthDecoder, PoseDecoder
from ..necks.layers import *

@DETECTORS.register_module()
class SABEV(BEVDepth4D):

    def __init__(self, use_bev_paste=True, bda_aug_conf=None, **kwargs):
        super(SABEV, self).__init__(**kwargs)
        # Use paste or not 
        self.use_bev_paste = use_bev_paste
        if use_bev_paste:
            self.loader = LoadAnnotationsBEVDepth(bda_aug_conf, None, is_train=True)

        self.depth_encoder = LiteMono()
        self.depth_decoder = DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc,
                                          scales=[0, 1, 2])
        self.mask_decoder = DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc,
                                          scales=[0, 1, 2],
                                          num_output_channels=(len([0, 1]))-1)
        self.pose_decoder = PoseDecoder(self.depth_encoder.num_ch_enc, 2)
        self.backproject_depth = {}
        self.project_3d = {}

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input, paste_idx, bda_paste, img_metas=None):
        x = self.image_encoder(img)
        bev_feat, img_preds = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input, paste_idx, bda_paste])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, img_preds

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = list(self.img_backbone(imgs))
        if self.with_img_neck:
            x[1] = self.img_neck(x[1:])
            if type(x) in [list, tuple]:
                x = x[:2]
        for i in range(2):
            _, output_dim, ouput_H, output_W = x[i].shape
            x[i] = x[i].view(B, N, output_dim, ouput_H, output_W)
        return x[:2][::-1]

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        # Inputs[0]: img data
        B, N, _, H, W = inputs[0].shape
        # Sabev self.num_frame == 2
        N = N // self.num_frame
        # 此时N为相机数
        imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
        # imgs[0] (B, N, 2, 3, H, W)   imgs[1] (B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        # imgs[0] (B, N, 1, 3, H, W)   imgs[1] (B, N, 1, 3, H, W) 
        # 把两帧图像分开来
        imgs = [t.squeeze(2) for t in imgs]
        # imgs[0] (B, N, 3, H, W)   imgs[1] (B, N, 3, H, W) 
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.view(B, self.num_frame, N, 3, 3),
            trans.view(B, self.num_frame, N, 3),
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        # 把两帧的其他参数也分开来
        # rots [[B, N, 3, 3], [B, N, 3, 3]] 其它同理
        rots, trans, intrins, post_rots, post_trans = extra
        if len(inputs) == 9:
            paste_idx = inputs[7]
            bda_paste = inputs[8]
        else:
            paste_idx = None
            bda_paste = None
        return imgs, rots, trans, intrins, post_rots, post_trans, bda, paste_idx, bda_paste

    def extract_img_feat(self, img, img_metas, **kwargs):

        imgs, rots, trans, intrins, post_rots, post_trans, bda, paste_idx, bda_paste = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        key_frame = True  # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, post_tran in zip(
                imgs, rots, trans, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input, paste_idx, bda_paste)
                if key_frame:
                    bev_feat, img_preds = self.prepare_bev_feat(*inputs_curr, img_metas)
                else:
                    with torch.no_grad():
                    # 注意，不是关键(当前)帧不需要梯度
                        bev_feat, _ = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
            bev_feat_list.append(bev_feat)
            key_frame = False

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], img_preds

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, img_preds = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, img_preds)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # Paste Method 
        if self.use_bev_paste:
            B = len(gt_bboxes_3d)
            paste_idx = []
            for i in range(B):
                for j in range(i, i + 1):
                    if j+1>=B: j-=B
                    paste_idx.append([i,j+1])
            # 作者想将每前后两帧的索引放到一起，注意最后一帧和第0帧索引放到一起
            # 例：B=8  paste_idx=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]]
            
            gt_boxes_paste = []
            gt_labels_paste = []
            bda_mat_paste = []
            for i in range(len(paste_idx)):
                gt_boxes_tmp = []
                gt_labels_tmp = []
                for j in paste_idx[i]:
                    gt_boxes_tmp.append(gt_bboxes_3d[j])
                    gt_labels_tmp.append(gt_labels_3d[j])
                # Two frame
                gt_boxes_tmp = torch.cat([tmp.tensor for tmp in gt_boxes_tmp], dim=0)
                gt_labels_tmp = torch.cat(gt_labels_tmp, dim=0)
                # rotate_bda为旋转角
                # 返回数据Augmentation后的gt_boxes，以及数据Augmentation所使用的matrix 
                rotate_bda, scale_bda, flip_dx, flip_dy = self.loader.sample_bda_augmentation()
                gt_boxes_tmp, bda_rot = self.loader.bev_transform(gt_boxes_tmp.cpu(), rotate_bda, scale_bda, flip_dx, flip_dy)

                gt_boxes_tmp = gt_bboxes_3d[0].new_box(gt_boxes_tmp.cuda())

                bda_mat_paste.append(bda_rot.cuda())
                gt_boxes_paste.append(gt_boxes_tmp)
                gt_labels_paste.append(gt_labels_tmp)
            
            gt_bboxes_3d = gt_boxes_paste
            gt_labels_3d = gt_labels_paste
            img_inputs.append(paste_idx)
            img_inputs.append(torch.stack(bda_mat_paste))

        B, N, C, H, W = img_inputs[0].shape
        for scale in [0, 1, 2]:
            h = H // (2 ** scale)
            w = W // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(B, h, w)
            self.project_3d[scale] = Project3D(B, h, w)


        # TODO: Try to use unsupervised method to predict depth
        all_features = self.depth_encoder(img_inputs[0].view(B*N, C, H, W))
        # img_inputs[0].shape torch.Size([48(2*N*B), 3, 256, 704])
        # x[0].shape torch.Size([48(2*N*B), 48, 64, 176])
        # x[1].shape torch.Size([48, 80, 32, 88])
        # x[2].shape torch.Size([48, 128, 16, 44])
        all_features = [torch.split(f, B*N//2, dim=0) for f in all_features]
        # all_features [((B*N//2, 48, 64, 176), (B*N//2, 48, 64, 176)), 
        #               ((B*N//2, 80, 32, 88), (B*N//2, 80, 32, 88)),
        #               ((B*N//2, 128, 16, 44), (B*N//2, 128, 16, 44))]
        


        # TODO: Change author original codes
        img_feats, pts_feats, img_preds = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        gt_semantic = kwargs['gt_semantic']
        # gt_depth torch.Size([B, N, 256, 704])
        # gt_semantic torch.Size([B, N, 256, 704])



        depth_features = {}
        for i, k in enumerate([0, 1]):
            depth_features[k] = [f[i] for f in all_features]
        # depth_features[0] [(B*N//2, 48, 64, 176), (B*N//2, 80, 32, 88), (B*N//2, 128, 16, 44)]
        # depth_features[1] [(B*N//2, 48, 64, 176), (B*N//2, 80, 32, 88), (B*N//2, 128, 16, 44)]
        outputs = self.depth_decoder(depth_features[0])
        # outputs[('disp', 0)].shape torch.Size([B*N, 1, 256, 704])
        # outputs[('disp', 1)].shape torch.Size([B*N, 1, 128, 352])
        # outputs[('disp', 2)].shape torch.Size([B*N, 1, 64, 176])
        outputs["predictive_mask"] = self.mask_decoder(depth_features[0])
        # outputs["predictive_mask"][('disp', 0)].shape torch.Size([B*N, 1, 256, 704])
        # outputs["predictive_mask"][('disp', 1)].shape torch.Size([B*N, 1, 128, 352])
        # outputs["predictive_mask"][('disp', 2)].shape torch.Size([B*N, 1, 64, 176])
        outputs.update(self.predict_poses(img_inputs[0].view(B*N, C, H, W), depth_features))
        # output.keys()
        # dict_keys([('disp', 2), ('disp', 1), ('disp', 0), 'predictive_mask',
        #            ('axisangle', 0, 1), ('translation', 0, 1), ('cam_T_cam', 0, 1)])
        # outputs[('axisangle', 0, 1)].shape torch.Size([B*N, 1, 1, 3])
        # outputs[('translation', 0, 1)].shape torch.Size([B*N, 1, 1, 3])
        # outputs[('cam_T_cam', 0, 1)].shape torch.Size([B*N, 4, 4])
        K = img_inputs[3].view(B, self.num_frame, N, 3, 3)
        inv_k = torch.inverse(K)
        # K.shape torch.Size([B, self.num_frame, N, 3, 3])
        # inv_k.shape torch.Size([B, self.num_frame, N, 3, 3])
        
        breakpoint()

        self.generate_images_pred(img_inputs[0].view(B*N, C, H, W), outputs)
        #losses = self.compute_losses(img_inputs[0].view(B*N, C, H, W), outputs)
        


        loss_depth, loss_semantic = \
            self.img_view_transformer.get_loss(img_preds, gt_depth, gt_semantic)
        losses = dict(loss_depth=loss_depth, loss_semantic=loss_semantic)
        with autocast(False):
            losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    


    '''Depth Predict Codes'''
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.
        # select what features the pose network takes as input

        pose_feats = {f_i: features[f_i] for f_i in [0, 1]}
        pose_inputs = [pose_feats[1], pose_feats[0]]

        axisangle, translation = self.pose_decoder(pose_inputs)
        outputs[("axisangle", 0, 1)] = axisangle
        outputs[("translation", 0, 1)] = translation

        # Invert the matrix if the frame id is negative
        outputs[("cam_T_cam", 0, 1)] = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=True)
        return outputs


    def generate_images_pred(self, inputs, outputs, K, inv_K):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in [0, 1, 2]:
            disp = outputs[("disp", scale)]

            _, depth = disp_to_depth(disp, 1.0, 60.0)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate([1]):
                
                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[0](depth, inv_K)
                pix_coords = self.project_3d[0](cam_points, K, T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, 0)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                """
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, 0)]
                """

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in [0, 1, 2]:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses


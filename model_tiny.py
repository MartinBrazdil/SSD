from collections import OrderedDict

import torch
from torch import nn
from detection_tutorial.utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import tee
import torchvision
from torchvision.models.vgg import VGG, vgg16, make_layers, cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class BasicBlock(nn.Module):
    def __init__(self, idx, cfg):
        self.layers = OrderedDict()

        def id(name, i=None):
            if i is None:
                return '{}{}'.format(name, idx)
            return '{}{}_{}'.format(name, idx, i)

        for i, (c0, c1) in enumerate(pairwise(cfg)):
            self.layers[id('conv', i)] = nn.Conv2d(c0, c1, kernel_size=3, padding=1, bias=False)
            self.layers[id('bn', i)] = nn.BatchNorm2d(c1)
        self.layers[id('pool')] = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        out = input
        for name, layer in self.layers.items():
            if 'conv' in name:
                out = layer(out)
            if 'bn' in name:
                out = layer(out)
                out = F.relu(out)
            if 'pool' in name:
                out = layer(out)
        return out


class TinyNet(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    #      0   1   2   3   4   5   6   7    8    9    10   11   12
    cfg = [32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64]
    # cfg = [16, 16, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64]

    def __init__(self):
        super(TinyNet, self).__init__()

        model = nn.Sequential(OrderedDict())

        model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 20, 5)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20, 64, 5)),
            ('relu2', nn.ReLU())
        ]))

        self.layers = OrderedDict()

        self.layers.update(BasicBlock(0, [3, 32, 32]).layers)
        # self.conv1_1 = nn.Conv2d(3, self.cfg[0], kernel_size=3, padding=1, bias=False)  # stride = 1, by default
        # self.bn1_1 = nn.BatchNorm2d(self.cfg[0])
        # self.conv1_2 = nn.Conv2d(self.cfg[0], self.cfg[1], kernel_size=3, padding=1, bias=False)
        # self.bn1_2 = nn.BatchNorm2d(self.cfg[1])
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layers.update(BasicBlock(1, [32, 32, 32]).layers)
        # self.conv2_1 = nn.Conv2d(self.cfg[1], self.cfg[2], kernel_size=3, padding=1, bias=False)
        # self.bn2_1 = nn.BatchNorm2d(self.cfg[2])
        # self.conv2_2 = nn.Conv2d(self.cfg[2], self.cfg[3], kernel_size=3, padding=1, bias=False)
        # self.bn2_2 = nn.BatchNorm2d(self.cfg[3])
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layers.update(BasicBlock(2, [32, 32, 32, 32]).layers)
        # self.conv3_1 = nn.Conv2d(self.cfg[3], self.cfg[4], kernel_size=3, padding=1, bias=False)
        # self.bn3_1 = nn.BatchNorm2d(self.cfg[4])
        # self.conv3_2 = nn.Conv2d(self.cfg[4], self.cfg[5], kernel_size=3, padding=1, bias=False)
        # self.bn3_2 = nn.BatchNorm2d(self.cfg[5])
        # self.conv3_3 = nn.Conv2d(self.cfg[5], self.cfg[6], kernel_size=3, padding=1, bias=False)
        # self.bn3_3 = nn.BatchNorm2d(self.cfg[6])
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.blocks.append(BasicBlock(3, [32, 64, 64, 64]).layers)
        # self.conv4_1 = nn.Conv2d(self.cfg[6], self.cfg[7], kernel_size=3, padding=1, bias=False)
        # self.bn4_1 = nn.BatchNorm2d(self.cfg[7])
        # self.conv4_2 = nn.Conv2d(self.cfg[7], self.cfg[8], kernel_size=3, padding=1, bias=False)
        # self.bn4_2 = nn.BatchNorm2d(self.cfg[8])
        # self.conv4_3 = nn.Conv2d(self.cfg[8], self.cfg[9], kernel_size=3, padding=1, bias=False)
        # self.bn4_3 = nn.BatchNorm2d(self.cfg[9])
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.blocks.append(BasicBlock(4, [64, 64, 64, 64]).layers)
        # self.conv5_1 = nn.Conv2d(self.cfg[9], self.cfg[10], kernel_size=3, padding=1, bias=False)
        # self.bn5_1 = nn.BatchNorm2d(self.cfg[10])
        # self.conv5_2 = nn.Conv2d(self.cfg[10], self.cfg[11], kernel_size=3, padding=1, bias=False)
        # self.bn5_2 = nn.BatchNorm2d(self.cfg[11])
        # self.conv5_3 = nn.Conv2d(self.cfg[11], self.cfg[12], kernel_size=3, padding=1, bias=False)
        # self.bn5_3 = nn.BatchNorm2d(self.cfg[12])
        # self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """


        out = F.relu(self.bn1_1(self.conv1_1(image)))  # (N, 64, 300, 300)
        out = F.relu(self.bn1_2(self.conv1_2(out)))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.bn2_1(self.conv2_1(out)))  # (N, 128, 150, 150)
        out = F.relu(self.bn2_2(self.conv2_2(out)))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.bn3_1(self.conv3_1(out)))  # (N, 256, 75, 75)
        out = F.relu(self.bn3_2(self.conv3_2(out)))  # (N, 256, 75, 75)
        out = F.relu(self.bn3_3(self.conv3_3(out)))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.bn4_1(self.conv4_1(out)))  # (N, 512, 38, 38)
        out = F.relu(self.bn4_2(self.conv4_2(out)))  # (N, 512, 38, 38)
        out = F.relu(self.bn4_3(self.conv4_3(out)))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.bn5_1(self.conv5_1(out)))  # (N, 512, 19, 19)
        out = F.relu(self.bn5_2(self.conv5_2(out)))  # (N, 512, 19, 19)
        out = F.relu(self.bn5_3(self.conv5_3(out)))  # (N, 512, 19, 19)
        conv5_3_feats = out

        # Lower-level feature maps
        return conv4_3_feats, conv5_3_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map

        n_boxes = {
            'conv4_3': 4,
            'conv5_3': 6
        }
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(TinyNet.cfg[9], n_boxes['conv4_3'] * 4, kernel_size=3, padding=1, bias=False)
        self.loc_conv5_3 = nn.Conv2d(TinyNet.cfg[12], n_boxes['conv5_3'] * 4, kernel_size=3, padding=1, bias=False)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(TinyNet.cfg[9], n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1, bias=False)
        self.cl_conv5_3 = nn.Conv2d(TinyNet.cfg[12], n_boxes['conv5_3'] * n_classes, kernel_size=3, padding=1, bias=False)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.kaiming_uniform_(c.weight)

    def forward(self, conv4_3_feats, conv5_3_feats):
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv5_3_feats: conv5_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv5_3 = self.loc_conv5_3(conv5_3_feats)  # (N, 16, 38, 38)
        l_conv5_3 = l_conv5_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv5_3 = l_conv5_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map


        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv5_3 = self.cl_conv5_3(conv5_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv5_3 = c_conv5_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv5_3 = c_conv5_3.view(batch_size, -1, self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map


        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv5_3], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv5_3], dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


class TinySSD(nn.Module):
    """
    The TinySSD network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes):
        super(TinySSD, self).__init__()

        self.n_classes = n_classes

        self.base = TinyNet()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        # self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 256, 1, 1))  # there are 512 channels in conv4_3_feats
        # nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv5_3_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv5_3_feats)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the TinySSD, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {
            'conv4_3': 38,
            'conv5_3': 19}

        obj_scales = {
            'conv4_3': 0.1,
            'conv5_3': 0.2}

        aspect_ratios = {
            'conv4_3': [1., 2., 0.5],
            'conv5_3': [1., 2., 3., 0.5, .333]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths TinySSD) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            # true_classes[i] = label_for_each_prior
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss


class MyModule(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        modules = nn.ModuleList()
        modules.append(MyModule.conv_block([in_c, 32, 32]))
        modules.append(MyModule.conv_block([32, 32, 32]))
        modules.append(MyModule.conv_block([32, 32, 32, 32]))
        modules.append(MyModule.conv_block([32, 64, 64, 64]))
        modules.append(MyModule.conv_block([64, 64, 64, 64]))
        self.model = nn.Sequential(*modules)

    @staticmethod
    def conv_block(cfg):
        seq = nn.Sequential()
        for i, (c0, c1) in enumerate(pairwise(cfg)):
            seq.add_module('conv' + str(i), nn.Conv2d(c0, c1, kernel_size=3, padding=1, bias=False))
            seq.add_module('bn' + str(i), nn.BatchNorm2d(c1))
        seq.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))
        return seq

    def forward(self, img):
        return self.model.forward(img)

if __name__ == '__main__':
    img = torch.ones([1, 3, 300, 300])
    # block = BasicBlock(0, [3, 32, 32])
    # block.forward(input)

    MyModule(3).forward(img)

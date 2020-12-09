from DLAnet import DlaNet
import torch
import cv2
import numpy as np
from utils import *
from dataset import ctDataset
import os
from torch.utils.data import DataLoader

class Predictor:
    def __init__(self, use_gpu):
        # Todo: the mean and std need to be modified according to our dataset
        # self.mean_ = np.array([0.5194416012442385, 0.5378052387430711, 0.533462090585746], \
        #                 dtype=np.float32).reshape(1, 1, 3)
        # self.std_  = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], \
        #                 dtype=np.float32).reshape(1, 1, 3)
        
        # input image size
        self.inp_width_  = 512
        self.inp_height_ = 512

        # confidence threshold
        self.thresh_ = 0.0

        self.use_gpu_ = use_gpu

    def nms(self, heat, kernel=3):
        ''' Non-maximal supression
        '''
        pad = (kernel - 1) // 2
        hmax = torch.nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        # hmax == heat when this point is local maximal
        keep = (hmax == heat).float()
        return heat * keep

    def find_top_k(self, heat, K):
        ''' Find top K key points (centers) in the headmap
        '''
        batch, cat, height, width = heat.size()
        topk_scores, topk_inds = torch.topk(heat.view(batch, cat, -1), K)
        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds // width).int().float()
        topk_xs   = (topk_inds % width).int().float() 
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_inds = gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_ys, topk_xs

    def pre_process(self, image):
        ''' Preprocess the image

            Args:
                image - the image that need to be preprocessed
            Return:
                images (tensor) - images have the shape (1，3，h，w)
        '''
        height = image.shape[0]
        width = image.shape[1]

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        # shrink the image size and normalize here
        inp_image = cv2.resize(image,(self.inp_width_, self.inp_height_))

        plt.imshow(cv2.cvtColor(inp_image, cv2.COLOR_BGR2RGB))
        plt.show()

        # inp_image = ((inp_image / 255. - self.mean_) / self.std_).astype(np.float32)
        inp_image = (inp_image / 255.).astype(np.float32)

        # from three to four dimension 
        # (h, w, 3) -> (3, h, w) -> (1，3，h，w)
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, self.inp_height_, self.inp_width_)
        images = torch.from_numpy(images)

        return images

    def post_process(self, xs, ys, wh, reg):
        ''' (Will modify args) Transfer all xs, ys, wh from heatmap size to input size
        '''
        for i in range(xs.size()[1]):
            xs[0, i, 0] = xs[0, i, 0] * 4
            ys[0, i, 0] = ys[0, i, 0] * 4
            wh[0, i, 0] = wh[0, i, 0] * 4
            wh[0, i, 1] = wh[0, i, 1] * 4


    def ctdet_decode(self, heads, K = 40):
        ''' Decoding the output

            Args:
                heads ([heatmap, width/height, regression]) - network results
            Return:
                detections([batch_size, K, [xmin, ymin, xmax, ymax, score]]) 
        '''
        heat, wh, reg = heads

        batch, cat, height, width = heat.size()

        if (not self.use_gpu_):
            plot_heapmap(heat[0,0,:,:])

        heat = self.nms(heat)

        if (not self.use_gpu_):
            plot_heapmap(heat[0,0,:,:])

        scores, inds, ys, xs = self.find_top_k(heat, K)
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

        wh = transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)

        self.post_process(xs, ys, wh, reg)
        
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2, 
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores], dim=2)
        
        return detections

    def draw_bbox(self, image, detections):
        ''' Given the original image and detections results (after threshold)
            Draw bounding boxes on the image
        '''
        height = image.shape[0]
        width = image.shape[1]
        inp_image = cv2.resize(image,(self.inp_width_, self.inp_height_))
        for i in range(detections.shape[0]):
            cv2.rectangle(inp_image, \
                        (detections[i,0],detections[i,1]), \
                        (detections[i,2],detections[i,3]), \
                        (0,255,0), 1)

        original_image = cv2.resize(inp_image,(width, height))

        return original_image


    def process(self, images):
        ''' The prediction process

            Args:
                images - input images (preprocessed)
            Returns:
                output - result from the network
        '''
        with torch.no_grad():
            output = model(images)
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']

            # Generate GT data for testing
            # hm, wh, reg = generate_gt_data(400)

            heads = [hm, wh, reg]
            if (self.use_gpu_):
                torch.cuda.synchronize()
            dets = self.ctdet_decode(heads, 40) # K is the number of remaining instances

        return output, dets


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    print("Use CUDA? ", use_gpu)

    model = DlaNet(34)

    if (use_gpu):
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
        # model = nn.DataParallel(model)
        # print('Using ', torch.cuda.device_count(), "CUDAs")
        print('cuda', torch.cuda.current_device(), torch.cuda.device_count())
        device = torch.device('cuda')
        model.load_state_dict(torch.load('../best.pth'))
        model.cuda()
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load('../best.pth', map_location=torch.device('cpu')))

    model.eval()

    # get the input from the same data loader
    full_dataset = ctDataset()
    full_dataset_len = full_dataset.__len__()
    train_size = int(0.8 * full_dataset_len)
    test_size = full_dataset_len - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], \
                                                                generator=torch.Generator().manual_seed(42))
    # batch size is one
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    my_predictor = Predictor(use_gpu)

    for i, sample in enumerate(test_loader):

        if use_gpu:
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)
        
        # predict the output
        # use the dataloader result instead of do the preprocess again
        output, dets = my_predictor.process(sample['input'])

        # transfer to numpy, and reshape [batch_size, K, 5] -> [K, 5]
        # only considered batch size 1 here
        dets_np = dets.detach().cpu().numpy()[0]

        # select detections above threshold
        threshold_mask = (dets_np[:, -1] > my_predictor.thresh_)
        dets_np = dets_np[threshold_mask, :]

        print("Result: ", dets_np)

        # need to convert from heatmap coordinate to image coordinate

        result_image = my_predictor.draw_bbox(sample['image'][0].numpy(), dets_np)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.show()
        # cv2.imwrite("../sample_result.jpg", result_image) 
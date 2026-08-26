import glob
import json
import dataset
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim

from losses_MAN.bay_loss import Bay_Loss
from losses_MAN.post_prob import Post_Prob
from myNetwork_MAN import network
from model import CSRNet
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms import functional
from torch.autograd import Variable
from dataset import Dataset
from dataset1_MAN import Dataset1
import os
import cv2
import time
import h5py
import matplotlib.pyplot as plt

plt.switch_backend('agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    targets = [[transposed_batch[2][i]] for i in range(len(transposed_batch[2]))]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    label = torch.tensor(transposed_batch[4])
    return images, points, targets, st_sizes, label


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class iCaRLmodel:

    def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate, dataset_dir):

        super(iCaRLmodel, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = network(feature_extractor, flag=1)
        self.exemplar_set = []
        self.exemplar_set_gt = []
        self.class_mean_set = []
        self.numclass = numclass
        self.test_model = network(feature_extractor, flag=1)
        self.increntmal_phase = 0
        self.train_list = list()
        self.val_list = list()
        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size
        self.workers = 4
        self.train_loader = None
        self.test_loader = None
        self.train_dataset = []  # aftertrain时候使用
        self.image_list = list()
        self.label_list = list()
        self.class_name = ['others', 'your_class_name_1', 'your_class_name_2', '...', 'your_class_name_n']
        self.transform = transforms.Compose([
            transforms.Resize([400, 400]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.post_prob = Post_Prob(8.0, 512, 16, 0.15, True, device)
        self.criterion_bay = Bay_Loss(True, device)
        self.criterion_mse = nn.MSELoss(reduction='sum').to(device)
        self.currenttime = 0
        self.printtime = 10
        self.dataset_dir = dataset_dir

    def beforeTrain(self):

        self.model.eval()
        self.train_loader, self.val_loader = self._get_train_and_val_dataloader()
        if self.numclass > self.task_size:
            self.model.Incremental_learning_weight(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_val_dataloader(self):
        stage = ['category1', 'category2', '...', 'categoryn']

        self.train_dataset = glob.glob(
            os.path.join(self.dataset_dir, stage[self.increntmal_phase], 'train_data/images', '*.jpg'))
        train_dataset = Dataset1('train', self.increntmal_phase, self.exemplar_set, self.exemplar_set_gt, self.dataset_dir)
        train_loader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, collate_fn=train_collate, drop_last=True)  # shuffle==True??
        print('{0}th phase:the length of the train_dataset:{1}'.format(self.increntmal_phase, len(train_dataset)))

        val_dataset = Dataset1('val', self.increntmal_phase, None, None, self.dataset_dir)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        print('{0}th phase:the length of the val_dataset:{1}'.format(self.increntmal_phase, len(val_dataset)))

        return train_loader, val_loader


    def train(self):
        best_mae = 1e6
        best_epoch = 0

        opt = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=5 * 1e-4)

        for epoch in range(self.epochs):
            start_time = time.time()
            self.adjust_learning_rate(opt, epoch)
            for i, (inputs, points, targets, st_sizes, label) in enumerate(self.train_loader):

                loss_value = self._compute_loss(inputs, st_sizes, points, targets, label)
                opt.zero_grad()

                loss_value.backward()
                opt.step()

                print('epoch:%d,total loss:%.3f\n' % (epoch, loss_value.item())) if (self.currenttime % self.printtime == 0) else None
            end_time = time.time()
            print('epoch:{},cost time:{:.2f} sec'.format(epoch, end_time - start_time))
            print('*begin val*')

            mae = self._test(self.val_loader, 'val', self.model)
            save_path = './checkpoint'
            state = {'epoch': epoch, 'model': self.model.state_dict(), 'optimizer': opt.state_dict(), 'mae': mae}
            torch.save(state, os.path.join(save_path, 'checkpoint_latest_' + str(self.increntmal_phase) + '.pth'))

            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(state, os.path.join(save_path, 'checkpoint_best_' + str(self.increntmal_phase) + '.pth'))
            print(' * best MAE ：%.3f,best epoch : %d' % (best_mae, best_epoch))

    def _test(self, test_loader, flag, model):
        model.eval()
        mae_final = 0
        mse_final = 0

        mae = 0
        mae_total = 0
        mse_total = 0
        class_num = 0

        if flag == 'test':
            num = {0: [0, 182], 1: [0, 248, 430], 2: [0, 182, 364, 612], 3: [0, 215, 397, 645, 827],
                   4: [0, 182, 364, 612, 794, 1009], 5: [0, 176, 358, 606, 788, 1003, 1185]
                   }
        elif flag == 'val':
            num = {0: [0, 59], 1: [0, 50, 109], 2: [0, 50, 109, 159], 3: [0, 40, 99, 149, 199],
                   4: [0, 40, 99, 149, 199, 239], 5: [0, 44, 103, 153, 203, 243, 283]
                   }
        bottom = 0
        upper = 1
        all_test_dataset_img = test_loader.dataset.image_list
        all_test_dataset_label = test_loader.dataset.label_list
        all_test_dataset_target = test_loader.dataset.target
        for index_ in range(len(num[self.increntmal_phase]) - 1):

            mae = 0
            mse = 0
            test_loader.dataset.image_list = all_test_dataset_img[
                                             num[self.increntmal_phase][bottom]:num[self.increntmal_phase][upper]]
            test_loader.dataset.label_list = all_test_dataset_label[
                                             num[self.increntmal_phase][bottom]:num[self.increntmal_phase][upper]]

            for i, (img, density) in enumerate(test_loader):
                model = model.to(device)
                density = density.to(device)

                img = img.to(device)
                with torch.no_grad():

                    output, cls, _ = model(img, 1)

                for index in range(cls.shape[0]):
                    channel_num = torch.argmax(cls[index])
                    output = output[:, channel_num:channel_num + 1, :, :]
                    mae += abs(output.data.sum() - density.sum().type(torch.FloatTensor).cuda())
                    mse += ((output.data.sum() - density.sum()) ** 2).item()

                    mae_total += abs(output.data.sum() - density.sum().type(torch.FloatTensor).cuda())
                    mse_total += ((output.data.sum() - density.sum()) ** 2).item()

            mae = mae / len(test_loader.dataset.image_list)
            mse = mse / len(test_loader.dataset.image_list)
            mse = mse ** 0.5
            print('class:%d,mae:%.3f，mse:%.3f' % (class_num, mae, mse))
            mae_final += mae
            mse_final += mse
            bottom = upper
            upper = upper + 1
            class_num += 1
        mae_final = mae_final / (self.increntmal_phase + 1)
        mse_final = mse_final / (self.increntmal_phase + 1)
        test_loader.dataset.image_list = all_test_dataset_img
        test_loader.dataset.label_list = all_test_dataset_label
        test_loader.dataset.target = all_test_dataset_target
        print(' * Average MAE :%.3f ' % (mae_final))
        print(' * Average MSE :%.3f ' % (mse_final))

        mae_total = mae_total / len(all_test_dataset_img)
        mse_total = mse_total / len(all_test_dataset_img)
        mse_total = mse_total ** 0.5
        print(' **  MAE :%.2f ' % (mae_total))
        print(' **  MSE :%.2f ' % (mse_total))
        return mae_final

    def _compute_loss(self, inputs, st_sizes, points, targets, label):
        loss, loss_out = 0, 0
        inputs = inputs.to(device)
        st_sizes = st_sizes.to(device)
        gd_count = np.array([len(p) for p in points], dtype=np.float32)
        points = [p.to(device) for p in points]

        targets = [[t[0].to(device)] for t in targets]

        output, cls, _ = self.model(inputs, 1)
        label_1 = label
        label = label.to(device)
        label = get_one_hot(label, self.increntmal_phase + 2)
        cls_loss = F.binary_cross_entropy_with_logits(cls, label)
        print('cls_loss:{0}'.format(cls_loss)) if (self.currenttime % self.printtime == 0) else None
        prob_list = self.post_prob(points, st_sizes)
        for index in range(cls.shape[0]):

            channel_num = label_1[index]

            prob_list_this = prob_list[index]
            targets_this = targets[index]
            output_this = output[index:index + 1, channel_num:channel_num + 1, :, :]

            loss_out += self.criterion_bay(prob_list_this, targets_this, output_this)

        loss = loss_out
        print('count loss:{0}'.format(loss_out)) if (self.currenttime % self.printtime == 0) else None
        return loss + cls_loss


    def afterTrain(self):
        mae = 0
        self.model.eval()
        m = int(self.memory_size / (self.numclass + 1))
        self._reduce_exemplar_sets(m)

        images = self.train_dataset
        self._construct_exemplar_set_herding(images, m)
        if self.increntmal_phase > 0:
            self.test_model.output_layer = nn.Conv2d(128, out_channels=self.numclass + 1, kernel_size=1)
            self.test_model.fc = nn.Linear(512, self.numclass + 1, bias=True)

        checkpoint_val = torch.load(
            os.path.join('./checkpoint', 'checkpoint_best_' + str(self.increntmal_phase) + '.pth'))
        self.model.load_state_dict(checkpoint_val['model'])
        self.test_model.load_state_dict(checkpoint_val['model'])
        self.numclass += self.task_size

        print('*begin test*')
        test_dataset = Dataset1('test', self.increntmal_phase, None, None, self.dataset_dir)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        print('{0}th phase:the length of the test_dataset:{1}'.format(self.increntmal_phase, len(test_dataset)))
        mae = self._test(test_loader, 'test', self.test_model)


        self.increntmal_phase += 1

    def _construct_exemplar_set_random(self, images, m):
        if self.increntmal_phase == 0:
            img_people = []
            img_other = []
            for index in range(len(images)):
                data = images[index].split('images' + os.sep)[1]
                if data.startswith('class1'):
                    img_people.append(images[index])
                else:
                    img_other.append(images[index])

            self.find_exemplar_set_and_exemplar_set_gt_random(img_people, m)
            self.find_exemplar_set_and_exemplar_set_gt_random(img_other, m)
        else:
            self.find_exemplar_set_and_exemplar_set_gt_random(images, m)

    def _construct_exemplar_set_herding(self, images, m):

        if self.increntmal_phase == 0:
            img_people = []
            img_other = []
            for index in range(len(images)):
                data = images[index].split('images' + os.sep)[1]
                if data.startswith('IMG'):
                    img_people.append(images[index])
                else:
                    img_other.append(images[index])

            class_mean_people, feature_extractor_output_people = self.compute_class_mean(img_people)
            self.find_exemplar_set_and_exemplar_set_gt_herding(class_mean_people, feature_extractor_output_people,
                                                               img_people, m)
            class_mean_other, feature_extractor_output_other = self.compute_class_mean(img_other)
            self.find_exemplar_set_and_exemplar_set_gt_herding(class_mean_other, feature_extractor_output_other,
                                                               img_other, m)
        else:
            class_mean, feature_extractor_output = self.compute_class_mean(images)
            self.find_exemplar_set_and_exemplar_set_gt_herding(class_mean, feature_extractor_output, images, m)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            self.exemplar_set_gt[index] = self.exemplar_set_gt[index][:m]

    def compute_class_mean(self, images):

        img = []
        for i in range(len(images)):
            img.append(images[i])

        x = self.transform_image(img[0]).unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            _, _, feature = self.model.feature_extractor(x)
        feature = F.normalize(feature).cpu().numpy()
        for index in range(1, len(img)):

            img[index] = self.transform_image(img[index]).unsqueeze(0)
            img[index] = img[index].to(device)
            with torch.no_grad():

                _, _, feature_this = self.model.feature_extractor(img[index])
            feature_this = F.normalize(feature_this).cpu().numpy()
            feature = np.concatenate((feature_this, feature), axis=0)

        class_mean = np.mean(feature, axis=0)
        return class_mean, feature

    def find_exemplar_set_and_exemplar_set_gt_herding(self, class_mean, feature_extractor_output, images, m):
        exemplar = []
        exemplar_gt = []
        now_class_mean = np.zeros((1, 128))
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
            img_name = images[index].split('images' + os.sep)[1]
            img_name = img_name.split('_')[0]

            if img_name in self.class_name:
                gt_path = images[index].replace('.jpg', '.npy').replace('images', 'ground_truth')
                exemplar_gt.append(gt_path)
            else:
                img_read = cv2.imread(images[index])
                height = img_read.shape[0]
                width = img_read.shape[1]
                gt = np.zeros((height, width))
                exemplar_gt.append(gt)
        self.exemplar_set.append(exemplar)
        self.exemplar_set_gt.append(exemplar_gt)
        print("the size of exemplar :%s" % (str(len(exemplar))))

    def find_exemplar_set_and_exemplar_set_gt_random(self, images, m):
        exemplar = []
        exemplar_gt = []
        for i in range(m):
            exemplar.append(images[i])
            img_name = images[i].split('images' + os.sep)[1]
            img_name = img_name.split('_')[0]
            if img_name in self.class_name:
                gt_path = images[i].replace('.jpg', '.npy').replace('images', 'ground_truth')
                exemplar_gt.append(gt_path)
            else:
                img_read = cv2.imread(images[i])
                height = img_read.shape[0]
                width = img_read.shape[1]
                gt = np.zeros((height, width))
                exemplar_gt.append(gt)

        self.exemplar_set.append(exemplar)
        self.exemplar_set_gt.append(exemplar_gt)
        print("the size of exemplar :{0}".format(len(exemplar)))

    def adjust_learning_rate(self, optimizer, epoch):
        if self.increntmal_phase > 0:
            original_lr = optimizer.param_groups[0]['lr']
            if epoch > 0 and epoch % 100 == 0:
                optimizer.param_groups[0]['lr'] = original_lr / 10
                print("change learning rate:{0}".format(optimizer.param_groups[0]['lr']))
            else:
                pass
        else:
            pass

    def transform_image(self, image):
        image = Image.open(image).convert('RGB')
        height, width = image.size[1], image.size[0]
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        if height > 2000 or width > 2000:
            height, width = 2000, 2000
            image = image.resize((width, height), Image.BILINEAR)
        else:
            image = image.resize((width, height), Image.BILINEAR)
        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image

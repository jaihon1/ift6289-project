from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image
from torch.utils.data import DataLoader

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data, prepare_data_valid
from model import RNN_ENCODER, CNN_ENCODER

from inception_score import inception_score

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss, discriminator_loss_qas
import os
import time
import numpy as np
import sys
from tqdm import tqdm

from vqa.preprocess_images import Net_VQA_process
import vqa.model as vqa_model
import vqa.utils as vqa_utils

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, data_dir, output_dir, data_loader, n_words, ixtoword, datatest=None, with_vqa=True, comet=False, cfg_file=None):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
        self.data_dir = data_dir
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.data_test = datatest
        self.with_vqa = with_vqa
        self.comet = comet
        if comet:
            import comet_ml
            comet_ml.init(project_name='ift6289')
            if cfg.TRAIN.NET_G != '' and cfg.TRAIN.FLAG:
                self.experiment = comet_ml.ExistingExperiment(api_key=os.environ['COMET_API_KEY'], 
                previous_experiment=os.environ['COMET_EXPERIMENT_KEY'], log_env_details=True,
                log_env_gpu=True, log_env_cpu=True,)
            else:
                from comet_ml import Experiment
                self.experiment = Experiment(api_key=os.environ['COMET_API_KEY'])
                self.experiment.log_parameters(cfg)

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        checkpoint_g = None
        checkpoint_d = None
        if cfg.TRAIN.NET_G != '':
            checkpoint_g = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(checkpoint_g['model_state_dict'])
            if cfg.TRAIN.FLAG:
                netG.train()
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = checkpoint_g['epoch']
            epoch = int(epoch)+1
            if self.comet:
                self.experiment.set_step(epoch*self.num_batches)
                self.experiment.set_epoch(epoch)
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                checkpoint_d = []
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth.tar' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    checkpoint_d.append(torch.load(Dname, map_location=lambda storage, loc: storage))
                    netsD[i].load_state_dict(checkpoint_d[i]['model_state_dict'])
                    if cfg.TRAIN.FLAG:
                        netsD[i].train()
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.to('cuda:0')
            image_encoder = image_encoder.to('cuda:0')
            netG.to('cuda:0')
            for i in range(len(netsD)):
                netsD[i].to('cuda:0')

        return [text_encoder, image_encoder, netG, netsD, epoch, checkpoint_g, checkpoint_d]


    def define_optimizers(self, netG, netsD, checkpoint_g, checkpoint_d):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            if checkpoint_d is not None:
                print('Loading optimizers for discriminator %d' % i)
                opt.load_state_dict(checkpoint_d[i]['optimizer'])
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        if checkpoint_g is not None:
            print('Loading optimizers for generator')
            optimizerG.load_state_dict(checkpoint_g['optimizer'])

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.to('cuda:0')
            fake_labels = fake_labels.to('cuda:0')
            match_labels = match_labels.to('cuda:0')

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch, loss, optimizer_g, optimizers_d):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save({'epoch': epoch, 'model_state_dict': netG.state_dict(), 'optimizer': optimizer_g.state_dict(),
                    'loss': loss[0]},
            '%s/netG_epoch_%d.pth.tar' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            # print(netD)
            # print(netD.state_dict())
            torch.save({'epoch': epoch, 'model_state_dict': netD.state_dict(),
                        'optimizer': optimizers_d[i].state_dict(), 'loss': loss[i+1]},
                '%s/netD%d.pth.tar' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch, checkpoint_g, checkpoint_d = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD, checkpoint_g, checkpoint_d)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        ##############
        # load models needed for vqa model
        if self.with_vqa:
            real_labels_qas, fake_labels_qas, match_labels_qas = self.prepare_labels()

            processed_image_net = Net_VQA_process()
            processed_image_net.eval()

            log = torch.load(os.path.join(self.data_dir, '2017-08-04_00.55.19.pth'))
            tokens = len(log['vocab']['question']) + 1

            VQA_net = torch.nn.DataParallel(vqa_model.Net(tokens))
            VQA_net.load_state_dict(log['weights'])
            # VQA_net.eval()

            log_softmax = nn.LogSoftmax()

            if cfg.CUDA:
                log_softmax.to('cuda:0')
                processed_image_net.to('cuda:0')
                VQA_net.to('cuda:0')

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.to('cuda:0'), fixed_noise.to('cuda:0')

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            netG.train()
            acc_epoch = 0
            data_iter = iter(self.data_loader)
            # step = 0
            for step in tqdm(range(self.num_batches)):
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys, qas_gan, qas_len, q_vqa, ans_vqa, item, q_len_vqa = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #############################
                # for vqa
                if self.with_vqa:
                    hidden_qas = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    sort_qas_lens, sorted_qas_indices = \
                        torch.sort(qas_len, 0, True)
                    qas_gan = qas_gan[sorted_qas_indices].squeeze()
                    match_labels_qas = match_labels_qas[sorted_qas_indices]
                    words_embs_qas, sent_emb_qas = text_encoder(qas_gan, sort_qas_lens, hidden_qas)
                    words_embs_qas, sent_emb_qas = words_embs_qas.detach(), sent_emb_qas.detach()
                    mask_qas = (qas_gan == 0)
                    num_words_qas = words_embs_qas.size(2)
                    if mask_qas.size(1) > num_words_qas:
                        mask_qas = mask_qas[:, :num_words_qas]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.detach().normal_(0, 1)
                # fake_imgs on captions
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                if self.with_vqa:
                    # fake images on QAs pairs
                    noise.detach().normal_(0,1)
                    fake_imgs_qas, _, mu_qas, logvar_qas = netG(noise, sent_emb_qas, words_embs_qas, mask_qas)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                lossD=[]
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    ############
                    # for vqa
                    if self.with_vqa:
                        errD+=discriminator_loss_qas(netsD[i], imgs[i][sorted_qas_indices], fake_imgs_qas[i],
                                                     sent_emb_qas, fake_labels_qas)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    lossD.append(errD.item())
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                # step += 1
                gen_iterations += 1

                #############################################
                # pass through vqa model
                if self.with_vqa:
                    fake_imgs_processed = []
                    loss_vqa = 0
                    VQA_net.zero_grad()
                    acc = []
                    for i in range(len(netsD)):
                        # prepare fake images for VQA model
                        fake_imgs_processed.append(processed_image_net(fake_imgs_qas[i]).type(torch.FloatTensor))
                        sort_q_lens, sorted_q_indices = \
                            torch.sort(q_len_vqa, 0, True)
                        q_vqa = q_vqa[sorted_q_indices]
                        ans_vqa = ans_vqa[sorted_q_indices]
                        out = VQA_net(fake_imgs_processed[i][sorted_q_indices], q_vqa, sort_q_lens)

                        nll = -log_softmax(out)
                        loss_vqa += (nll * ans_vqa / 10).sum(dim=1).mean()
                        acc.append(vqa_utils.batch_accuracy(out.detach(), ans_vqa.detach()).mean().item())
                        acc_epoch+=float(sum(acc)/3)
                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.data.item()

                ##################
                # compute loss with added vqa model
                if self.with_vqa:
                    loss_logs = generator_loss(netsD, image_encoder, fake_imgs_qas, real_labels_qas,
                                               words_embs_qas, sent_emb_qas, match_labels_qas, sort_qas_lens,
                                               class_ids[sorted_qas_indices.cpu().numpy()])
                    # loss vqa
                    kl_loss_qas = KL_loss(mu_qas, logvar_qas)
                    errG_total += kl_loss_qas
                    errG_total += loss_vqa
                    errG_total += loss_logs[0]
                    G_logs += loss_logs[1]
                    G_logs += 'kl_loss_qas: %.2f ' % kl_loss_qas.data.item()
                    G_logs += 'Accuracy VQA: %.2f' % float(sum(acc)/3)

                ######################
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(p.detach(), alpha=0.001)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
                if self.comet:
                    self.experiment.log_metrics({'loss_d0': lossD[0], 'loss_d1': lossD[1], 'loss_d2': lossD[2],
                                                 'loss_G': errG_total}, step=self.num_batches*epoch+step)
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch, [errG_total.item()]+lossD, optimizerG, optimizersD)
            
            mean_inc_score, std_inc_score, fid = 0, 0, 0
            if self.data_test:
                mean_inc_score, std_inc_score, fid, _ = self.evaluate(netG, text_encoder, epoch)
            
            if self.comet:
                self.experiment.log_metrics({'accuracy_vqa':acc_epoch/step, 'inc_score':mean_inc_score,
                                             'std_inc_score': std_inc_score}, step=epoch)
        self.save_model(netG, avg_param_G, netsD, self.max_epoch, [errG_total.item()]+lossD, optimizerG, optimizersD)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).detach().cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.to('cuda:0')
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.to('cuda:0')
            text_encoder.eval()

            # batch_size = self.batch_size
            # nz = cfg.GAN.Z_DIM
            # noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            # noise = noise.to('cuda:0')

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict['model_state_dict'])
            print('Load G from: ', model_dir)
            mean_is, std_is, fid, images = self.evaluate(netG, text_encoder, state_dict['epoch'])
            print('Inception Score: %s, FID: %s' % (mean_is, fid))

            processed_image_net = Net_VQA_process()
            processed_image_net.eval()

            log = torch.load(os.path.join(self.data_dir, '2017-08-04_00.55.19.pth'))
            tokens = len(log['vocab']['question']) + 1

            VQA_net = torch.nn.DataParallel(vqa_model.Net(tokens))
            VQA_net.load_state_dict(log['weights'])
            VQA_net.eval()

            if cfg.CUDA:
                processed_image_net.to('cuda:0')
                VQA_net.to('cuda:0')

            fake_imgs = torch.stack(images)
            loader_images = DataLoader(fake_imgs, batch_size=self.batch_size)
            acc = []
            for data, imgs in zip(self.data_loader, loader_images):
                imgs_processed = processed_image_net(imgs).type(torch.FloatTensor)
                imgs, captions, cap_lens, class_ids, keys, qas_gan, qas_len, q_vqa, ans_vqa, item, q_len_vqa = prepare_data(data)
                sort_q_lens, sorted_q_indices = \
                    torch.sort(q_len_vqa, 0, True)
                q_vqa = q_vqa[sorted_q_indices]
                ans_vqa = ans_vqa[sorted_q_indices]
                out = VQA_net(imgs_processed[sorted_q_indices], q_vqa, sort_q_lens)

                acc.append(vqa_utils.batch_accuracy(out.detach(), ans_vqa.detach()).mean().item())
            mean_acc = float(sum(acc)/len(loader_images))
            print('Mean accuracy: %s' % mean_acc)
            # # the path to save generated images
            # s_tmp = model_dir[:model_dir.rfind('.pth')]
            # save_dir = '%s/%s' % (s_tmp, split_dir)
            # mkdir_p(save_dir)
            #
            # cnt = 0
            #
            # for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            #     for step, data in enumerate(self.data_loader, 0):
            #         cnt += batch_size
            #         if step % 100 == 0:
            #             print('step: ', step)
            #         # if step > 50:
            #         #     break
            #
            #         imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
            #
            #         hidden = text_encoder.init_hidden(batch_size)
            #         # words_embs: batch_size x nef x seq_len
            #         # sent_emb: batch_size x nef
            #         words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            #         words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            #         mask = (captions == 0)
            #         num_words = words_embs.size(2)
            #         if mask.size(1) > num_words:
            #             mask = mask[:, :num_words]
            #
            #         #######################################################
            #         # (2) Generate fake images
            #         ######################################################
            #         noise.detach().normal_(0, 1)
            #         fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
            #         for j in range(batch_size):
            #             s_tmp = '%s/single/%s' % (save_dir, keys[j])
            #             folder = s_tmp[:s_tmp.rfind('/')]
            #             if not os.path.isdir(folder):
            #                 print('Make a new folder: ', folder)
            #                 mkdir_p(folder)
            #             k = -1
            #             # for k in range(len(fake_imgs)):
            #             im = fake_imgs[k][j].detach().cpu().numpy()
            #             # [-1, 1] --> [0, 255]
            #             im = (im + 1.0) * 127.5
            #             im = im.astype(np.uint8)
            #             im = np.transpose(im, (1, 2, 0))
            #             im = Image.fromarray(im)
            #             fullpath = '%s_s%d.png' % (s_tmp, k)
            #             im.save(fullpath)

    def evaluate(self, netG, text_encoder, epoch):
        netG.eval()
        text_encoder.eval()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        images_to_return = None
        with torch.no_grad():
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.to('cuda:0')

            # the path to save generated images
            save_dir = '%s/epoch_%d/%s' % (self.image_dir, epoch, 'valid')
            mkdir_p(save_dir)

            cnt = 0
            mean, std = 0, 0
            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                images = []
                for step, data in enumerate(self.data_test, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)
                    # if step > 50:
                    #     break

                    imgs, captions, cap_lens, class_ids, keys = prepare_data_valid(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.detach().normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].detach().cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        images.append(im)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        if step<10 or not cfg.TRAIN.FLAG:
                            fullpath = '%s_s%d.png' % (s_tmp, k)
                            im.save(fullpath)
            mean, std = inception_score(images, resize=True)
            fid = 0
            if not cfg.TRAIN.FLAG:
                valid_dir = os.path.join(self.data_dir, 'images_val_crop')
                fid = os.popen('python -m pytorch_fid %s %s' % (valid_dir, folder)).read().rstrip('\n').split(' ')[-1]
                images_to_return = images
        return mean, std, float(fid), images_to_return

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.to('cuda:0')
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.to('cuda:0')
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.to('cuda:0')
                cap_lens = cap_lens.to('cuda:0')
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.to('cuda:0')
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.detach().normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().detach().numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].detach().cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)

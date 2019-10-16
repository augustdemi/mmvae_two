import os
import numpy as np

import torch.optim as optim
from mnist.dataset import digit
from mnist.datasets import DIGIT
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# -----------------------------------------------------------------------------#

from utils import DataGather, mkdirs, grid2gif2, apply_poe, sample_gaussian, sample_gumbel_softmax, \
    get_log_pz_qz_prodzi_qzCx
from mnist.model import *
from loss import cross_entropy_label, reconstruction_loss
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

###############################################################################

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        self.name = '%s_lamkl_%s_zA_%s_zB_%s_zS_%s_HYPER_beta1_%s_beta2_%s_beta3_%s_beta11_%s_beta22_%s_beta33_%s_lA_%s_lB_%s' % \
                    (
                        args.dataset, args.lamkl, args.zA_dim, args.zB_dim, args.zS_dim, args.beta1, args.beta2,
                        args.beta3, args.beta11, args.beta22, args.beta33,args.lambdaA, args.lambdaB)
        # to be appended by run_id

        self.use_cuda = args.cuda and torch.cuda.is_available()

        self.max_iter = int(args.max_iter)

        # do it every specified iters
        self.print_iter = args.print_iter
        self.ckpt_save_iter = args.ckpt_save_iter
        self.output_save_iter = args.output_save_iter

        # data info
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.nc = 3
        self.categ = args.categ

        # self.N = self.latent_values.shape[0]
        self.eval_metrics_iter = args.eval_metrics_iter

        # networks and optimizers
        self.batch_size = args.batch_size
        self.zA_dim = args.zA_dim
        self.zB_dim = args.zB_dim
        self.zS_dim = args.zS_dim
        self.lamkl = args.lamkl
        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.beta3 = args.beta3

        self.beta11 = args.beta11
        self.beta22 = args.beta22
        self.beta33 = args.beta33
        self.is_mss = args.is_mss
        self.cross_loss = args.cross_loss

        self.lambdaA = args.lambdaA
        self.lambdaB = args.lambdaB
        self.paired_cnt = args.paired_cnt
        self.unsup = args.unsup

        # visdom setup
        self.viz_on = args.viz_on
        if self.viz_on:
            self.win_id = dict(
                recon='win_recon', kl='win_kl', tc='win_tc', mi='win_mi', dw_kl='win_dw_kl', acc='win_acc', mgll='win_mgll', acc_te='win_acc_te', mgll_te='win_mgll_te'
            )
            self.line_gather = DataGather(
                'iter', 'recon_both', 'recon_A', 'recon_B',
                'kl_A', 'kl_B', 'kl_POE',
                'tc_loss', 'mi_loss', 'dw_kl_loss',
                'tc_loss_A', 'mi_loss_A', 'dw_kl_loss_A',
                'tc_loss_B', 'mi_loss_B', 'dw_kl_loss_B',
                'tc_loss_POEA', 'mi_loss_POEA', 'dw_kl_loss_POEA',
                'tc_loss_POEB', 'mi_loss_POEB', 'dw_kl_loss_POEB',
                'marginal_ll_A_infA', 'marginal_ll_A_poe', 'marginal_ll_A_pAsB', 'acc_infB', 'acc_POE', 'acc_sinfA',
                'marginal_ll_A_infA_te', 'marginal_ll_A_poe_te', 'marginal_ll_A_pAsB_te', 'acc_infB_te', 'acc_POE_te', 'acc_sinfA_te',
                'acc_infA', 'acc_infA_te'
            )

            # if self.eval_metrics:
            #     self.win_id['metrics'] = 'win_metrics'

            import visdom

            self.viz_port = args.viz_port  # port number, eg, 8097
            self.viz = visdom.Visdom(port=self.viz_port)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter

            self.viz_init()

        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");
        mkdirs("ckpts");
        mkdirs("outputs")

        # set run id
        if args.run_id < 0:  # create a new id
            k = 0;
            rfname = os.path.join("records", self.name + '_run_0.txt')
            while os.path.exists(rfname):
                k += 1
                rfname = os.path.join("records", self.name + '_run_%d.txt' % k)
            self.run_id = k
        else:  # user-provided id
            self.run_id = args.run_id

        # finalize name
        self.name = self.name + '_run_' + str(self.run_id)

        # records (text file to store console outputs)
        self.record_file = 'records/%s.txt' % self.name

        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)

        # outputs
        self.output_dir_recon = os.path.join("outputs", self.name + '_recon')
        # dir for reconstructed images
        self.output_dir_synth = os.path.join("outputs", self.name + '_synth')
        # dir for synthesized images
        self.output_dir_trvsl = os.path.join("outputs", self.name + '_trvsl')

        #### create a new model or load a previously saved model

        self.ckpt_load_iter = args.ckpt_load_iter
        self.n_pts = args.n_pts
        self.n_data = args.n_data

        if self.ckpt_load_iter == 0:  # create a new model

            if self.categ:
                self.encoderA = ImageEncoder(self.zA_dim, self.zS_dim)
                self.encoderB = TextEncoder(self.zB_dim, self.zS_dim)
                self.decoderA = ImageDecoder(self.zA_dim, self.zS_dim)
                self.decoderB = TextDecoder(self.zB_dim, self.zS_dim)
            else:
                self.encoderA = EncoderSingle3(self.zA_dim, self.zS_dim)
                self.encoderB = EncoderSingle3(self.zB_dim, self.zS_dim)
                self.decoderA = DecoderSingle3(self.zA_dim, self.zS_dim)
                self.decoderB = DecoderSingle3(self.zB_dim, self.zS_dim)


        else:  # load a previously saved model

            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')

        if self.use_cuda:
            print('Models moved to GPU...')
            self.encoderA = self.encoderA.cuda()
            self.encoderB = self.encoderB.cuda()
            self.decoderA = self.decoderA.cuda()
            self.decoderB = self.decoderB.cuda()
            print('...done')

        # get VAE parameters
        vae_params = \
            list(self.encoderA.parameters()) + \
            list(self.encoderB.parameters()) + \
            list(self.decoderA.parameters()) + \
            list(self.decoderB.parameters())


        # create optimizers
        self.optim_vae = optim.Adam(
            vae_params,
            lr=self.lr_VAE,
            betas=[self.beta1_VAE, self.beta2_VAE]
        )


    def kl_loss(self, log_pz, log_qz, log_prod_qzi, log_q_zCx):
        mi_loss = (log_q_zCx - log_qz).mean()
        tc_loss = (log_qz - log_prod_qzi).sum(dim=0).div(self.batch_size)
        dw_kl_loss = (log_prod_qzi - log_pz).mean()
        loss_kl_infA = self.beta1 * mi_loss + self.beta2 * tc_loss + self.beta3 * dw_kl_loss
        return mi_loss, tc_loss, dw_kl_loss, loss_kl_infA

    def log_prob(self, z_private, z_shared, mu, logvar, relaxedCateg, train=True):
        if train:
            n_data = len(self.data_loader.dataset)
        else:
            n_data = len(self.test_data_loader.dataset)
        log_pz, log_qz, log_prod_qzi, log_q_zCx, _ = get_log_pz_qz_prodzi_qzCx(
            {'cont': z_private, 'disc': z_shared}, {'cont': (mu, logvar), 'disc': relaxedCateg}, n_data, is_mss=self.is_mss)
        return log_pz, log_qz, log_prod_qzi, log_q_zCx

    ####
    def train(self):

        self.set_mode(train=True)

        # prepare dataloader (iterable)
        print('Start loading data...')
        if self.categ:
            dset = digit('./data', train=True)
        else:
            dset = DIGIT('./data', train=True)
        self.data_loader = torch.utils.data.DataLoader(dset, batch_size=self.batch_size, shuffle=True)

        ############ for weakly supervised ############
        paired_idx = dset.get_paired_data(self.paired_cnt)

        paired_XA = [0] * len(paired_idx)
        paired_XB = [0] * len(paired_idx)
        for i, idx in enumerate(paired_idx):
            paired_XA[i], paired_XB[i]= \
                self.data_loader.dataset.__getitem__(idx)[0:2]
            if self.use_cuda:
                paired_XA[i] = paired_XA[i].cuda()
                paired_XB[i] = paired_XB[i].cuda()
        paired_XA = torch.stack(paired_XA)
        paired_XB = torch.stack(paired_XB)

        ##############################################

        test_dset = digit('./data', train=False)
        self.test_data_loader = torch.utils.data.DataLoader(test_dset, batch_size=self.batch_size, shuffle=True)
        # self.test_data_loader = self.data_loader
        self.N = len(self.data_loader.dataset)
        print('...done')

        # iterators from dataloader
        iterator1 = iter(self.data_loader)

        iter_per_epoch = len(iterator1)

        start_iter = self.ckpt_load_iter + 1
        epoch = int(start_iter / iter_per_epoch)

        for iteration in range(start_iter, self.max_iter + 1):
            # iteration = iteration-1
            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)
                epoch += 1
                iterator1 = iter(self.data_loader)

            # ============================================
            #          TRAIN THE VAE (ENC & DEC)
            # ============================================

            # sample a mini-batch
            XA, XB, _, _ = next(iterator1)  # (n x C x H x W)
            if not self.unsup:
                XA = paired_XA
                XB = paired_XB

            if self.use_cuda:
                XA = XA.cuda()
                XB = XB.cuda()

            # zA, zS = encA(xA)
            if self.categ:
                muA_infA, stdA_infA, logvarA_infA, cate_prob_infA = self.encoderA(XA)
                # zB, zS = encB(xB)
                muB_infB, stdB_infB, logvarB_infB, cate_prob_infB = self.encoderB(XB)

                '''
                POE: should be the paramter for the distribution
                induce zS = encAB(xA,xB) via POE, that is,
                    q(zA,zB,zS | xA,xB) := qI(zA|xA) * qT(zB|xB) * q(zS|xA,xB)
                        where q(zS|xA,xB) \propto p(zS) * qI(zS|xA) * qT(zS|xB)
                '''
                cate_prob_POE = cate_prob_infA * cate_prob_infB

                # encoder samples (for training)
                ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
                ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)

                # ZS need the distribution class to calculate the pmf value in decompositon of KL
                Eps = 1e-12
                # distribution
                relaxedCategA = ExpRelaxedCategorical(torch.tensor(.67), logits=torch.log(cate_prob_infA + Eps))
                relaxedCategB = ExpRelaxedCategorical(torch.tensor(.67), logits=torch.log(cate_prob_infB + Eps))
                relaxedCategS = ExpRelaxedCategorical(torch.tensor(.67), logits=torch.log(cate_prob_POE + Eps))

                # sampling
                log_ZS_infA = relaxedCategA.rsample()
                ZS_infA = torch.exp(log_ZS_infA)
                log_ZS_infB = relaxedCategB.rsample()
                ZS_infB = torch.exp(log_ZS_infB)
                log_ZS_POE = relaxedCategS.rsample()
                ZS_POE = torch.exp(log_ZS_POE)
                ZS_POE = torch.sqrt(ZS_POE)
                # the above sampling of ZS_infA/B are same 'way' as below
                # ZS_infA = sample_gumbel_softmax(self.use_cuda, cate_prob_infA)
                # ZS_infB = sample_gumbel_softmax(self.use_cuda, cate_prob_infB)

            else:
                muA_infA, stdA_infA, logvarA_infA, \
                muS_infA, stdS_infA, logvarS_infA = self.encoderA(XA)

                # zB, zS = encB(xB)
                muB_infB, stdB_infB, logvarB_infB, \
                muS_infB, stdS_infB, logvarS_infB = self.encoderB(XB)

                # zS = encAB(xA,xB) via POE
                muS_POE, stdS_POE, logvarS_POE = apply_poe(
                    self.use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB,
                )

                # encoder samples (for training)
                ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
                ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)
                ZS_POE  = sample_gaussian(self.use_cuda, muS_POE, stdS_POE)
                # encoder samples (for cross-modal prediction)
                ZS_infA = sample_gaussian(self.use_cuda, muS_infA, stdS_infA)
                ZS_infB = sample_gaussian(self.use_cuda, muS_infB, stdS_infB)

            #### For all cate_prob_infA(statiscts), total 64, get log_prob_ZS_infB2 for each of ZS_infB2(sample) ==> 64*64. marig. out for q_z for MI


            # one modal
            XA_infA_recon = self.decoderA(ZA_infA, ZS_infA)
            XB_infB_recon = self.decoderB(ZB_infB, ZS_infB)
            # POE
            XA_POE_recon = self.decoderA(ZA_infA, ZS_POE)
            XB_POE_recon = self.decoderB(ZB_infB, ZS_POE)
            # cross shared
            XA_sinfB_recon = self.decoderA(ZA_infA, ZS_infB)
            XB_sinfA_recon = self.decoderB(ZB_infB, ZS_infA)

            # one modal
            loss_recon_infA = reconstruction_loss(XA, torch.sigmoid(XA_infA_recon).view(XA.shape[0],-1,28,28), distribution="bernoulli")
            loss_recon_infB = cross_entropy_label(XB_infB_recon, XB)
            # POE
            loss_recon_POE = \
                self.lambdaA * reconstruction_loss(XA, torch.sigmoid(XA_POE_recon).view(XA.shape[0],-1,28,28), distribution="bernoulli") + \
                self.lambdaB * cross_entropy_label(XB_POE_recon, XB)

            # if self.paired_cnt and iteration % iter_per_epoch != 0:
            #     loss_recon = self.lambdaA * loss_recon_infA + self.lambdaB * loss_recon_infB
            #     loss_recon /= 2.0
            if self.unsup:
                loss_recon = self.lambdaA * loss_recon_infA + self.lambdaB * loss_recon_infB
                # loss_recon /= 2.0
            else:
                loss_recon = self.lambdaA * loss_recon_infA + self.lambdaB * loss_recon_infB + loss_recon_POE
                # loss_recon /= 3.0
                if self.cross_loss:
                    # cross shared
                    loss_reconA_sinfB = reconstruction_loss(XA, torch.sigmoid(XA_sinfB_recon).view(XA.shape[0], -1, 28, 28),
                                                            distribution="bernoulli")
                    loss_reconB_sinfA = cross_entropy_label(XB_sinfA_recon, XB)
                    loss_cross = self.lambdaA * loss_reconA_sinfB + self.lambdaB * loss_reconB_sinfA
                    loss_recon += loss_cross
                    # loss_recon /= 5.0

            #================================== decomposed KL ========================================

            if self.categ:
                log_pz_A, log_qz_A, log_prod_qzi_A, log_q_zCx_A = self.log_prob(ZA_infA, ZS_infA, muA_infA, logvarA_infA, relaxedCategA)
                log_pz_B, log_qz_B, log_prod_qzi_B, log_q_zCx_B = self.log_prob(ZB_infB, ZS_infB, muB_infB, logvarB_infB, relaxedCategB)
                log_pz_POEA, log_qz_POEA, log_prod_qzi_POEA, log_q_zCx_POEA = self.log_prob(ZA_infA, ZS_POE, muA_infA, logvarA_infA, relaxedCategS)
                log_pz_POEB, log_qz_POEB, log_prod_qzi_POEB, log_q_zCx_POEB = self.log_prob(ZB_infB, ZS_POE, muB_infB, logvarB_infB, relaxedCategS)
                if self.cross_loss:
                    log_pz_A_sB, log_qz_A_sB, log_prod_qzi_A_sB, log_q_zCx_A_sB = self.log_prob(ZA_infA, ZS_infB, muA_infA, logvarA_infA, relaxedCategB)
                    log_pz_B_sA, log_qz_B_sA, log_prod_qzi_B_sA, log_q_zCx_B_sA = self.log_prob(ZB_infB, ZS_infA, muB_infB, logvarB_infB, relaxedCategA)
            else:
                log_pz_A, log_qz_A, log_prod_qzi_A, log_q_zCx_A, _ = get_log_pz_qz_prodzi_qzCx(
                    {'cont':  torch.cat((ZA_infA, ZS_infA), dim=1)}, {'cont': (torch.cat((muA_infA, muS_infA), dim=1),  torch.cat((logvarA_infA, logvarS_infA), dim=1))},
                    len(self.data_loader.dataset),
                    is_mss=self.is_mss)

                log_pz_B, log_qz_B, log_prod_qzi_B, log_q_zCx_B, _ = get_log_pz_qz_prodzi_qzCx(
                    {'cont': torch.cat((ZB_infB, ZS_infB), dim=1)}, {'cont': (torch.cat((muB_infB, muS_infB), dim=1), torch.cat((logvarB_infB, logvarS_infB), dim=1))},
                    len(self.data_loader.dataset),
                    is_mss=self.is_mss)

                log_pz_POEA, log_qz_POEA, log_prod_qzi_POEA, log_q_zCx_POEA, _ = get_log_pz_qz_prodzi_qzCx(
                    {'cont': torch.cat((ZA_infA, ZS_POE), dim=1)}, {'cont': (torch.cat((muA_infA, muS_POE), dim=1), torch.cat((logvarA_infA, logvarS_POE), dim=1))},
                    len(self.data_loader.dataset),
                    is_mss=self.is_mss)

                log_pz_POEB, log_qz_POEB, log_prod_qzi_POEB, log_q_zCx_POEB, _ = get_log_pz_qz_prodzi_qzCx(
                    {'cont': torch.cat((ZB_infB, ZS_POE), dim=1)}, {'cont': (torch.cat((muB_infB, muS_POE), dim=1), torch.cat((logvarB_infB, logvarS_POE), dim=1))},
                    len(self.data_loader.dataset),
                    is_mss=self.is_mss)
            # loss_kl_infA
            mi_loss_A, tc_loss_A, dw_kl_loss_A, loss_kl_infA = self.kl_loss(log_pz_A, log_qz_A, log_prod_qzi_A, log_q_zCx_A)
            # loss_kl_infB
            mi_loss_B, tc_loss_B, dw_kl_loss_B, loss_kl_infB = self.kl_loss(log_pz_B, log_qz_B, log_prod_qzi_B, log_q_zCx_B)
            # loss_kl_POEA
            mi_loss_POEA, tc_loss_POEA, dw_kl_loss_POEA, loss_kl_POEA = self.kl_loss(log_pz_POEA, log_qz_POEA, log_prod_qzi_POEA, log_q_zCx_POEA)
            # loss_kl_POEB
            mi_loss_POEB, tc_loss_POEB, dw_kl_loss_POEB, loss_kl_POEB = self.kl_loss(log_pz_POEB, log_qz_POEB, log_prod_qzi_POEB, log_q_zCx_POEB)
            # loss_kl_POE
            loss_kl_POE = 0.5 * (loss_kl_POEA + loss_kl_POEB)



            # if self.paired_cnt and iteration % iter_per_epoch != 0:
            if self.unsup:
                loss_kl = loss_kl_infA + loss_kl_infB
                tc_loss = tc_loss_A + tc_loss_B
                mi_loss = mi_loss_A + mi_loss_B
                dw_kl_loss = dw_kl_loss_A + dw_kl_loss_B
                # loss_kl /= 2.0
                # tc_loss /= 2.0
                # mi_loss /= 2.0
                # dw_kl_loss /= 2.0
            else:
                loss_kl = loss_kl_infA + loss_kl_infB + loss_kl_POE
                tc_loss = tc_loss_A + tc_loss_B + 0.5 * (tc_loss_POEA + tc_loss_POEB)
                mi_loss = mi_loss_A + mi_loss_B + 0.5 * (mi_loss_POEA + mi_loss_POEB)
                dw_kl_loss = dw_kl_loss_A + dw_kl_loss_B + 0.5 * (dw_kl_loss_POEA + dw_kl_loss_POEB)
                # loss_kl /= 3.0
                # tc_loss /= 3.0
                # mi_loss /= 3.0
                # dw_kl_loss /= 3.0
                if self.cross_loss:
                    # loss_kl_infA_sB
                    mi_loss_A_sB, tc_loss_A_sB, dw_kl_loss_A_sB, loss_kl_infA_sB = self.kl_loss(log_pz_A_sB, log_qz_A_sB,
                                                                                                log_prod_qzi_A_sB,
                                                                                                log_q_zCx_A_sB)
                    # loss_kl_infB_sA
                    mi_loss_B_sA, tc_loss_B_sA, dw_kl_loss_B_sA, loss_kl_infB_sA = self.kl_loss(log_pz_B_sA, log_qz_B_sA,
                                                                                                log_prod_qzi_B_sA,
                                                                                                log_q_zCx_B_sA)
                    loss_kl += loss_kl_infA_sB + loss_kl_infB_sA
                    tc_loss += tc_loss_A_sB + tc_loss_B_sA
                    mi_loss += mi_loss_A_sB + mi_loss_B_sA
                    dw_kl_loss += dw_kl_loss_A_sB + dw_kl_loss_B_sA
                    loss_kl /= 5.0
                    tc_loss /= 5.0
                    mi_loss /= 5.0
                    dw_kl_loss /= 5.0

            ################## total loss for vae ####################
            vae_loss = loss_recon + loss_kl


            ####### update vae ##########
            self.optim_vae.zero_grad()
            vae_loss.backward()
            self.optim_vae.step()



            # print the losses
            if iteration % self.print_iter == 0:
                prn_str = ( \
                                      '[iter %d (epoch %d)] vae_loss: %.3f ' + \
                                      '(recon: %.3f, kl: %.3f)\n' + \
                                      '    rec_infA = %.3f, rec_infB = %.3f, rec_POE = %.3f\n' + \
                                      '    kl_infA = %.3f, kl_infB = %.3f, kl_POE = %.3f'
                              ) % \
                          (iteration, epoch,
                           vae_loss.item(), loss_recon.item(), loss_kl.item(),
                           loss_recon_infA.item(), loss_recon_infB.item(), loss_recon_POE.item(),
                           loss_kl_infA.item(), loss_kl_infB.item(), loss_kl_POE.item()
                           )

                print(prn_str)
                if self.record_file:
                    record = open(self.record_file, 'a')
                    record.write('%s\n' % (prn_str,))
                    record.close()

            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                self.save_checkpoint(iteration)

            # save output images (recon, synth, etc.)
            if iteration % self.output_save_iter == 0:
                # self.save_embedding(iteration, index, muA_infA, muB_infB, muS_infA, muS_infB, muS_POE)
                z_A, z_B, z_AS, z_BS, z_S = self.get_stat()

                print('------------ traverse interpolation ------------')
                print('interpolationA: ', np.min(np.array(z_A)), np.max(np.array(z_A)))
                # print('interpolationB: ', np.min(np.array(z_B)), np.max(np.array(z_B)))
                if not self.categ:
                    print('interpolationAS: ', np.min(np.array(z_AS)), np.max(np.array(z_AS)))
                    print('interpolationBS: ', np.min(np.array(z_BS)), np.max(np.array(z_BS)))
                    print('interpolationS: ', np.min(np.array(z_S)), np.max(np.array(z_S)))
                # 1) save the recon images
                self.save_recon(iteration)
                self.save_recon(iteration, train=False)

                self.save_synth_cross_modal(iteration, z_A, z_B, howmany=3)
                self.save_synth_cross_modal(iteration, z_A, z_B, train=False, howmany=3)
                # self.save_traverse(iteration, z_A, z_B)
                self.save_traverseA(iteration)
                self.save_traverseA(iteration, train=False)
                # self.save_traverse(iteration, z_A, z_B, train=False)

            # if iteration % self.eval_metrics_iter == 0:
            #     self.save_synth_cross_modal(iteration, z_A, z_B, train=False, howmany=3)





            # (visdom) insert current line stats
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                z_A, z_B, z_AS, z_BS, z_S = self.get_stat()

                marginal_ll_A_infA, marginal_ll_A_poe, marginal_ll_A_pAsB, acc_infB, acc_POE, acc_sinfA, acc_infA = self.get_loglikelihood(z_B)
                marginal_ll_A_infA_te, marginal_ll_A_poe_te, marginal_ll_A_pAsB_te, acc_infB_te, acc_POE_te, acc_sinfA_te, acc_infA_te = self.get_loglikelihood(z_B, train=False)

                prn_str = ( \
                                    '[iter %d (epoch %d)]\n' + \
                                    '    marginal_ll_A_infA = %.3f, marginal_ll_A_poe = %.3f, marginal_ll_A_pAsB = %.3f\n' + \
                                    '    marginal_ll_A_infA_te = %.3f, marginal_ll_A_poe_te = %.3f, marginal_ll_A_pAsB_te = %.3f\n' + \
                                    '    acc_infB = %.3f, acc_POE = %.3f, acc_sinfA = %.3f, acc_infA = %.3f\n' + \
                                      '    acc_infB_te = %.3f, acc_POE_te = %.3f, acc_sinfA_te = %.3f, acc_infA_te = %.3f]'
                              ) % \
                          (iteration, epoch,
                           marginal_ll_A_infA.item(), marginal_ll_A_poe.item(), marginal_ll_A_pAsB.item(),
                           marginal_ll_A_infA_te.item(), marginal_ll_A_poe_te.item(), marginal_ll_A_pAsB_te.item(),
                           acc_infB.item(), acc_POE.item(), acc_sinfA.item(), acc_infA.item(),
                           acc_infB_te.item(), acc_POE_te.item(), acc_sinfA_te.item(), acc_infA_te.item()
                           )
                print(prn_str)
                print('======================================================================================================')

                self.line_gather.insert(iter=iteration,
                                        recon_both=loss_recon_POE.item(),
                                        recon_A=loss_recon_infA.item(),
                                        recon_B=loss_recon_infB.item(),
                                        kl_A=loss_kl_infA.item(),
                                        kl_B=loss_kl_infB.item(),
                                        kl_POE=loss_kl_POE.item(),
                                        tc_loss=tc_loss.item(),
                                        mi_loss=mi_loss.item(),
                                        dw_kl_loss=dw_kl_loss.item(),
                                        tc_loss_A=tc_loss_A.item(),
                                        mi_loss_A=mi_loss_A.item(),
                                        dw_kl_loss_A=dw_kl_loss_A.item(),
                                        tc_loss_B=tc_loss_B.item(),
                                        mi_loss_B=mi_loss_B.item(),
                                        dw_kl_loss_B=dw_kl_loss_B.item(),
                                        tc_loss_POEA=tc_loss_POEA.item(),
                                        mi_loss_POEA=mi_loss_POEA.item(),
                                        dw_kl_loss_POEA=dw_kl_loss_POEA.item(),
                                        tc_loss_POEB=tc_loss_POEB.item(),
                                        mi_loss_POEB=mi_loss_POEB.item(),
                                        dw_kl_loss_POEB=dw_kl_loss_POEB.item(),
                                        marginal_ll_A_infA=marginal_ll_A_infA.item(),
                                        marginal_ll_A_poe=marginal_ll_A_poe.item(),
                                        marginal_ll_A_pAsB=marginal_ll_A_pAsB.item(),
                                        acc_infB=acc_infB.item(),
                                        acc_POE=acc_POE.item(),
                                        acc_sinfA=acc_sinfA.item(),
                                        acc_infA=acc_infA.item(),
                                        marginal_ll_A_infA_te=marginal_ll_A_infA_te.item(),
                                        marginal_ll_A_poe_te=marginal_ll_A_poe_te.item(),
                                        marginal_ll_A_pAsB_te=marginal_ll_A_pAsB_te.item(),
                                        acc_infB_te=acc_infB_te.item(),
                                        acc_POE_te=acc_POE_te.item(),
                                        acc_sinfA_te=acc_sinfA_te.item(),
                                        acc_infA_te=acc_infA_te.item(),
                                        )


            # (visdom) visualize line stats (then flush out)
            if self.viz_on and (iteration % self.viz_la_iter == 0):
                self.visualize_line()
                self.line_gather.flush()

            # evaluate metrics
            # if self.eval_metrics and (iteration % self.eval_metrics_iter == 0):
            #
            #     metric1, _ = self.eval_disentangle_metric1()
            #     metric2, _ = self.eval_disentangle_metric2()
            #
            #     prn_str = ( '********\n[iter %d (epoch %d)] ' + \
            #       'metric1 = %.4f, metric2 = %.4f\n********' ) % \
            #       (iteration, epoch, metric1, metric2)
            #     print(prn_str)
            #     if self.record_file:
            #         record = open(self.record_file, 'a')
            #         record.write('%s\n' % (prn_str,))
            #         record.close()
            #
            #     # (visdom) visulaize metrics
            #     if self.viz_on:
            #         self.visualize_line_metrics(iteration, metric1, metric2)
            #

    ####
    def eval_disentangle_metric1(self):

        # some hyperparams
        num_pairs = 800  # # data pairs (d,y) for majority vote classification
        bs = 50  # batch size
        nsamps_per_factor = 100  # samples per factor
        nsamps_agn_factor = 5000  # factor-agnostic samples

        self.set_mode(train=False)

        # 1) estimate variances of latent points factor agnostic

        dl = DataLoader(
            self.data_loader.dataset, batch_size=bs,
            shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        iterator = iter(dl)

        M = []
        for ib in range(int(nsamps_agn_factor / bs)):

            # sample a mini-batch
            XAb, XBb, _, _, _ = next(iterator)  # (bs x C x H x W)
            if self.use_cuda:
                XAb = XAb.cuda()
                XBb = XBb.cuda()

            # z = encA(xA)
            mu_infA, _, logvar_infA = self.encoderA(XAb)

            # z = encB(xB)
            mu_infB, _, logvar_infB = self.encoderB(XBb)

            # z = encAB(xA,xB) via POE
            mu_POE, _, _ = apply_poe(
                self.use_cuda, mu_infA, logvar_infA, mu_infB, logvar_infB,
            )

            mub = mu_POE

            M.append(mub.cpu().detach().numpy())

        M = np.concatenate(M, 0)

        # estimate sample vairance and mean of latent points for each dim
        vars_agn_factor = np.var(M, 0)

        # 2) estimatet dim-wise vars of latent points with "one factor fixed"

        factor_ids = range(0, len(self.latent_sizes))  # true factor ids
        vars_per_factor = np.zeros([num_pairs, self.z_dim])
        true_factor_ids = np.zeros(num_pairs, np.int)  # true factor ids

        # prepare data pairs for majority-vote classification
        i = 0
        for j in factor_ids:  # for each factor

            # repeat num_paris/num_factors times
            for r in range(int(num_pairs / len(factor_ids))):

                # a true factor (id and class value) to fix
                fac_id = j
                fac_class = np.random.randint(self.latent_sizes[fac_id])

                # randomly select images (with the fixed factor)
                indices = np.where(
                    self.latent_classes[:, fac_id] == fac_class)[0]
                np.random.shuffle(indices)
                idx = indices[:nsamps_per_factor]
                M = []
                for ib in range(int(nsamps_per_factor / bs)):
                    XAb, XBb, _, _, _ = dl.dataset[idx[(ib * bs):(ib + 1) * bs]]
                    if XAb.shape[0] < 1:  # no more samples
                        continue;
                    if self.use_cuda:
                        XAb = XAb.cuda()
                        XBb = XBb.cuda()
                    mu_infA, _, logvar_infA = self.encoderA(XAb)
                    mu_infB, _, logvar_infB = self.encoderB(XBb)
                    mu_POE, _, _ = apply_poe(self.use_cuda,
                                             mu_infA, logvar_infA, mu_infB, logvar_infB,
                                             )
                    mub = mu_POE
                    M.append(mub.cpu().detach().numpy())
                M = np.concatenate(M, 0)

                # estimate sample var and mean of latent points for each dim
                if M.shape[0] >= 2:
                    vars_per_factor[i, :] = np.var(M, 0)
                else:  # not enough samples to estimate variance
                    vars_per_factor[i, :] = 0.0

                    # true factor id (will become the class label)
                true_factor_ids[i] = fac_id

                i += 1

        # 3) evaluate majority vote classification accuracy

        # inputs in the paired data for classification
        smallest_var_dims = np.argmin(
            vars_per_factor / (vars_agn_factor + 1e-20), axis=1)

        # contingency table
        C = np.zeros([self.z_dim, len(factor_ids)])
        for i in range(num_pairs):
            C[smallest_var_dims[i], true_factor_ids[i]] += 1

        num_errs = 0  # # misclassifying errors of majority vote classifier
        for k in range(self.z_dim):
            num_errs += np.sum(C[k, :]) - np.max(C[k, :])

        metric1 = (num_pairs - num_errs) / num_pairs  # metric = accuracy

        self.set_mode(train=True)

        return metric1, C

    ####
    def eval_disentangle_metric2(self):

        # some hyperparams
        num_pairs = 800  # # data pairs (d,y) for majority vote classification
        bs = 50  # batch size
        nsamps_per_factor = 100  # samples per factor
        nsamps_agn_factor = 5000  # factor-agnostic samples

        self.set_mode(train=False)

        # 1) estimate variances of latent points factor agnostic

        dl = DataLoader(
            self.data_loader.dataset, batch_size=bs,
            shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        iterator = iter(dl)

        M = []
        for ib in range(int(nsamps_agn_factor / bs)):

            # sample a mini-batch
            XAb, XBb, _, _, _ = next(iterator)  # (bs x C x H x W)
            if self.use_cuda:
                XAb = XAb.cuda()
                XBb = XBb.cuda()

            # z = encA(xA)
            mu_infA, _, logvar_infA = self.encoderA(XAb)

            # z = encB(xB)
            mu_infB, _, logvar_infB = self.encoderB(XBb)

            # z = encAB(xA,xB) via POE
            mu_POE, _, _ = apply_poe(
                self.use_cuda, mu_infA, logvar_infA, mu_infB, logvar_infB,
            )

            mub = mu_POE

            M.append(mub.cpu().detach().numpy())

        M = np.concatenate(M, 0)

        # estimate sample vairance and mean of latent points for each dim
        vars_agn_factor = np.var(M, 0)

        # 2) estimatet dim-wise vars of latent points with "one factor varied"

        factor_ids = range(0, len(self.latent_sizes))  # true factor ids
        vars_per_factor = np.zeros([num_pairs, self.z_dim])
        true_factor_ids = np.zeros(num_pairs, np.int)  # true factor ids

        # prepare data pairs for majority-vote classification
        i = 0
        for j in factor_ids:  # for each factor

            # repeat num_paris/num_factors times
            for r in range(int(num_pairs / len(factor_ids))):

                # randomly choose true factors (id's and class values) to fix
                fac_ids = list(np.setdiff1d(factor_ids, j))
                fac_classes = \
                    [np.random.randint(self.latent_sizes[k]) for k in fac_ids]

                # randomly select images (with the other factors fixed)
                if len(fac_ids) > 1:
                    indices = np.where(
                        np.sum(self.latent_classes[:, fac_ids] == fac_classes, 1)
                        == len(fac_ids)
                    )[0]
                else:
                    indices = np.where(
                        self.latent_classes[:, fac_ids] == fac_classes
                    )[0]
                np.random.shuffle(indices)
                idx = indices[:nsamps_per_factor]
                M = []
                for ib in range(int(nsamps_per_factor / bs)):
                    XAb, XBb, _, _, _ = dl.dataset[idx[(ib * bs):(ib + 1) * bs]]
                    if XAb.shape[0] < 1:  # no more samples
                        continue;
                    if self.use_cuda:
                        XAb = XAb.cuda()
                        XBb = XBb.cuda()
                    mu_infA, _, logvar_infA = self.encoderA(XAb)
                    mu_infB, _, logvar_infB = self.encoderB(XBb)
                    mu_POE, _, _ = apply_poe(self.use_cuda,
                                             mu_infA, logvar_infA, mu_infB, logvar_infB,
                                             )
                    mub = mu_POE
                    M.append(mub.cpu().detach().numpy())
                M = np.concatenate(M, 0)

                # estimate sample var and mean of latent points for each dim
                if M.shape[0] >= 2:
                    vars_per_factor[i, :] = np.var(M, 0)
                else:  # not enough samples to estimate variance
                    vars_per_factor[i, :] = 0.0

                # true factor id (will become the class label)
                true_factor_ids[i] = j

                i += 1

        # 3) evaluate majority vote classification accuracy

        # inputs in the paired data for classification
        largest_var_dims = np.argmax(
            vars_per_factor / (vars_agn_factor + 1e-20), axis=1)

        # contingency table
        C = np.zeros([self.z_dim, len(factor_ids)])
        for i in range(num_pairs):
            C[largest_var_dims[i], true_factor_ids[i]] += 1

        num_errs = 0  # # misclassifying errors of majority vote classifier
        for k in range(self.z_dim):
            num_errs += np.sum(C[k, :]) - np.max(C[k, :])

        metric2 = (num_pairs - num_errs) / num_pairs  # metric = accuracy

        self.set_mode(train=True)

        return metric2, C

    def marginal_loglikelihood(self, X, X_recon, z_private, z_shared, mu, logvar, relaxedCateg):
        log_pz, _, _, log_q_zCx, _ = get_log_pz_qz_prodzi_qzCx(
            {'cont': z_private, 'disc': z_shared}, {'cont': (mu, logvar), 'disc': relaxedCateg},
            len(self.data_loader.dataset),
            is_mss=self.is_mss)
        p_xCz = - reconstruction_loss(X, X_recon.view(X.shape[0],-1,28,28), distribution="bernoulli") * X.shape[0]
        marginal_ll = (p_xCz + log_pz.sum() - log_q_zCx.sum()) / X.shape[0]
        return marginal_ll



    def get_loglikelihood(self, z_B_stat, train=True):
        self.set_mode(train=False)
        mkdirs(self.output_dir_recon)
        np.random.seed(0)
        if train:
            data_loader = self.data_loader
            fixed_idxs = np.random.randint(59999, size=600)
        else:
            data_loader = self.test_data_loader
            fixed_idxs = np.random.randint(5999, size=600)

        fixed_idxs100 = fixed_idxs

        XA = [0] * len(fixed_idxs100)
        XB = [0] * len(fixed_idxs100)
        label = [0] * len(fixed_idxs100)

        for i, idx in enumerate(fixed_idxs100):
            XA[i], XB[i], label[i] = \
                data_loader.dataset.__getitem__(idx)[0:3]

            if self.use_cuda:
                XA[i] = XA[i].cuda()
                XB[i] = XB[i].cuda()

        XA = torch.stack(XA)
        XB = torch.stack(XB)
        batch_size = XA.shape[0]

        if self.categ:
            muA_infA, stdA_infA, logvarA_infA, cate_prob_infA = self.encoderA(XA)
            # zB, zS = encB(xB)
            muB_infB, stdB_infB, logvarB_infB, cate_prob_infB = self.encoderB(XB)

            '''
            POE: should be the paramter for the distribution
            induce zS = encAB(xA,xB) via POE, that is,
                q(zA,zB,zS | xA,xB) := qI(zA|xA) * qT(zB|xB) * q(zS|xA,xB)
                    where q(zS|xA,xB) \propto p(zS) * qI(zS|xA) * qT(zS|xB)
            '''
            cate_prob_POE = cate_prob_infA * cate_prob_infB

            # encoder samples (for training)
            ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
            ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)

            # encoder samples (for cross-modal prediction)
            Eps = 1e-12
            # distribution
            relaxedCategA = ExpRelaxedCategorical(torch.tensor(.67), logits=torch.log(cate_prob_infA + Eps))
            relaxedCategB = ExpRelaxedCategorical(torch.tensor(.67), logits=torch.log(cate_prob_infB + Eps))
            relaxedCategS = ExpRelaxedCategorical(torch.tensor(.67), logits=torch.log(cate_prob_POE + Eps))

            # sampling
            log_ZS_infA = relaxedCategA.rsample()
            ZS_infA = torch.exp(log_ZS_infA)
            log_ZS_infB = relaxedCategB.rsample()
            ZS_infB = torch.exp(log_ZS_infB)
            log_ZS_POE = relaxedCategS.rsample()
            ZS_POE = torch.exp(log_ZS_POE)
            ZS_POE = torch.sqrt(ZS_POE)

        else:
            muA_infA, stdA_infA, logvarA_infA, \
            muS_infA, stdS_infA, logvarS_infA = self.encoderA(XA)

            # zB, zS = encB(xB)
            muB_infB, stdB_infB, logvarB_infB, \
            muS_infB, stdS_infB, logvarS_infB = self.encoderB(XB)

            # zS = encAB(xA,xB) via POE
            muS_POE, stdS_POE, logvarS_POE = apply_poe(
                self.use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB,
            )

            # encoder samples (for training)
            ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
            ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)
            ZS_POE = sample_gaussian(self.use_cuda, muS_POE, stdS_POE)
            # encoder samples (for cross-modal prediction)
            ZS_infA = sample_gaussian(self.use_cuda, muS_infA, stdS_infA)
            ZS_infB = sample_gaussian(self.use_cuda, muS_infB, stdS_infB)

        # reconstructed samples (given joint modal observation)
        XA_POE_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_POE)).view(XA.shape[0],-1,28,28)
        XB_POE_recon = self.decoderB(ZB_infB, ZS_POE)

        # reconstructed samples (given single modal observation)
        XA_infA_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_infA)).view(XA.shape[0],-1,28,28)
        XB_infB_recon = self.decoderB(ZB_infB, ZS_infB)

        #cross reconst
        XA_sinfB_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_infB)).view(XA.shape[0], -1, 28, 28)
        XB_sinfA_recon = self.decoderB(ZB_infB, ZS_infA)


        #cross synth
        ZB = torch.randn(batch_size, self.zB_dim)
        z_B_stat = np.array(z_B_stat)
        z_B_stat_mean = np.mean(z_B_stat, 0)
        ZB = ZB + torch.Tensor(z_B_stat_mean)
        if self.use_cuda:
            ZB = ZB.cuda()
        XB_synth = self.decoderB(ZB, ZS_infA)


        acc_infB = ((XB_infB_recon.argmax(dim=1) == XB).sum() / torch.FloatTensor([batch_size]))[0]
        acc_POE = ((XB_POE_recon.argmax(dim=1) == XB).sum() / torch.FloatTensor([batch_size]))[0]
        acc_sinfA = ((XB_sinfA_recon.argmax(dim=1) == XB).sum() / torch.FloatTensor([batch_size]))[0]
        acc_infA = ((XB_synth.argmax(dim=1) == XB).sum() / torch.FloatTensor([batch_size]))[0]

        marginal_ll_A_infA = self.marginal_loglikelihood(XA, XA_infA_recon, ZA_infA, ZS_infA, muA_infA, logvarA_infA, relaxedCategA)
        marginal_ll_A_poe = self.marginal_loglikelihood(XA, XA_POE_recon, ZA_infA, ZS_POE, muA_infA, logvarA_infA, relaxedCategS)
        marginal_ll_A_pAsB = self.marginal_loglikelihood(XA, XA_sinfB_recon, ZA_infA, ZS_infB, muA_infA, logvarA_infA, relaxedCategB)
        self.set_mode(train=True)
        return marginal_ll_A_infA, marginal_ll_A_poe, marginal_ll_A_pAsB, acc_infB, acc_POE, acc_sinfA, acc_infA


    def save_recon(self, iters, train=True):
        self.set_mode(train=False)

        mkdirs(self.output_dir_recon)

        if train:
            data_loader = self.data_loader
            fixed_idxs = []
            for i in range(10):
                fixed_idxs.append(5800 * i + 2005)
            out_dir = os.path.join(self.output_dir_recon, 'train')
        else:
            data_loader = self.test_data_loader
            fixed_idxs = [2, 982, 2300, 3400, 4500, 5500, 6500, 7500, 8500, 9500]
            out_dir = os.path.join(self.output_dir_recon, 'test')

        fixed_idxs100 = []
        for idx in fixed_idxs:
            for i in range(10):
                fixed_idxs100.append(idx + i)

        XA = [0] * len(fixed_idxs100)
        XB = [0] * len(fixed_idxs100)
        label = [0] * len(fixed_idxs100)

        for i, idx in enumerate(fixed_idxs100):
            XA[i], XB[i], label[i] = \
                data_loader.dataset.__getitem__(idx)[0:3]

            if self.use_cuda:
                XA[i] = XA[i].cuda()
                XB[i] = XB[i].cuda()

        XA = torch.stack(XA)
        XB = torch.stack(XB)
        batch_size = XA.shape[0]
        if self.categ:
            muA_infA, stdA_infA, logvarA_infA, cate_prob_infA = self.encoderA(XA)
            # zB, zS = encB(xB)
            muB_infB, stdB_infB, logvarB_infB, cate_prob_infB = self.encoderB(XB)

            '''
            POE: should be the paramter for the distribution
            induce zS = encAB(xA,xB) via POE, that is,
                q(zA,zB,zS | xA,xB) := qI(zA|xA) * qT(zB|xB) * q(zS|xA,xB)
                    where q(zS|xA,xB) \propto p(zS) * qI(zS|xA) * qT(zS|xB)
            '''
            cate_prob_POE = cate_prob_infA * cate_prob_infB

            # encoder samples (for training)
            ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
            ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)

            ZS_infA = sample_gumbel_softmax(self.use_cuda, cate_prob_infA, train=False)
            ZS_infB = sample_gumbel_softmax(self.use_cuda, cate_prob_infB, train=False)
            ZS_POE = sample_gumbel_softmax(self.use_cuda, cate_prob_POE, train=False)
            ZS_POE = torch.sqrt(ZS_POE)

        else:
            muA_infA, stdA_infA, logvarA_infA, \
            muS_infA, stdS_infA, logvarS_infA = self.encoderA(XA)

            # zB, zS = encB(xB)
            muB_infB, stdB_infB, logvarB_infB, \
            muS_infB, stdS_infB, logvarS_infB = self.encoderB(XB)

            # zS = encAB(xA,xB) via POE
            muS_POE, stdS_POE, logvarS_POE = apply_poe(
                self.use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB,
            )

            # encoder samples (for training)
            ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
            ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)
            ZS_POE = sample_gaussian(self.use_cuda, muS_POE, stdS_POE)
            # encoder samples (for cross-modal prediction)
            ZS_infA = sample_gaussian(self.use_cuda, muS_infA, stdS_infA)
            ZS_infB = sample_gaussian(self.use_cuda, muS_infB, stdS_infB)




        # reconstructed samples (given joint modal observation)
        XA_POE_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_POE)).view(XA.shape[0],-1,28,28)
        XB_POE_recon = self.decoderB(ZB_infB, ZS_POE)

        # reconstructed samples (given single modal observation)
        XA_infA_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_infA)).view(XA.shape[0],-1,28,28)
        XB_infB_recon = self.decoderB(ZB_infB, ZS_infB)

        #cross reconst
        XA_sinfB_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_infB)).view(XA.shape[0], -1, 28, 28)
        XB_sinfA_recon = self.decoderB(ZB_infB, ZS_infA)

        WS = torch.ones(XA.shape)
        if self.use_cuda:
            WS = WS.cuda()

        imgs = [XA, XA_infA_recon, XA_POE_recon, XA_sinfB_recon, WS]
        merged = torch.cat(
            imgs, dim=0
        )


        perm = torch.arange(0, len(imgs) * batch_size).view(len(imgs), batch_size).transpose(1, 0)
        perm = perm.contiguous().view(-1)
        merged = merged[perm, :].cpu()

        # save the results as image
        fname = os.path.join(out_dir, 'reconA_%s.jpg' % iters)
        mkdirs(out_dir)
        save_image(
            tensor=merged, filename=fname, nrow=len(imgs) * int(np.sqrt(batch_size)),
            pad_value=1
        )

        self.set_mode(train=True)

    ####
    def save_synth_pure(self, iters, howmany=100):

        self.set_mode(train=False)

        decoderA = self.decoderA
        decoderB = self.decoderB

        Z = torch.randn(howmany, self.z_dim)
        if self.use_cuda:
            Z = Z.cuda()

        # do synthesis
        XA = torch.sigmoid(decoderA(Z)).data
        XB = torch.sigmoid(decoderB(Z)).data

        WS = torch.ones(XA.shape)
        if self.use_cuda:
            WS = WS.cuda()

        perm = torch.arange(0, 3 * howmany).view(3, howmany).transpose(1, 0)
        perm = perm.contiguous().view(-1)
        merged = torch.cat([XA, XB, WS], dim=0)
        merged = merged[perm, :].cpu()

        # save the results as image
        fname = os.path.join(
            self.output_dir_synth, 'synth_pure_%s.jpg' % iters
        )
        mkdirs(self.output_dir_synth)
        save_image(
            tensor=merged, filename=fname, nrow=3 * int(np.sqrt(howmany)),
            pad_value=1
        )

        self.set_mode(train=True)

    ####
    def save_synth_cross_modal(self, iters, z_A_stat, z_B_stat, train=True, howmany=3, save_img=True):

        self.set_mode(train=False)

        if train:
            data_loader = self.data_loader
            fixed_idxs = [1, 3, 5, 7, 2, 11, 13, 15, 17, 19]
        else:
            data_loader = self.test_data_loader
            fixed_idxs = [3, 2, 1, 18, 4, 15, 11, 17, 61, 99]


        fixed_XA = [0] * len(fixed_idxs)
        fixed_XB = [0] * len(fixed_idxs)
        label = [0] * len(fixed_idxs)
        for i, idx in enumerate(fixed_idxs):
            fixed_XA[i], fixed_XB[i], label[i] = \
                data_loader.dataset.__getitem__(idx)[0:3]

            if self.use_cuda:
                fixed_XA[i] = fixed_XA[i].cuda()
                fixed_XB[i] = fixed_XB[i].cuda()
        fixed_XA = torch.stack(fixed_XA)
        fixed_XB = torch.stack(fixed_XB)
        label = torch.LongTensor(label)
        if self.use_cuda:
            label = label.cuda()


        if self.categ:
            _, _, _, cate_prob_infA = self.encoderA(fixed_XA)
            _, _, _, cate_prob_infB = self.encoderB(fixed_XB)
            ZS_infA = sample_gumbel_softmax(self.use_cuda, cate_prob_infA, train=False)
            ZS_infB = sample_gumbel_softmax(self.use_cuda, cate_prob_infB, train=False)
        else:
            _, _, _, muS_infA, stdS_infA, _ = self.encoderA(fixed_XA)
            _, _, _, muS_infB, stdS_infB, _ = self.encoderB(fixed_XB)
            ZS_infA = sample_gaussian(self.use_cuda, muS_infA, stdS_infA)
            ZS_infB = sample_gaussian(self.use_cuda, muS_infB, stdS_infB)

        if self.use_cuda:
            ZS_infA = ZS_infA.cuda()
            ZS_infB = ZS_infB.cuda()

        decoderA = self.decoderA
        decoderB = self.decoderB

        # mkdirs(os.path.join(self.output_dir_synth, str(iters)))

        n = len(fixed_idxs)

        ######## 1) generate xB from given xA (A2B) ########

        XB_synth_list = []
        label_list = []

        for k in range(howmany):
            ZB = torch.randn(n, self.zB_dim)
            z_B_stat = np.array(z_B_stat)
            z_B_stat_mean = np.mean(z_B_stat, 0)
            ZB = ZB + torch.Tensor(z_B_stat_mean)

            if self.use_cuda:
                ZB = ZB.cuda()
            XB_synth = decoderB(ZB, ZS_infA)  # given XA
            XB_synth_list.extend(XB_synth)
            label_list.extend(label)
        XB_synth_list = torch.stack(XB_synth_list)
        label_list = torch.stack(label_list)

        if train:
            fname = os.path.join(
                self.output_dir_synth,
                'synth_cross_modal_A2B_%s.txt' % iters
            )
        else:
            fname = os.path.join(
                self.output_dir_synth,
                'eval_synth_cross_modal_A2B_%s.txt' % iters
            )
        mkdirs(self.output_dir_synth)
        file1 = open(fname, "w")
        for i in range(len(XB_synth_list)):
            file1.write('Text (%d): %s\n' % (label_list[i], XB_synth_list[i].argmax()))
        acc = (XB_synth_list.argmax(1) == label_list).sum() / torch.FloatTensor([label_list.shape[0]])[0]

        file1.write(str(acc))
        file1.close()


        ######## 2) generate xA from given xB (B2A) ########
        WS = torch.ones(fixed_XA.shape)
        if self.use_cuda:
            WS = WS.cuda()

        perm = torch.arange(0, (howmany + 2) * n).view(howmany + 2, n).transpose(1, 0)
        perm = perm.contiguous().view(-1)

        merged = torch.cat([fixed_XA], dim=0)
        XA_synth_list = []
        for k in range(howmany):
            ZA = torch.randn(n, self.zA_dim)
            z_A_stat = np.array(z_A_stat)
            z_A_stat_mean = np.mean(z_A_stat, 0)
            ZA = ZA + torch.Tensor(z_A_stat_mean)

            if self.use_cuda:
                ZA = ZA.cuda()
            XA_synth = torch.sigmoid(decoderA(ZA, ZS_infB)).view(ZA.shape[0], -1, 28, 28)  # given XB
            XA_synth_list.extend(XA_synth)
            merged = torch.cat([merged, XA_synth], dim=0)
        merged = torch.cat([merged, WS], dim=0)
        merged = merged[perm, :].cpu()

        # save the results as image
        if train:
            fname = os.path.join(
                self.output_dir_synth,
                'synth_cross_modal_B2A_%s.jpg' % iters
            )
        else:
            fname = os.path.join(
                self.output_dir_synth,
                'eval_synth_cross_modal_B2A_%s.jpg' % iters
            )
        mkdirs(self.output_dir_synth)
        save_image(
            tensor=merged, filename=fname, nrow=(howmany + 2) * int(np.sqrt(n)),
            pad_value=1
        )
        self.set_mode(train=True)



    def acc_total(self, z_A_stat, z_B_stat, train=True, howmany=3):

        self.set_mode(train=False)

        if train:
            data_loader = self.data_loader
            fixed_idxs = [0, 5923, 11923, 17881, 23881, 29723, 35144, 41062, 47062, 52913]
        else:
            data_loader = self.test_data_loader
            fixed_idxs = [0, 980, 1980, 2980, 3980, 4962, 5854, 6812, 7812, 8786]


        # increase into 20 times more data
        fixed_idxs1000 = []

        for i in range(10):
            for j in range(100):
                fixed_idxs1000.append(fixed_idxs[i] + j)
        fixed_XA = [0] * len(fixed_idxs1000)
        fixed_XB = [0] * len(fixed_idxs1000)
        label = [0] * len(fixed_idxs1000)

        for i, idx in enumerate(fixed_idxs1000):
            fixed_XA[i], fixed_XB[i], label[i] = \
                data_loader.dataset.__getitem__(idx)[0:3]
            if self.use_cuda:
                fixed_XA[i] = fixed_XA[i].cuda()
                fixed_XB[i] = fixed_XB[i].cuda()
        n = len(fixed_idxs1000)
        # check if each digit was selected.
        cnt = [0] * 10
        for l in label:
            cnt[l] += 1
        print('cnt of digit:')
        print(cnt)

        fixed_XA = torch.stack(fixed_XA)
        fixed_XB = torch.stack(fixed_XB)
        label = torch.LongTensor(label)

        if self.use_cuda:
            label = label.cuda()

        muA_infA, stdA_infA, logvarA_infA, cate_prob_infA = self.encoderA(fixed_XA)
        muB_infB, stdB_infB, logvarB_infB, cate_prob_infB = self.encoderB(fixed_XB)

        ################### ACC for reconstructed img

        # zS = encAB(xA,xB) via POE
        cate_prob_POE = torch.tensor(1 / 10) * cate_prob_infA * cate_prob_infB

        # encoder samples (for training)
        ZA_infA = sample_gaussian(self.use_cuda, muA_infA, stdA_infA)
        ZB_infB = sample_gaussian(self.use_cuda, muB_infB, stdB_infB)

        # encoder samples (for cross-modal prediction)
        ZS_infA = sample_gumbel_softmax(self.use_cuda, cate_prob_infA, train=False)
        ZS_infB = sample_gumbel_softmax(self.use_cuda, cate_prob_infB, train=False)
        ZS_POE = sample_gumbel_softmax(self.use_cuda, cate_prob_POE, train=False)


        # reconstructed samples (given joint modal observation)
        XA_POE_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_POE))
        XB_POE_recon = torch.sigmoid(self.decoderB(ZB_infB, ZS_POE))

        # reconstructed samples (given single modal observation)
        XA_infA_recon = torch.sigmoid(self.decoderA(ZA_infA, ZS_infA))
        XB_infB_recon = torch.sigmoid(self.decoderB(ZB_infB, ZS_infB))

        print('=========== Reconstructed ACC  ============')
        print('PoeA')
        poeA_acc = self.check_acc(XA_POE_recon, label, train=train)
        print('InfA')
        infA_acc = self.check_acc(XA_infA_recon, label, train=train)
        print('PoeB')
        poeB_acc = self.check_acc(XB_POE_recon, label, dataset='fmnist', train=train)
        print('InfB')
        infB_acc = self.check_acc(XB_infB_recon, label, dataset='fmnist', train=train)

        print('=========== Acc by discrete variable  ============')
        pred_ZS_infA = torch.argmax(ZS_infA, dim=1)
        pred_ZS_infB = torch.argmax(ZS_infB, dim=1)
        pred_ZS_POE = torch.argmax(ZS_POE, dim=1)

        acc_ZS_infA = pred_ZS_infA.eq(label.view_as(pred_ZS_infA)).sum().item() / n
        acc_ZS_infB = pred_ZS_infB.eq(label.view_as(pred_ZS_infB)).sum().item() / n
        acc_ZS_POE = pred_ZS_POE.eq(label.view_as(pred_ZS_POE)).sum().item() / n

        ############################################################################
        ################### ACC for synthesized img ###################
        ############################################################################

        ######## 1) generate xB from given xA (A2B) ########
        XB_synth_list = []
        label_list = []
        for k in range(howmany):
            ZB = torch.randn(n, self.zB_dim)
            z_B_stat = np.array(z_B_stat)
            z_B_stat_mean = np.mean(z_B_stat, 0)
            ZB = ZB + torch.Tensor(z_B_stat_mean)

            if self.use_cuda:
                ZB = ZB.cuda()
            XB_synth = torch.sigmoid(self.decoderB(ZB, ZS_infA))  # given XA
            XB_synth_list.extend(XB_synth)
            label_list.extend(label)
        print('=========== cross-synth ACC for XB_synth ============')
        XB_synth_list = torch.stack(XB_synth_list)
        label_list = torch.LongTensor(label_list)
        if self.use_cuda:
            label_list = label_list.cuda()

        synB_acc = self.check_acc(XB_synth_list, label_list, dataset='fmnist', train=train)


        ######## 2) generate xA from given xB (B2A) ########
        XA_synth_list = []
        for k in range(howmany):
            ZA = torch.randn(n, self.zA_dim)
            z_A_stat = np.array(z_A_stat)
            z_A_stat_mean = np.mean(z_A_stat, 0)
            ZA = ZA + torch.Tensor(z_A_stat_mean)

            if self.use_cuda:
                ZA = ZA.cuda()
            XA_synth = torch.sigmoid(self.decoderA(ZA, ZS_infB))  # given XB
            XA_synth_list.extend(XA_synth)
        print('=========== cross-synth ACC for XA_synth ============')
        XA_synth_list = torch.stack(XA_synth_list)
        synA_acc = self.check_acc(XA_synth_list, label_list, train=train)

        self.set_mode(train=True)
        return (synA_acc, synB_acc, poeA_acc, poeB_acc, infA_acc, infB_acc, acc_ZS_infA, acc_ZS_infB, acc_ZS_POE)

    def get_stat(self):
        z_A, z_B, z_AS, z_BS, z_S = [], [], [], [], []
        for _ in range(10000):
            rand_i = np.random.randint(self.N)
            random_XA, random_XB = self.data_loader.dataset.__getitem__(rand_i)[0:2]
            if self.use_cuda:
                random_XA = random_XA.cuda()
                random_XB = random_XB.cuda()
            random_XA = random_XA.unsqueeze(0)
            random_XB = random_XB.unsqueeze(0)

            if self.categ:
                muA_infA, _, _, cate_prob_infA = self.encoderA(random_XA)
                muB_infB, _, _, cate_prob_infB = self.encoderB(random_XB)
                # cate_prob_POE = torch.tensor(1 / self.zS_dim) * cate_prob_infA * cate_prob_infB
                # fixed_zS = sample_gumbel_softmax(self.use_cuda, cate_prob_POE, train=False)
            else:
                muA_infA, _, _, muS_infA, _, logvarS_infA = self.encoderA(random_XA)
                muB_infB, _, _, muS_infB, _, logvarS_infB = self.encoderB(random_XB)
                muS_POE, _, _ = apply_poe(
                    self.use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB,
                )
                z_AS.append(muS_infA.cpu().detach().numpy()[0])
                z_BS.append(muS_infB.cpu().detach().numpy()[0])
                z_S.append(muS_POE.cpu().detach().numpy()[0])
            z_A.append(muA_infA.cpu().detach().numpy()[0])
            z_B.append(muB_infB.cpu().detach().numpy()[0])

        return z_A, z_B, z_AS, z_BS, z_S


    def save_traverseA(self, iters, loc=-1, train=True):

        self.set_mode(train=False)

        encoderA = self.encoderA
        decoderA = self.decoderA
        interpolationA = torch.tensor(np.linspace(-3, 3, self.zS_dim))

        if self.record_file:
            ####
            if train:
                data_loader = self.data_loader
                fixed_idxs = [1,3,5,7,2,11,13,15,17,19]
                out_dir = os.path.join(self.output_dir_trvsl, str(iters), 'trainA')
            else:
                data_loader = self.test_data_loader
                fixed_idxs = [3,2,1,18,4,15,11,17,61,99]
                out_dir = os.path.join(self.output_dir_trvsl, str(iters), 'testA')

            fixed_XA = [0] * len(fixed_idxs)
            label = [0] * len(fixed_idxs)

            for i, idx in enumerate(fixed_idxs):

                fixed_XA[i], _, label[i] = \
                    data_loader.dataset.__getitem__(idx)[0:3]
                if self.use_cuda:
                    fixed_XA[i] = fixed_XA[i].cuda()
                fixed_XA[i] = fixed_XA[i].unsqueeze(0)

            fixed_XA = torch.cat(fixed_XA, dim=0)
            # cnt = [0] * 10
            # for l in label:
            #     cnt[l] += 1
            # print('cnt of digit:')
            # print(cnt)

            fixed_zmuA, _, _, cate_prob_infA = encoderA(fixed_XA)

            # fixed_zS = sample_gumbel_softmax(self.use_cuda, fixed_cate_probS, train=False)
            fixed_zS = sample_gumbel_softmax(self.use_cuda, cate_prob_infA, train=False)


            saving_shape=torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1).shape

        ####

        WS = torch.ones(saving_shape)
        if self.use_cuda:
            WS = WS.cuda()

        # do traversal and collect generated images
        zA_ori, zS_ori = fixed_zmuA, fixed_zS

        tempA = [] # zA_dim + zS_dim , num_trv, 1, 32*num_samples, 32
        for row in range(self.zA_dim):
            if loc != -1 and row != loc:
                continue
            zA = zA_ori.clone()

            temp = []
            for val in interpolationA:
                zA[:, row] = val
                sampleA = torch.sigmoid(decoderA(zA, zS_ori)).data
                sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

            tempA.append(torch.cat(temp, dim=0).unsqueeze(0)) # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

        temp = []
        for i in range(self.zS_dim):
            zS = np.zeros((1, self.zS_dim))
            zS[0, i % self.zS_dim] = 1.
            zS = torch.Tensor(zS)
            zS = torch.cat([zS] * len(fixed_idxs), dim=0)

            if self.use_cuda:
                zS = zS.cuda()

            sampleA = torch.sigmoid(decoderA(zA_ori, zS)).data
            sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs = torch.cat(tempA, dim=0) #torch.Size([11, 10, 1, 384, 32])


        # save the generated files, also the animated gifs

        mkdirs(self.output_dir_trvsl)
        mkdirs(out_dir)

        for j, val in enumerate(interpolationA):
            # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
            I = gifs[:,j]
            save_image(
                tensor=I.cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + self.zA_dim + 1 + 1 + 1 + self.zB_dim,
                pad_value=1)
            # make animated gif
        grid2gif2(
            out_dir, str(os.path.join(out_dir, 'mnist_traverse' + '.gif')), delay=10
        )

        self.set_mode(train=True)



    ###
    def save_traverseB(self, iters, z_A, z_B, loc=-1):

        self.set_mode(train=False)

        encoderB = self.encoderB
        decoderB = self.decoderB
        interpolationA = torch.tensor(np.linspace(-3, 3, self.zS_dim))

        print('------------ traverse interpolation ------------')
        print('interpolationA: ', np.min(np.array(z_A)), np.max(np.array(z_A)))
        print('interpolationB: ', np.min(np.array(z_B)), np.max(np.array(z_B)))

        if self.record_file:
            ####
            fixed_idxs = [3246, 7001, 14305, 19000, 27444, 33100, 38000, 45231, 51000, 55121]

            fixed_XA = [0] * len(fixed_idxs)
            fixed_XB = [0] * len(fixed_idxs)

            for i, idx in enumerate(fixed_idxs):

                fixed_XA[i], fixed_XB[i] = \
                    self.data_loader.dataset.__getitem__(idx)[0:2]
                if self.use_cuda:
                    fixed_XA[i] = fixed_XA[i].cuda()
                    fixed_XB[i] = fixed_XB[i].cuda()
                fixed_XA[i] = fixed_XA[i].unsqueeze(0)
                fixed_XB[i] = fixed_XB[i].unsqueeze(0)

            fixed_XA = torch.cat(fixed_XA, dim=0)
            fixed_XB = torch.cat(fixed_XB, dim=0)


            # zB, zS = encB(xB)
            fixed_zmuB, _, _, cate_prob_infB = encoderB(fixed_XB)


            # fixed_zS = sample_gumbel_softmax(self.use_cuda, fixed_cate_probS, train=False)
            fixed_zS = sample_gumbel_softmax(self.use_cuda, cate_prob_infB, train=False)


            saving_shape=torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1).shape

        ####

        WS = torch.ones(saving_shape)
        if self.use_cuda:
            WS = WS.cuda()

        # do traversal and collect generated images
        gifs = []

        zB_ori, zS_ori = fixed_zmuB, fixed_zS

        tempB = [] # zA_dim + zS_dim , num_trv, 1, 32*num_samples, 32
        for row in range(self.zB_dim):
            if loc != -1 and row != loc:
                continue
            zB = zB_ori.clone()

            temp = []
            for val in interpolationA:
                zB[:, row] = val
                sampleB = torch.sigmoid(decoderB(zB, zS_ori)).data
                temp.append((torch.cat([sampleB[i] for i in range(sampleB.shape[0])], dim=1)).unsqueeze(0))

            tempB.append(torch.cat(temp, dim=0).unsqueeze(0)) # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

        temp = []
        for i in range(self.zS_dim):
            zS = np.zeros((1, self.zS_dim))
            zS[0, i % self.zS_dim] = 1.
            zS = torch.Tensor(zS)
            zS = torch.cat([zS] * len(fixed_idxs), dim=0)

            if self.use_cuda:
                zS = zS.cuda()

            sampleB = torch.sigmoid(decoderB(zB_ori, zS)).data
            temp.append((torch.cat([sampleB[i] for i in range(sampleB.shape[0])], dim=1)).unsqueeze(0))
        tempB.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs = torch.cat(tempB, dim=0) #torch.Size([11, 10, 1, 384, 32])


        # save the generated files, also the animated gifs
        out_dir = os.path.join(self.output_dir_trvsl, str(iters), 'trainB')
        mkdirs(self.output_dir_trvsl)
        mkdirs(out_dir)

        for j, val in enumerate(interpolationA):
            # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
            I = gifs[:,j]
            save_image(
                tensor=I.cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + self.zA_dim + 1 + 1 + 1 + self.zB_dim,
                pad_value=1)
            # make animated gif
        grid2gif2(
            out_dir, str(os.path.join(out_dir, 'fmnist_traverse' + '.gif')), delay=10
        )

        self.set_mode(train=True)

    ###
    def save_traverse(self, iters, z_A, z_B, loc=-1, train=True):

        self.set_mode(train=False)

        encoderA = self.encoderA
        encoderB = self.encoderB
        decoderA = self.decoderA
        decoderB = self.decoderB

        if self.categ:
            interpolation = torch.tensor(np.linspace(-3, 3, self.zS_dim))
        else:
            interpolation = torch.tensor(np.linspace(-3, 3, 10))

        print('------------ traverse interpolation ------------')
        print('interpolationA: ', np.min(np.array(z_A)), np.max(np.array(z_A)))
        print('interpolationB: ', np.min(np.array(z_B)), np.max(np.array(z_B)))
        if train:
            data_loader = self.data_loader
            fixed_idxs = []
            for i in range(10):
                fixed_idxs.append(5800 * i + 2005)
            out_dir = os.path.join(self.output_dir_trvsl, str(iters), 'train')
        else:
            data_loader = self.test_data_loader
            fixed_idxs = [2, 982, 2300, 3400, 4500, 5500, 6500, 7500, 8500, 9500]
            out_dir = os.path.join(self.output_dir_trvsl, str(iters), 'test')

        print('>>>>>>>>>fixed_idxs: ', fixed_idxs)
        if self.record_file:
            ####

            fixed_XA = [0] * len(fixed_idxs)
            fixed_XB = [0] * len(fixed_idxs)

            for i, idx in enumerate(fixed_idxs):

                fixed_XA[i], fixed_XB[i] = \
                    data_loader.dataset.__getitem__(idx)[0:2]
                if self.use_cuda:
                    fixed_XA[i] = fixed_XA[i].cuda()
                    fixed_XB[i] = fixed_XB[i].cuda()
                fixed_XA[i] = fixed_XA[i].unsqueeze(0)
                fixed_XB[i] = fixed_XB[i].unsqueeze(0)

            fixed_XA = torch.cat(fixed_XA, dim=0)
            fixed_XB = torch.cat(fixed_XB, dim=0)


        if self.categ:
            fixed_zmuA, _, _, cate_prob_infA = encoderA(fixed_XA)
            fixed_zmuB, _, _, cate_prob_infB = encoderB(fixed_XB)
            fixed_cate_probS = cate_prob_infA * cate_prob_infB
            fixed_zS = sample_gumbel_softmax(self.use_cuda, fixed_cate_probS, train=False)
            fixed_zS = torch.sqrt(fixed_zS)

        else:
            fixed_zmuA, _, _, \
            muS_infA, stdS_infA, logvarS_infA = encoderA(fixed_XA)
            fixed_zmuB, _, _, \
            muS_infB, stdS_infB, logvarS_infB = encoderB(fixed_XB)
            fixed_zS, _, _ = apply_poe(
                self.use_cuda,
                muS_infA, logvarS_infA, muS_infB, logvarS_infB
            )


        ####

        # do traversal and collect generated images
        zA_ori, zB_ori, zS_ori = fixed_zmuA, fixed_zmuB, fixed_zS


        tempAll = [] # zA_dim + zS_dim , num_trv, 1, 32*num_samples, 32
        ###A_private
        for row in range(self.zA_dim):
            if loc != -1 and row != loc:
                continue
            zA = zA_ori.clone()

            temp = []
            for val in interpolation:
                zA[:, row] = val
                sampleA = torch.sigmoid(decoderA(zA, zS_ori)).data
                sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

            tempAll.append(torch.cat(temp, dim=0).unsqueeze(0)) # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32
        ###shared
        if self.categ:
            #A
            temp = []
            for i in range(self.zS_dim):
                zS = np.zeros((1, self.zS_dim))
                zS[0, i % self.zS_dim] = 1.
                zS = torch.Tensor(zS)
                zS = torch.cat([zS] * len(fixed_idxs), dim=0)

                if self.use_cuda:
                    zS = zS.cuda()

                sampleA = torch.sigmoid(decoderA(zA_ori, zS)).data
                sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
            tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))
            #B
            temp = []
            for i in range(self.zS_dim):
                zS = np.zeros((1, self.zS_dim))
                zS[0, i % self.zS_dim] = 1.
                zS = torch.Tensor(zS)
                zS = torch.cat([zS] * len(fixed_idxs), dim=0)

                if self.use_cuda:
                    zS = zS.cuda()

                sampleB = torch.sigmoid(decoderB(zB_ori, zS)).data
                sampleB = sampleB.view(sampleB.shape[0], -1, 28, 28)
                temp.append((torch.cat([sampleB[i] for i in range(sampleB.shape[0])], dim=1)).unsqueeze(0))
            tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))
        else:
            #A
            for row in range(self.zS_dim):
                if loc != -1 and row != loc:
                    continue
                zS = zS_ori.clone()

                temp = []
                for val in interpolation:
                    zS[:, row] = val
                    sampleA = torch.sigmoid(decoderA(zA_ori, zS)).data
                    sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
                    temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
                tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))
            #B
            for row in range(self.zS_dim):
                if loc != -1 and row != loc:
                    continue
                zS = zS_ori.clone()

                temp = []
                for val in interpolation:
                    zS[:, row] = val
                    sampleB = torch.sigmoid(decoderB(zB_ori, zS)).data
                    sampleB = sampleB.view(sampleB.shape[0], -1, 28, 28)
                    temp.append((torch.cat([sampleB[i] for i in range(sampleB.shape[0])], dim=1)).unsqueeze(0))
                tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))


        ###B_private
        for row in range(self.zB_dim):
            if loc != -1 and row != loc:
                continue
            zB = zB_ori.clone()

            temp = []
            for val in interpolation:
                zB[:, row] = val
                sampleB = torch.sigmoid(decoderB(zB, zS_ori)).data
                sampleB = sampleB.view(sampleB.shape[0], -1, 28, 28)
                temp.append((torch.cat([sampleB[i] for i in range(sampleB.shape[0])], dim=1)).unsqueeze(0))

            tempAll.append(torch.cat(temp, dim=0).unsqueeze(0)) # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32


        gifs = torch.cat(tempAll, dim=0) #torch.Size([11, 10, 1, 384, 32])


        # save the generated files, also the animated gifs
        mkdirs(out_dir)

        for j, val in enumerate(interpolation):
            # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
            I = gifs[:,j]
            if self.categ:
                save_image(
                    tensor=I.cpu(),
                    filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                    nrow=1 + self.zA_dim + 1 + self.zS_dim + 1 + self.zB_dim,
                    pad_value=1)
            else:
                save_image(
                    tensor=I.cpu(),
                    filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                    nrow=1 + self.zA_dim + 1 + 1 + 1 + self.zB_dim,
                    pad_value=1)
            # make animated gif
        grid2gif2(
            out_dir, str(os.path.join(out_dir, 'both_traverse' + '.gif')), delay=10
        )

        self.set_mode(train=True)

    ####
    def viz_init(self):

        self.viz.close(env=self.name + '/lines', win=self.win_id['recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['kl'])
        # self.viz.close(env=self.name + '/lines', win=self.win_id['acc'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['tc'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['mi'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['dw_kl'])

        self.viz.close(env=self.name + '/lines', win=self.win_id['mgll'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['acc'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['mgll_te'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['acc_te'])

        # if self.eval_metrics:
        #     self.viz.close(env=self.name+'/lines', win=self.win_id['metrics'])

    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        recon_both = torch.Tensor(data['recon_both'])
        recon_A = torch.Tensor(data['recon_A'])
        recon_B = torch.Tensor(data['recon_B'])
        kl_A = torch.Tensor(data['kl_A'])
        kl_B = torch.Tensor(data['kl_B'])
        kl_POE = torch.Tensor(data['kl_POE'])

        # poeA_acc = torch.Tensor(data['poeA_acc'])
        # infA_acc = torch.Tensor(data['infA_acc'])
        # synA_acc = torch.Tensor(data['synA_acc'])
        # poeB_acc = torch.Tensor(data['poeB_acc'])
        # infB_acc = torch.Tensor(data['infB_acc'])
        # synB_acc = torch.Tensor(data['synB_acc'])

        # acc_ZS_infA = torch.Tensor(data['acc_ZS_infA'])
        # acc_ZS_infB = torch.Tensor(data['acc_ZS_infB'])
        # acc_ZS_POE = torch.Tensor(data['acc_ZS_POE'])

        tc_loss = torch.Tensor(data['tc_loss'])
        mi_loss =  torch.Tensor(data['mi_loss'])
        dw_kl_loss =  torch.Tensor(data['dw_kl_loss'])

        tc_loss_A = torch.Tensor(data['tc_loss_A'])
        mi_loss_A =  torch.Tensor(data['mi_loss_A'])
        dw_kl_loss_A =  torch.Tensor(data['dw_kl_loss_A'])
        tc_loss_B = torch.Tensor(data['tc_loss_B'])
        mi_loss_B =  torch.Tensor(data['mi_loss_B'])
        dw_kl_loss_B =  torch.Tensor(data['dw_kl_loss_B'])
        tc_loss_POEA = torch.Tensor(data['tc_loss_POEA'])
        mi_loss_POEA =  torch.Tensor(data['mi_loss_POEA'])
        dw_kl_loss_POEA =  torch.Tensor(data['dw_kl_loss_POEA'])
        tc_loss_POEB = torch.Tensor(data['tc_loss_POEB'])
        mi_loss_POEB =  torch.Tensor(data['mi_loss_POEB'])
        dw_kl_loss_POEB =  torch.Tensor(data['dw_kl_loss_POEB'])

        marginal_ll_A_infA = torch.Tensor(data['marginal_ll_A_infA'])
        marginal_ll_A_poe = torch.Tensor(data['marginal_ll_A_poe'])
        marginal_ll_A_pAsB = torch.Tensor(data['marginal_ll_A_pAsB'])
        acc_infB = torch.Tensor(data['acc_infB'])
        acc_POE = torch.Tensor(data['acc_POE'])
        acc_sinfA = torch.Tensor(data['acc_sinfA'])
        acc_infA = torch.Tensor(data['acc_infA'])
        marginal_ll_A_infA_te = torch.Tensor(data['marginal_ll_A_infA_te'])
        marginal_ll_A_poe_te = torch.Tensor(data['marginal_ll_A_poe_te'])
        marginal_ll_A_pAsB_te = torch.Tensor(data['marginal_ll_A_pAsB_te'])
        acc_infB_te = torch.Tensor(data['acc_infB_te'])
        acc_POE_te = torch.Tensor(data['acc_POE_te'])
        acc_sinfA_te = torch.Tensor(data['acc_sinfA_te'])
        acc_infA_te = torch.Tensor(data['acc_infA_te'])

        recons = torch.stack(
            [recon_both.detach(), recon_A.detach(), recon_B.detach()], -1
        )
        kls = torch.stack(
            [kl_A.detach(), kl_B.detach(), kl_POE.detach()], -1
        )
        tc = torch.stack(
            [tc_loss.detach(), tc_loss_A.detach(), tc_loss_B.detach(), tc_loss_POEA.detach(), tc_loss_POEB.detach()], -1
        )

        mi = torch.stack(
            [mi_loss.detach(), mi_loss_A.detach(), mi_loss_B.detach(), mi_loss_POEA.detach(), mi_loss_POEB.detach()], -1
        )

        dw_kl = torch.stack(
            [dw_kl_loss.detach(), dw_kl_loss_A.detach(), dw_kl_loss_B.detach(), dw_kl_loss_POEA.detach(), dw_kl_loss_POEB.detach()], -1
        )

        acc = torch.stack(
            [acc_infB.detach(), acc_POE.detach(), acc_sinfA.detach(), acc_infA.detach()], -1
        )

        mgll = torch.stack(
            [marginal_ll_A_infA.detach(), marginal_ll_A_poe.detach(), marginal_ll_A_pAsB.detach()], -1
        )

        acc_te = torch.stack(
            [acc_infB_te.detach(), acc_POE_te.detach(), acc_sinfA_te.detach(), acc_infA_te.detach()], -1
        )

        mgll_te = torch.stack(
            [marginal_ll_A_infA_te.detach(), marginal_ll_A_poe_te.detach(), marginal_ll_A_pAsB_te.detach()], -1
        )

        self.viz.line(
            X=iters, Y=recons, env=self.name + '/lines',
            win=self.win_id['recon'], update='append',
            opts=dict(xlabel='iter', ylabel='recon losses',
                      title='Recon Losses', legend=['both', 'A', 'B'])
        )

        self.viz.line(
            X=iters, Y=kls, env=self.name + '/lines',
            win=self.win_id['kl'], update='append',
            opts=dict(xlabel='iter', ylabel='kl losses',
                      title='KL Losses', legend=['A', 'B', 'POE']),
        )

        self.viz.line(
            X=iters, Y=tc, env=self.name + '/lines',
            win=self.win_id['tc'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='tc', legend=['tc', 'tc_infA', 'tc_infB', 'tc_poeA', 'tc_poeB']),
        )

        self.viz.line(
            X=iters, Y=mi, env=self.name + '/lines',
            win=self.win_id['mi'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='mi', legend=['mi', 'mi_infA', 'mi_infB', 'mi_poeA', 'mi_poeB']),
        )

        self.viz.line(
            X=iters, Y=dw_kl, env=self.name + '/lines',
            win=self.win_id['dw_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='dw_kl', legend=['dw_kl', 'dw_kl_infA', 'dw_kl_infB', 'dw_kl_poeA', 'dw_kl_poeB']))
        self.viz.line(
            X=iters, Y=acc, env=self.name + '/lines',
            win=self.win_id['acc'], update='append',
            opts=dict(xlabel='iter', ylabel='acc',
            title = 'Accuracy of modalB', legend = ['acc_infB', 'acc_POE', 'acc_sinfA', 'acc_infA']),
        )
        self.viz.line(
            X=iters, Y=acc_te, env=self.name + '/lines',
            win=self.win_id['acc_te'], update='append',
            opts=dict(xlabel='iter', ylabel='acc_te',
            title = 'Accuracy of modalB test set', legend = ['acc_infB_te', 'acc_POE_te', 'acc_sinfA_te', 'acc_sinfA_te']),
        )
        self.viz.line(
            X=iters, Y=mgll, env=self.name + '/lines',
            win=self.win_id['mgll'], update='append',
            opts=dict(xlabel='iter', ylabel='mgll',
            title = 'marginal LL of modalA', legend = ['marginal_ll_A_infA', 'marginal_ll_A_poe', 'marginal_ll_A_pAsB']),
        )
        self.viz.line(
            X=iters, Y=mgll_te, env=self.name + '/lines',
            win=self.win_id['mgll_te'], update='append',
            opts=dict(xlabel='iter', ylabel='mgll_te',
            title = 'marginal LL of modalA test set', legend = ['marginal_ll_A_infA_te', 'marginal_ll_A_poe_te', 'marginal_ll_A_pAsB_te']),
        )
    ####
    def visualize_line_metrics(self, iters, metric1, metric2):

        # prepare data to plot
        iters = torch.tensor([iters], dtype=torch.int64).detach()
        metric1 = torch.tensor([metric1])
        metric2 = torch.tensor([metric2])
        metrics = torch.stack([metric1.detach(), metric2.detach()], -1)

        self.viz.line(
            X=iters, Y=metrics, env=self.name + '/lines',
            win=self.win_id['metrics'], update='append',
            opts=dict(xlabel='iter', ylabel='metrics',
                      title='Disentanglement metrics',
                      legend=['metric1', 'metric2'])
        )

    def set_mode(self, train=True):

        if train:
            self.encoderA.train()
            self.encoderB.train()
            self.decoderA.train()
            self.decoderB.train()
        else:
            self.encoderA.eval()
            self.encoderB.eval()
            self.decoderA.eval()
            self.decoderB.eval()

    ####
    def save_checkpoint(self, iteration):

        encoderA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderA.pt' % iteration
        )
        encoderB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderB.pt' % iteration
        )
        decoderA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderA.pt' % iteration
        )
        decoderB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderB.pt' % iteration
        )


        mkdirs(self.ckpt_dir)

        torch.save(self.encoderA, encoderA_path)
        torch.save(self.encoderB, encoderB_path)
        torch.save(self.decoderA, decoderA_path)
        torch.save(self.decoderB, decoderB_path)

    ####
    def load_checkpoint(self):

        encoderA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderA.pt' % self.ckpt_load_iter
        )
        encoderB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderB.pt' % self.ckpt_load_iter
        )
        decoderA_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderA.pt' % self.ckpt_load_iter
        )
        decoderB_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderB.pt' % self.ckpt_load_iter
        )

        if self.use_cuda:
            self.encoderA = torch.load(encoderA_path)
            self.encoderB = torch.load(encoderB_path)
            self.decoderA = torch.load(decoderA_path)
            self.decoderB = torch.load(decoderB_path)
        else:
            self.encoderA = torch.load(encoderA_path, map_location='cpu')
            self.encoderB = torch.load(encoderB_path, map_location='cpu')
            self.decoderA = torch.load(decoderA_path, map_location='cpu')
            self.decoderB = torch.load(decoderB_path, map_location='cpu')

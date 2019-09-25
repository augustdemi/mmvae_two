import os
import numpy as np

import torch.optim as optim
from datasets import DIGIT
from torch.utils.data import DataLoader
from torchvision.utils import save_image
# -----------------------------------------------------------------------------#
from classifier import Net
from utils import DataGather, mkdirs, grid2gif2, apply_poe, sample_gaussian, sample_gumbel_softmax, \
    get_log_pz_qz_prodzi_qzCx
from model import *
from loss import kl_loss_function, reconstruction_loss
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

###############################################################################

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        self.name = '%s_lamkl_%s_zA_%s_zB_%s_zS_%s_HYPER_beta1_%s_beta2_%s_beta3_%s_beta11_%s_beta22_%s_beta33_%s' % \
                    (
                        args.dataset, args.lamkl, args.zA_dim, args.zB_dim, args.zS_dim, args.beta1, args.beta2,
                        args.beta3, args.beta11, args.beta22, args.beta33)
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

        # visdom setup
        self.viz_on = args.viz_on
        if self.viz_on:
            self.win_id = dict(
                recon='win_recon', kl='win_kl', capa='win_capa', tc='win_tc', mi='win_mi', dw_kl='win_dw_kl', acc='win_acc', disc_latent_acc='win_acc_disc_latent'
            )
            self.line_gather = DataGather(
                'iter', 'recon_both', 'recon_A', 'recon_B',
                'kl_A', 'kl_B', 'kl_POE',
                'tc_loss', 'mi_loss', 'dw_kl_loss',
                'tc_loss_A', 'mi_loss_A', 'dw_kl_loss_A',
                'tc_loss_B', 'mi_loss_B', 'dw_kl_loss_B',
                'tc_loss_POEA', 'mi_loss_POEA', 'dw_kl_loss_POEA',
                'tc_loss_POEB', 'mi_loss_POEB', 'dw_kl_loss_POEB',
                'poeA_acc', 'infA_acc', 'synA_acc',
                'poeB_acc', 'infB_acc', 'synB_acc',
                'acc_ZS_infA', 'acc_ZS_infB', 'acc_ZS_POE'
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
            self.encoderA = EncoderA(self.zA_dim, self.zS_dim)
            self.encoderB = EncoderA(self.zB_dim, self.zS_dim)
            self.decoderA = DecoderA(self.zA_dim, self.zS_dim)
            self.decoderB = DecoderA(self.zB_dim, self.zS_dim)

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


    ####
    def train(self):

        self.set_mode(train=True)

        # prepare dataloader (iterable)
        print('Start loading data...')
        dset = DIGIT('./data', train=True)
        self.data_loader = torch.utils.data.DataLoader(dset, batch_size=self.batch_size, shuffle=True)
        test_dset = DIGIT('./data', train=False)
        self.test_data_loader = torch.utils.data.DataLoader(test_dset, batch_size=self.batch_size, shuffle=True)
        print('test: ', len(test_dset))
        self.N = len(self.data_loader.dataset)
        print('...done')

        # iterators from dataloader
        iterator1 = iter(self.data_loader)

        iter_per_epoch = len(iterator1)

        start_iter = self.ckpt_load_iter + 1
        epoch = int(start_iter / iter_per_epoch)

        for iteration in range(start_iter, self.max_iter + 1):

            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)
                epoch += 1
                iterator1 = iter(self.data_loader)

            # ============================================
            #          TRAIN THE VAE (ENC & DEC)
            # ============================================

            # sample a mini-batch
            XA, XB, label, index = next(iterator1)  # (n x C x H x W)

            index = index.cpu().detach().numpy()
            if self.use_cuda:
                XA = XA.cuda()
                XB = XB.cuda()

            # zA, zS = encA(xA)
            muA_infA, stdA_infA, logvarA_infA, cate_prob_infA = self.encoderA(XA)

            # zB, zS = encB(xB)
            muB_infB, stdB_infB, logvarB_infB, cate_prob_infB = self.encoderB(XB)

            '''
            POE: should be the paramter for the distribution
            induce zS = encAB(xA,xB) via POE, that is,
                q(zA,zB,zS | xA,xB) := qI(zA|xA) * qT(zB|xB) * q(zS|xA,xB)
                    where q(zS|xA,xB) \propto p(zS) * qI(zS|xA) * qT(zS|xB)
            '''
            cate_prob_POE = torch.tensor(1 / 10) * cate_prob_infA * cate_prob_infB

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
            # the above sampling of ZS_infA/B are same 'way' as below
            # ZS_infA = sample_gumbel_softmax(self.use_cuda, cate_prob_infA)
            # ZS_infB = sample_gumbel_softmax(self.use_cuda, cate_prob_infB)



            #### For all cate_prob_infA(statiscts), total 64, get log_prob_ZS_infB2 for each of ZS_infB2(sample) ==> 64*64. marig. out for q_z for MI

            # reconstructed samples (given joint modal observation)
            XA_POE_recon = self.decoderA(ZA_infA, ZS_POE)
            XB_POE_recon = self.decoderB(ZB_infB, ZS_POE)

            # reconstructed samples (given single modal observation)
            XA_infA_recon = self.decoderA(ZA_infA, ZS_infA)
            XB_infB_recon = self.decoderB(ZB_infB, ZS_infB)

            # loss_recon_infA = F.l1_loss(torch.sigmoid(XA_infA_recon), XA, reduction='sum').div(XA.size(0))
            loss_recon_infA = reconstruction_loss(XA, torch.sigmoid(XA_infA_recon), distribution="bernoulli")
            #
            loss_recon_infB = reconstruction_loss(XB, torch.sigmoid(XB_infB_recon), distribution="bernoulli")
            #
            loss_recon_POE = \
                F.l1_loss(torch.sigmoid(XA_POE_recon), XA, reduction='sum').div(XA.size(0)) + \
                F.l1_loss(torch.sigmoid(XB_POE_recon), XB, reduction='sum').div(XB.size(0))
            #

            loss_recon = loss_recon_infA + loss_recon_infB + loss_recon_POE

            #================================== decomposed KL ========================================

            log_pz_A, log_qz_A, log_prod_qzi_A, log_q_zCx_A, _ = get_log_pz_qz_prodzi_qzCx({'cont': ZA_infA, 'disc': ZS_infA}, {'cont': (muA_infA, logvarA_infA), 'disc': relaxedCategA},
                                                                                len(self.data_loader.dataset),
                                                                                is_mss=self.is_mss)


            log_pz_B, log_qz_B, log_prod_qzi_B, log_q_zCx_B, _ = get_log_pz_qz_prodzi_qzCx({'cont': ZB_infB, 'disc': ZS_infB}, {'cont': (muB_infB, logvarB_infB), 'disc': relaxedCategB},
                                                                                len(self.data_loader.dataset),
                                                                                is_mss=self.is_mss)

            log_pz_POEA, log_qz_POEA, log_prod_qzi_POEA, log_q_zCx_POEA, _ = get_log_pz_qz_prodzi_qzCx({'cont': ZA_infA, 'disc': ZS_POE}, {'cont': (muA_infA, logvarA_infA), 'disc': relaxedCategS},
                                                                                len(self.data_loader.dataset),
                                                                                is_mss=self.is_mss)

            log_pz_POEB, log_qz_POEB, log_prod_qzi_POEB, log_q_zCx_POEB, _ = get_log_pz_qz_prodzi_qzCx({'cont': ZB_infB, 'disc': ZS_POE}, {'cont': (muB_infB, logvarB_infB), 'disc': relaxedCategS},
                                                                                len(self.data_loader.dataset),
                                                                                is_mss=self.is_mss)
            # loss_kl_infA
            mi_loss_A = (log_q_zCx_A - log_qz_A).mean()
            tc_loss_A = (log_qz_A - log_prod_qzi_A).sum(dim=0).div(self.batch_size)
            dw_kl_loss_A = (log_prod_qzi_A - log_pz_A).mean()
            loss_kl_infA = self.beta1 * mi_loss_A + self.beta2 * tc_loss_A + self.beta3 * dw_kl_loss_A
            # loss_kl_infB
            mi_loss_B = (log_q_zCx_B - log_qz_B).mean()
            tc_loss_B = (log_qz_B - log_prod_qzi_B).sum(dim=0).div(self.batch_size)
            dw_kl_loss_B = (log_prod_qzi_B - log_pz_B).mean()
            loss_kl_infB = self.beta11 * mi_loss_B + self.beta22 * tc_loss_B + self.beta33 * dw_kl_loss_B
            # loss_kl_POEA
            mi_loss_POEA = (log_q_zCx_POEA - log_qz_POEA).mean()
            tc_loss_POEA = (log_qz_POEA - log_prod_qzi_POEA).sum(dim=0).div(self.batch_size)
            dw_kl_loss_POEA = (log_prod_qzi_POEA - log_pz_POEA).mean()
            loss_kl_POEA = self.beta1 * mi_loss_POEA + self.beta2 * tc_loss_POEA + self.beta3 * dw_kl_loss_POEA
            # loss_kl_POEB
            mi_loss_POEB = (log_q_zCx_POEB - log_qz_POEB).mean()
            tc_loss_POEB = (log_qz_POEB - log_prod_qzi_POEB).sum(dim=0).div(self.batch_size)
            dw_kl_loss_POEB = (log_prod_qzi_POEB - log_pz_POEB).mean()
            loss_kl_POEB = self.beta11 * mi_loss_POEB + self.beta22 * tc_loss_POEB + self.beta33 * dw_kl_loss_POEB
            # loss_kl_POE
            loss_kl_POE = 0.5 * (loss_kl_POEA + loss_kl_POEB)

            loss_kl = loss_kl_infA + loss_kl_infB + loss_kl_POE

            tc_loss = tc_loss_A + tc_loss_B + 0.5 * (tc_loss_POEA + tc_loss_POEB)
            mi_loss = mi_loss_A + mi_loss_B + 0.5 * (mi_loss_POEA + mi_loss_POEB)
            dw_kl_loss = dw_kl_loss_A + dw_kl_loss_B + 0.5 * (dw_kl_loss_POEA + dw_kl_loss_POEB)

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
                z_A, z_B, z_S = self.get_stat()
                # 1) save the recon images


                # 2) save the pure-synthesis images
                # self.save_synth_pure( iteration, howmany=100 )
                # 3) save the cross-modal-synthesis images


                # 4) save the latent traversed images
                # self.save_traverseA(iteration, z_A, z_B, z_S)
                # self.save_traverseB(iteration, z_A, z_B, z_S)

                self.save_recon(iteration)
                self.save_recon(iteration, train=False)
                self.save_synth_cross_modal(iteration, z_A, z_B, howmany=3)
                self.save_synth_cross_modal(iteration, z_A, z_B, train=False, howmany=3)
                self.save_traverse(iteration, z_A, z_B, z_S)
                self.save_traverse(iteration, z_A, z_B, z_S, train=False)



            if iteration % self.eval_metrics_iter == 0:
                self.save_synth_cross_modal(iteration, z_A, z_B, train=False, howmany=3)





            # (visdom) insert current line stats
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                z_A, z_B, z_S = self.get_stat()

                # print(">>>>>> Train ACC")
                # (_, _, _, _, _, _) = self.acc_total(z_A, z_B, train=True, howmany=3)

                print(">>>>>> Test ACC")
                (synA_acc, synB_acc, poeA_acc, poeB_acc, infA_acc, infB_acc, acc_ZS_infA, acc_ZS_infB, acc_ZS_POE) \
                    = self.acc_total(z_A, z_B, train=False, howmany=3)

                self.line_gather.insert(iter=iteration,
                                        recon_both=loss_recon_POE.item(),
                                        recon_A=loss_recon_infA.item(),
                                        recon_B=loss_recon_infB.item(),
                                        kl_A=loss_kl_infA.item(),
                                        kl_B=loss_kl_infB.item(),
                                        kl_POE=loss_kl_POE.item(),
                                        synA_acc=synA_acc,
                                        synB_acc=synB_acc,
                                        poeA_acc=poeA_acc,
                                        poeB_acc=poeB_acc,
                                        infA_acc=infA_acc,
                                        infB_acc=infB_acc,
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
                                        acc_ZS_infA = acc_ZS_infA,
                                        acc_ZS_infB = acc_ZS_infB,
                                        acc_ZS_POE = acc_ZS_POE
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



    def check_acc(self, data, target, dataset='mnist', train=True):
        device = torch.device("cuda" if self.use_cuda else "cpu")
        model = Net().to(device)
        print('loaded: ', dataset + "_cnn_dict.pt")
        model.load_state_dict(torch.load(dataset + "_cnn_dict.pt"))

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(target)

        if train:
            print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format('Train',
                                                                                       test_loss, correct, len(target),
                                                                                       100. * correct / len(target)))
        else:
            print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format('Test',
                                                                                       test_loss, correct, len(target),
                                                                                       100. * correct / len(target)))

        pred = pred.resize(pred.size()[0])
        paired = torch.stack((target, pred), dim=1)

        temp = {}
        for i in range(10):
            temp.update({i: []})
        for i in range(int(paired.size()[0])):
            temp[int(paired[i][0])].append(paired[i])

        for i in range(10):
            one_paired = torch.stack(temp[i])
            one_target = one_paired[:,0]
            one_pred = one_paired[:,1]
            corr = one_pred.eq(one_target.view_as(one_pred)).sum().item()
            print('ACC of digit {}: {:.2f}'.format(i, corr / len(temp[i])) )
        print('-------------------------------------------------')
        return correct / len(target)

    def save_recon(self, iters, train=True):
        self.set_mode(train=False)

        mkdirs(self.output_dir_recon)

        if train:
            data_loader = self.data_loader
            fixed_idxs = [3246, 7001, 14308, 19000, 27447, 33103, 38002, 45232, 51000, 55125]
            out_dir = os.path.join(self.output_dir_recon, 'train')
        else:
            data_loader = self.test_data_loader
            fixed_idxs = [2, 982, 2300, 3400, 4500, 5500, 6500, 7500, 8500, 9500]
            out_dir = os.path.join(self.output_dir_recon, 'test')

        fixed_idxs60 = []
        for idx in fixed_idxs:
            for i in range(6):
                fixed_idxs60.append(idx + i)

        XA = [0] * len(fixed_idxs60)
        XB = [0] * len(fixed_idxs60)
        label = [0] * len(fixed_idxs60)

        for i, idx in enumerate(fixed_idxs60):
            XA[i], XB[i], label[i] = \
                data_loader.dataset.__getitem__(idx)[0:3]

            if self.use_cuda:
                XA[i] = XA[i].cuda()
                XB[i] = XB[i].cuda()

        XA = torch.stack(XA)
        XB = torch.stack(XB)
        label = torch.LongTensor(label)

        muA_infA, stdA_infA, logvarA_infA, cate_prob_infA = self.encoderA(XA)

        # zB, zS = encB(xB)
        muB_infB, stdB_infB, logvarB_infB, cate_prob_infB = self.encoderB(XB)

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

        # print('=========== save_rec: poeA ACC ============')
        # poeA_acc = self.check_acc(XA_POE_recon, label)
        # print('=========== save_rec: infA ACC ============')
        # infA_acc = self.check_acc(XA_infA_recon, label)
        #
        #
        # print('=========== save_rec: poeB ACC ============')
        # poeB_acc = self.check_acc(XB_POE_recon, label, dataset='fmnist')
        # print('=========== save_rec: infB ACC ============')
        # infB_acc = self.check_acc(XB_infB_recon, label, dataset='fmnist')

        #######################

        WS = torch.ones(XA.shape)
        if self.use_cuda:
            WS = WS.cuda()

        n = XA.shape[0]
        perm = torch.arange(0, 4 * n).view(4, n).transpose(1, 0)
        perm = perm.contiguous().view(-1)

        merged = torch.cat(
            [XA, XA_infA_recon, XA_POE_recon, WS], dim=0
        )
        merged = merged[perm, :].cpu()

        # save the results as image
        fname = os.path.join(out_dir, 'reconA_%s.jpg' % iters)
        mkdirs(out_dir)
        save_image(
            tensor=merged, filename=fname, nrow=4 * int(np.sqrt(n)),
            pad_value=1
        )

        WS = torch.ones(XB.shape)
        if self.use_cuda:
            WS = WS.cuda()

        n = XB.shape[0]
        perm = torch.arange(0, 4 * n).view(4, n).transpose(1, 0)
        perm = perm.contiguous().view(-1)

        ## ingr
        merged = torch.cat(
            [XB, XB_infB_recon, XB_POE_recon, WS], dim=0
        )
        merged = merged[perm, :].cpu()

        # save the results as image
        fname = os.path.join(out_dir, 'reconB_%s.jpg' % iters)
        mkdirs(out_dir)
        save_image(
            tensor=merged, filename=fname, nrow=4 * int(np.sqrt(n)),
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
    def save_synth_cross_modal(self, iters, z_A_stat, z_B_stat, train=True, howmany=3):

        self.set_mode(train=False)

        if train:
            data_loader = self.data_loader
            fixed_idxs = [3246, 7001, 14308, 19000, 27447, 33103, 38002, 45232, 51000, 55125]
        else:
            data_loader = self.test_data_loader
            fixed_idxs = [2, 982, 2300, 3400, 4500, 5500, 6500, 7500, 8500, 9500]

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

        _, _, _, cate_prob_infA = self.encoderA(fixed_XA)

        # zB, zS = encB(xB)
        _, _, _, cate_prob_infB = self.encoderB(fixed_XB)

        ZS_infA = sample_gumbel_softmax(self.use_cuda, cate_prob_infA, train=False)
        ZS_infB = sample_gumbel_softmax(self.use_cuda, cate_prob_infB, train=False)

        if self.use_cuda:
            ZS_infA = ZS_infA.cuda()
            ZS_infB = ZS_infB.cuda()

        decoderA = self.decoderA
        decoderB = self.decoderB

        # mkdirs(os.path.join(self.output_dir_synth, str(iters)))


        WS = torch.ones(fixed_XA.shape)
        if self.use_cuda:
            WS = WS.cuda()

        n = len(fixed_idxs)

        perm = torch.arange(0, (howmany + 2) * n).view(howmany + 2, n).transpose(1, 0)
        perm = perm.contiguous().view(-1)

        ######## 1) generate xB from given xA (A2B) ########

        merged = torch.cat([fixed_XA], dim=0)
        XB_synth_list = []
        label_list = []

        for k in range(howmany):
            # z_B_stat = np.array(z_B_stat)
            # z_B_stat_mean = np.mean(z_B_stat, 0)
            # ZB = torch.Tensor(z_B_stat_mean)
            # ZB_list = []
            # for _ in range(n):
            #     ZB_list.append(ZB)
            # ZB = torch.stack(ZB_list)

            ZB = torch.randn(n, self.zB_dim)
            z_B_stat = np.array(z_B_stat)
            z_B_stat_mean = np.mean(z_B_stat, 0)
            ZB = ZB + torch.Tensor(z_B_stat_mean)

            if self.use_cuda:
                ZB = ZB.cuda()
            XB_synth = torch.sigmoid(decoderB(ZB, ZS_infA))  # given XA
            XB_synth_list.extend(XB_synth)
            label_list.extend(label)
            # merged = torch.cat([merged, fixed_XA_3ch], dim=0)
            merged = torch.cat([merged, XB_synth], dim=0)
        merged = torch.cat([merged, WS], dim=0)
        merged = merged[perm, :].cpu()

        # print('=========== cross-synth ACC for XB_synth ============')
        # XB_synth_list = torch.stack(XB_synth_list)
        # label_list = torch.LongTensor(label_list)
        # self.check_acc(XB_synth_list, label_list, dataset='fmnist')

        # save the results as image
        if train:
            fname = os.path.join(
                self.output_dir_synth,
                'synth_cross_modal_A2B_%s.jpg' % iters
            )
        else:
            fname = os.path.join(
                self.output_dir_synth,
                'eval_synth_cross_modal_A2B_%s.jpg' % iters
            )
        mkdirs(self.output_dir_synth)
        save_image(
            tensor=merged, filename=fname, nrow=(howmany + 2) * int(np.sqrt(n)),
            pad_value=1
        )

        ######## 2) generate xA from given xB (B2A) ########
        merged = torch.cat([fixed_XB], dim=0)
        XA_synth_list = []
        for k in range(howmany):
            # z_A_stat = np.array(z_A_stat)
            # z_A_stat_mean = np.mean(z_A_stat, 0)
            # ZA = torch.Tensor(z_A_stat_mean)
            # ZA_list = []
            # for _ in range(n):
            #     ZA_list.append(ZA)
            # ZA = torch.stack(ZA_list)

            ZA = torch.randn(n, self.zA_dim)
            z_A_stat = np.array(z_A_stat)
            z_A_stat_mean = np.mean(z_A_stat, 0)
            ZA = ZA + torch.Tensor(z_A_stat_mean)

            if self.use_cuda:
                ZA = ZA.cuda()
            XA_synth = torch.sigmoid(decoderA(ZA, ZS_infB))  # given XB
            XA_synth_list.extend(XA_synth)

            # merged = torch.cat([merged, fixed_XB[:,:,2:30, 2:30]], dim=0)
            merged = torch.cat([merged, XA_synth], dim=0)
        merged = torch.cat([merged, WS], dim=0)
        merged = merged[perm, :].cpu()

        # print('=========== cross-synth ACC for XA_synth ============')
        # XA_synth_list = torch.stack(XA_synth_list)
        # label_list = torch.LongTensor(label_list)
        # self.check_acc(XA_synth_list, label_list, dataset='mnist')

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
        encoderA = self.encoderA
        encoderB = self.encoderB

        z_A, z_B, z_S = [], [], []
        for _ in range(10000):
            rand_i = np.random.randint(self.N)
            random_XA, random_XB = self.data_loader.dataset.__getitem__(rand_i)[0:2]
            if self.use_cuda:
                random_XA = random_XA.cuda()
                random_XB = random_XB.cuda()
            random_XA = random_XA.unsqueeze(0)
            random_XB = random_XB.unsqueeze(0)

            muA_infA, stdA_infA, logvarA_infA, cate_prob_infA = self.encoderA(random_XA)

            # zB, zS = encB(xB)
            muB_infB, stdB_infB, logvarB_infB, cate_prob_infB = self.encoderB(random_XB)
            cate_prob_POE = torch.tensor(1 / 10) * cate_prob_infA * cate_prob_infB
            z_A.append(muA_infA.cpu().detach().numpy()[0])
            z_B.append(muB_infB.cpu().detach().numpy()[0])
            z_S.append(cate_prob_POE.cpu().detach().numpy()[0])
        return z_A, z_B, z_S


    def save_traverseA(self, iters, z_A, z_B, z_S, loc=-1):

        self.set_mode(train=False)

        encoderA = self.encoderA
        encoderB = self.encoderB
        decoderA = self.decoderA
        decoderB = self.decoderB
        interpolationA = torch.tensor(np.linspace(-3, 3, self.zS_dim))

        print('------------ traverse interpolation ------------')
        print('interpolationA: ', np.min(np.array(z_A)), np.max(np.array(z_A)))
        print('interpolationB: ', np.min(np.array(z_B)), np.max(np.array(z_B)))
        print('interpolationS: ', np.min(np.array(z_S)), np.max(np.array(z_S)))

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
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs = torch.cat(tempA, dim=0) #torch.Size([11, 10, 1, 384, 32])


        # save the generated files, also the animated gifs
        out_dir = os.path.join(self.output_dir_trvsl, str(iters), 'trainA')
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
    def save_traverseB(self, iters, z_A, z_B, z_S, loc=-1):

        self.set_mode(train=False)

        encoderB = self.encoderB
        decoderB = self.decoderB
        interpolationA = torch.tensor(np.linspace(-3, 3, self.zS_dim))

        print('------------ traverse interpolation ------------')
        print('interpolationA: ', np.min(np.array(z_A)), np.max(np.array(z_A)))
        print('interpolationB: ', np.min(np.array(z_B)), np.max(np.array(z_B)))
        print('interpolationS: ', np.min(np.array(z_S)), np.max(np.array(z_S)))

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
    def save_traverse(self, iters, z_A, z_B, z_S, loc=-1, train=True):

        self.set_mode(train=False)

        encoderA = self.encoderA
        encoderB = self.encoderB
        decoderA = self.decoderA
        decoderB = self.decoderB
        interpolationA = torch.tensor(np.linspace(-3, 3, self.zS_dim))

        print('------------ traverse interpolation ------------')
        print('interpolationA: ', np.min(np.array(z_A)), np.max(np.array(z_A)))
        print('interpolationB: ', np.min(np.array(z_B)), np.max(np.array(z_B)))
        print('interpolationS: ', np.min(np.array(z_S)), np.max(np.array(z_S)))
        if train:
            data_loader = self.data_loader
            fixed_idxs = [3246, 7001, 14308, 19000, 27447, 33103, 38002, 45232, 51000, 55125]
            out_dir = os.path.join(self.output_dir_trvsl, str(iters), 'train')
        else:
            data_loader = self.test_data_loader
            fixed_idxs = [2, 982, 2300, 3400, 4500, 5500, 6500, 7500, 8500, 9500]
            out_dir = os.path.join(self.output_dir_trvsl, str(iters), 'test')


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

            fixed_zmuA, _, _, cate_prob_infA = encoderA(fixed_XA)

            # zB, zS = encB(xB)
            fixed_zmuB, _, _, cate_prob_infB = encoderB(fixed_XB)

            fixed_cate_probS = torch.tensor(1 / 10) * cate_prob_infA * cate_prob_infB

            fixed_zS = sample_gumbel_softmax(self.use_cuda, fixed_cate_probS, train=False)


            saving_shape=torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1).shape

        ####

        WS = torch.ones(saving_shape)
        if self.use_cuda:
            WS = WS.cuda()

        # do traversal and collect generated images
        gifs = []

        zA_ori, zB_ori, zS_ori = fixed_zmuA, fixed_zmuB, fixed_zS

        ###A

        tempAll = [] # zA_dim + zS_dim , num_trv, 1, 32*num_samples, 32
        for row in range(self.zA_dim):
            if loc != -1 and row != loc:
                continue
            zA = zA_ori.clone()

            temp = []
            for val in interpolationA:
                zA[:, row] = val
                sampleA = torch.sigmoid(decoderA(zA, zS_ori)).data
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

            tempAll.append(torch.cat(temp, dim=0).unsqueeze(0)) # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

        temp = []
        for i in range(self.zS_dim):
            zS = np.zeros((1, self.zS_dim))
            zS[0, i % self.zS_dim] = 1.
            zS = torch.Tensor(zS)
            zS = torch.cat([zS] * len(fixed_idxs), dim=0)

            if self.use_cuda:
                zS = zS.cuda()

            sampleA = torch.sigmoid(decoderA(zA_ori, zS)).data
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))

        ###B

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
        tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))


        for row in range(self.zB_dim):
            if loc != -1 and row != loc:
                continue
            zB = zB_ori.clone()

            temp = []
            for val in interpolationA:
                zB[:, row] = val
                sampleB = torch.sigmoid(decoderB(zB, zS_ori)).data
                temp.append((torch.cat([sampleB[i] for i in range(sampleB.shape[0])], dim=1)).unsqueeze(0))

            tempAll.append(torch.cat(temp, dim=0).unsqueeze(0)) # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32



        gifs = torch.cat(tempAll, dim=0) #torch.Size([11, 10, 1, 384, 32])


        # save the generated files, also the animated gifs
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
            out_dir, str(os.path.join(out_dir, 'both_traverse' + '.gif')), delay=10
        )

        self.set_mode(train=True)

    ####
    def viz_init(self):

        self.viz.close(env=self.name + '/lines', win=self.win_id['recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['capa'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['acc'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['tc'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['mi'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['dw_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['disc_latent_acc'])

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

        poeA_acc = torch.Tensor(data['poeA_acc'])
        infA_acc = torch.Tensor(data['infA_acc'])
        synA_acc = torch.Tensor(data['synA_acc'])
        poeB_acc = torch.Tensor(data['poeB_acc'])
        infB_acc = torch.Tensor(data['infB_acc'])
        synB_acc = torch.Tensor(data['synB_acc'])

        acc_ZS_infA = torch.Tensor(data['acc_ZS_infA'])
        acc_ZS_infB = torch.Tensor(data['acc_ZS_infB'])
        acc_ZS_POE = torch.Tensor(data['acc_ZS_POE'])

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
            [poeA_acc.detach(), infA_acc.detach(), synA_acc.detach(), poeB_acc.detach(), infB_acc.detach(), synB_acc.detach()], -1
        )

        disc_latent_acc = torch.stack(
            [acc_ZS_infA.detach(), acc_ZS_infB.detach(), acc_ZS_POE.detach()], -1
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
            X=iters, Y=acc, env=self.name + '/lines',
            win=self.win_id['acc'], update='append',
            opts=dict(xlabel='iter', ylabel='accuracy',
            title = 'Classification Acc', legend = ['poeA_acc', 'infA_acc', 'synA_acc', 'poeB_acc', 'infB_acc', 'synB_acc']),
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
            X=iters, Y=disc_latent_acc, env=self.name + '/lines',
            win=self.win_id['disc_latent_acc'], update='append',
            opts=dict(xlabel='iter', ylabel='disc_latent_acc',
            title = 'Discrete latent Acc', legend = ['acc_ZS_infA', 'acc_ZS_infB', 'acc_ZS_POE']),
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

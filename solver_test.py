import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#-----------------------------------------------------------------------------#

from utils import DataGather, mkdirs, grid2gif, apply_poe, sample_gaussian
from model import * 
from dataset import create_dataloader
#from utils_williams import *

###############################################################################

class Solver(object):
    
    ####
    def __init__(self, args):
        
        self.args = args
        
        self.name = '%s_lamkl_%s_z_%s' % \
            (args.dataset, args.lamkl, args.z_dim)
        self.name = self.name + '_run_' + str(args.run_id)
        
        self.use_cuda = args.cuda and torch.cuda.is_available()
        
        # data info
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        if args.dataset == 'idaz_elli_3df':  # multi-modal data
            self.nc = 1
        else:
            self.nc = 3
        
        # groundtruth factor labels (only available for some datasets)
            
        if self.dataset=='idaz_elli_3df':
            
            # latent factor = (id, azimuth, elevation, lighting)
            #   id = {0,1,...,49} (50)
            #   azimuth = {-1.0,-0.9,...,0.9,1.0} (21)
            #   elevation = {-1.0,0.8,...,0.8,1.0} (11)
            #   lighting = {-1.0,0.8,...,0.8,1.0} (11)
            # (number of variations = 50*21*11*11 = 127050)
            
            latent_classes, latent_values = np.load( os.path.join( 
                self.dset_dir, '3d_faces/rtqichen/gt_factor_labels.npy' ) )
            self.latent_values = latent_values
                # latent values (actual values);(127050 x 4)
            self.latent_classes = latent_classes
                # classes ({0,1,...,K}-valued); (127050 x 4)
            self.latent_sizes = np.array([50, 21, 11, 11])
            self.N = self.latent_values.shape[0]

            
        # networks and optimizers
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        self.lamkl = args.lamkl
        
        # what to do in this test        
        self.num_synth = args.num_synth
        self.num_trvsl = args.num_trvsl
        self.num_eval_metric1 = args.num_eval_metric1
        self.num_eval_metric2 = args.num_eval_metric2
        
        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)
        
        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");  mkdirs("outputs")
        
        # records
        self.record_file = 'records/%s.txt' % ("test_" + self.name)

        # outputs
        self.output_dir_recon = os.path.join( "outputs", 
                                              "test_" + self.name + '_recon' )
        self.output_dir_synth = os.path.join( "outputs",  
                                              "test_" + self.name + '_synth' )
        self.output_dir_trvsl = os.path.join( "outputs",  
                                              "test_" + self.name + '_trvsl' )
        
        
        # load a previously saved model
        self.ckpt_load_iter = args.ckpt_load_iter
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
        
        self.set_mode(train=False)


    ####
    def test(self):
            
        # prepare dataloader (iterable)
        print('Start loading data...')
        self.data_loader = create_dataloader(self.args)
        print('...done')
        
        # iterator from dataloader
        iterator = iter(self.data_loader)
        iter_per_epoch = len(iterator)
        
        #----#
        
        # image synthesis
        if self.num_synth > 0:
            prn_str = 'Start doing image synthesis...'
            print(prn_str)
            self.dump_to_record(prn_str)
            for ii in range(self.num_synth):
                # save the pure-synthesis images
                self.save_synth_pure( 
                    str(self.ckpt_load_iter) + '_' + str(ii),  howmany=100 )                
                # save the cross-modal-synthesis images
                self.save_synth_cross_modal( 
                    str(self.ckpt_load_iter) + '_' + str(ii) )
        
        # latent traversal
        if self.num_trvsl > 0:
            prn_str = 'Start doing latent traversal...'
            print(prn_str)
            self.dump_to_record(prn_str)
            # self.save_traverse_new( self.ckpt_load_iter, self.num_trvsl, 
            #                         limb=-4, limu=4, inter=0.1 )
            self.save_traverse_new( self.ckpt_load_iter, self.num_trvsl, 
                                    limb=-3, limu=3, inter=0.1 )
        
        # metric1
        if self.num_eval_metric1 > 0:
            prn_str = 'Start evaluating metric1...'
            print(prn_str)
            self.dump_to_record(prn_str)
            #
            metric1s = np.zeros(self.num_eval_metric1)
            C1s = np.zeros([ self.num_eval_metric1, 
                             self.z_dim,
                             len(self.latent_sizes) ])
            for ii in range(self.num_eval_metric1):
                metric1s[ii], C1s[ii] = self.eval_disentangle_metric1()
                prn_str = 'eval metric1: %d/%d done' % \
                          (ii+1, self.num_eval_metric1)
                print(prn_str)
                self.dump_to_record(prn_str)
            #
            prn_str = 'metric1:\n' + str(metric1s)
            prn_str += '\nC1:\n' + str(C1s)
            print(prn_str)
            self.dump_to_record(prn_str)
            
        
        # metric2
        if self.num_eval_metric2 > 0:
            prn_str = 'Start evaluating metric2...'
            print(prn_str)
            self.dump_to_record(prn_str)
            #
            metric2s = np.zeros(self.num_eval_metric2)
            C2s = np.zeros([ self.num_eval_metric2, 
                             self.z_dim,
                             len(self.latent_sizes) ])            
            for ii in range(self.num_eval_metric2):
                metric2s[ii], C2s[ii] = self.eval_disentangle_metric2()
                prn_str = 'eval metric2: %d/%d done' % \
                          (ii+1, self.num_eval_metric2)
                print(prn_str)
                self.dump_to_record(prn_str)
            #
            prn_str = 'metric2:\n' + str(metric2s)
            prn_str += '\nC2:\n' + str(C2s)
            print(prn_str)
            self.dump_to_record(prn_str)
   

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
          shuffle=True, num_workers=self.args.num_workers, pin_memory=True )
        iterator = iter(dl)
        
        M = []
        for ib in range(int(nsamps_agn_factor/bs)):
            
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
            for r in range(int(num_pairs/len(factor_ids))):

                # a true factor (id and class value) to fix
                fac_id = j
                fac_class = np.random.randint(self.latent_sizes[fac_id])

                # randomly select images (with the fixed factor)
                indices = np.where( 
                  self.latent_classes[:,fac_id]==fac_class )[0]
                np.random.shuffle(indices)
                idx = indices[:nsamps_per_factor]
                M = []
                for ib in range(int(nsamps_per_factor/bs)):
                    XAb, XBb, _, _, _ = dl.dataset[ idx[(ib*bs):(ib+1)*bs] ]
                    if XAb.shape[0]<1:  # no more samples
                        continue;
                    if self.use_cuda:
                        XAb = XAb.cuda()
                        XBb = XBb.cuda()                    
                    mu_infA, _, logvar_infA = self.encoderA(XAb)
                    mu_infB, _, logvar_infB = self.encoderB(XBb)
                    mu_POE, _, _ = apply_poe( self.use_cuda, 
                        mu_infA, logvar_infA, mu_infB, logvar_infB,
                    )
                    mub = mu_POE
                    M.append(mub.cpu().detach().numpy())
                M = np.concatenate(M, 0)
                                
                # estimate sample var and mean of latent points for each dim
                if M.shape[0]>=2:
                    vars_per_factor[i,:] = np.var(M, 0)
                else:  # not enough samples to estimate variance
                    vars_per_factor[i,:] = 0.0                
                
                # true factor id (will become the class label)
                true_factor_ids[i] = fac_id

                i += 1
                
        # 3) evaluate majority vote classification accuracy
 
        # inputs in the paired data for classification
        smallest_var_dims = np.argmin(
          vars_per_factor / (vars_agn_factor + 1e-20), axis=1 )
    
        # contingency table
        C = np.zeros([self.z_dim,len(factor_ids)])
        for i in range(num_pairs):
            C[ smallest_var_dims[i], true_factor_ids[i] ] += 1
        
        num_errs = 0  # # misclassifying errors of majority vote classifier
        for k in range(self.z_dim):
            num_errs += np.sum(C[k,:]) - np.max(C[k,:])
        
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
          shuffle=True, num_workers=self.args.num_workers, pin_memory=True )
        iterator = iter(dl)
        
        M = []
        for ib in range(int(nsamps_agn_factor/bs)):
            
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
            for r in range(int(num_pairs/len(factor_ids))):
                                
                # randomly choose true factors (id's and class values) to fix
                fac_ids = list(np.setdiff1d(factor_ids,j))
                fac_classes = \
                  [ np.random.randint(self.latent_sizes[k]) for k in fac_ids ]

                # randomly select images (with the other factors fixed)
                if len(fac_ids)>1:
                    indices = np.where( 
                      np.sum(self.latent_classes[:,fac_ids]==fac_classes,1)
                      == len(fac_ids) 
                    )[0]
                else:
                    indices = np.where(
                      self.latent_classes[:,fac_ids]==fac_classes 
                    )[0]
                np.random.shuffle(indices)
                idx = indices[:nsamps_per_factor]
                M = []
                for ib in range(int(nsamps_per_factor/bs)):                    
                    XAb, XBb, _, _, _ = dl.dataset[ idx[(ib*bs):(ib+1)*bs] ]
                    if XAb.shape[0]<1:  # no more samples
                        continue;
                    if self.use_cuda:
                        XAb = XAb.cuda()
                        XBb = XBb.cuda()
                    mu_infA, _, logvar_infA = self.encoderA(XAb)
                    mu_infB, _, logvar_infB = self.encoderB(XBb)
                    mu_POE, _, _ = apply_poe( self.use_cuda, 
                        mu_infA, logvar_infA, mu_infB, logvar_infB,
                    )
                    mub = mu_POE
                    M.append(mub.cpu().detach().numpy())
                M = np.concatenate(M, 0)
                
                # estimate sample var and mean of latent points for each dim
                if M.shape[0]>=2:
                    vars_per_factor[i,:] = np.var(M, 0)
                else:  # not enough samples to estimate variance
                    vars_per_factor[i,:] = 0.0
                    
                # true factor id (will become the class label)
                true_factor_ids[i] = j

                i += 1
                
        # 3) evaluate majority vote classification accuracy
            
        # inputs in the paired data for classification
        largest_var_dims = np.argmax(
          vars_per_factor / (vars_agn_factor + 1e-20), axis=1 )
    
        # contingency table
        C = np.zeros([self.z_dim,len(factor_ids)])
        for i in range(num_pairs):
            C[ largest_var_dims[i], true_factor_ids[i] ] += 1
        
        num_errs = 0  # # misclassifying errors of majority vote classifier
        for k in range(self.z_dim):
            num_errs += np.sum(C[k,:]) - np.max(C[k,:])
    
        metric2 = (num_pairs - num_errs) / num_pairs  # metric = accuracy    
        
        self.set_mode(train=True)

        return metric2, C


    ####
    def save_synth_pure(self, iters, howmany=100):
        
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
        
        perm = torch.arange(0,3*howmany).view(3,howmany).transpose(1,0)
        perm = perm.contiguous().view(-1)
        merged = torch.cat([XA, XB, WS], dim=0)
        merged = merged[perm,:].cpu()
    
        # save the results as image
        fname = os.path.join(
            self.output_dir_synth, 'synth_pure_%s.jpg' % iters
        )
        mkdirs(self.output_dir_synth)
        save_image( 
          tensor=merged, filename=fname, nrow=3*int(np.sqrt(howmany)), 
          pad_value=1
        )
        
    
    ####
    def save_synth_cross_modal(self, iters):
        
        decoderA = self.decoderA
        decoderB = self.decoderB
        
        # sample a mini-batch
        iterator1 = iter(self.data_loader)
        XA, XB, idsA, idsB, ids = next(iterator1)  # (n x C x H x W)
        if self.use_cuda:
            XA = XA.cuda()
            XB = XB.cuda()
            
        # z = encA(xA)
        mu_infA, std_infA, _ = self.encoderA(XA)
        
        # z = encB(xB)
        mu_infB, std_infB, _ = self.encoderB(XB)
        
        # encoder samples (for cross-modal prediction)
        Z_infA = sample_gaussian(self.use_cuda, mu_infA, std_infA)
        Z_infB = sample_gaussian(self.use_cuda, mu_infB, std_infB)

        WS = torch.ones(XA.shape)
        if self.use_cuda:
            WS = WS.cuda()
        
        n = XA.shape[0]
                
        perm = torch.arange(0,(1+2)*n).view(1+2,n).transpose(1,0)
        perm = perm.contiguous().view(-1)
        
        # 1) generate xB from given xA (A2B)
        
        merged = torch.cat([XA], dim=0)
        XB_synth = torch.sigmoid(decoderB(Z_infA)).data  # given XA
        merged = torch.cat([merged, XB_synth], dim=0)
        merged = torch.cat([merged, WS], dim=0)
        merged = merged[perm,:].cpu()
    
        # save the results as image
        fname = os.path.join( 
            self.output_dir_synth, 
            'synth_cross_modal_A2B_%s.jpg' % iters 
        )
        mkdirs(self.output_dir_synth)
        save_image( 
          tensor=merged, filename=fname, nrow=(1+2)*int(np.sqrt(n)), 
          pad_value=1
        )

        # 2) generate xA from given xB (B2A)
        
        merged = torch.cat([XB], dim=0)
        XA_synth = torch.sigmoid(decoderA(Z_infB)).data  # given XB
        merged = torch.cat([merged, XA_synth], dim=0)
        merged = torch.cat([merged, WS], dim=0)
        merged = merged[perm,:].cpu()
    
        # save the results as image
        fname = os.path.join( 
            self.output_dir_synth, 
            'synth_cross_modal_B2A_%s.jpg' % iters 
        )
        mkdirs(self.output_dir_synth)
        save_image( 
          tensor=merged, filename=fname, nrow=(1+2)*int(np.sqrt(n)), 
          pad_value=1
        )
        
    
    ####
    def save_traverse_new( self, iters, num_reps, 
                           limb=-3, limu=3, inter=2/3, loc=-1 ):
        
        encoderA = self.encoderA
        encoderB = self.encoderB
        decoderA = self.decoderA
        decoderB = self.decoderB
        interpolation = torch.arange(limb, limu+0.001, inter)
        
        np.random.seed(123)
        rii = np.random.randint(self.N, size=num_reps)
        #--#
        prn_str = '(TRAVERSAL) random image IDs = {}'.format(rii)
        print(prn_str)
        self.dump_to_record(prn_str)
        #--#
        random_XA = [0]*num_reps
        random_XB = [0]*num_reps
        random_zmu = [0]*num_reps
        for i, i2 in enumerate(rii):
            random_XA[i], random_XB[i] = \
                self.data_loader.dataset.__getitem__(i2)[0:2]
            if self.use_cuda:
                random_XA[i] = random_XA[i].cuda()
                random_XB[i] = random_XB[i].cuda()
            random_XA[i] = random_XA[i].unsqueeze(0)
            random_XB[i] = random_XB[i].unsqueeze(0)
            #
            mu_infA, std_infA, logvar_infA = encoderA(random_XA[i])
            mu_infB, std_infB, logvar_infB = encoderB(random_XB[i])
            random_zmu[i], _, _ = apply_poe(
                self.use_cuda, mu_infA, logvar_infA, mu_infB, logvar_infB
            )

        if self.dataset.lower() == 'idaz_elli_3df':
            
            fixed_idxs = [10306, 7246, 21440]
            
            fixed_XA = [0]*len(fixed_idxs)
            fixed_XB = [0]*len(fixed_idxs)
            fixed_zmu = [0]*len(fixed_idxs)
            
            for i, idx in enumerate(fixed_idxs):
                
                fixed_XA[i], fixed_XB[i] = \
                    self.data_loader.dataset.__getitem__(idx)[0:2]
                if self.use_cuda:
                    fixed_XA[i] = fixed_XA[i].cuda()
                    fixed_XB[i] = fixed_XB[i].cuda()
                fixed_XA[i] = fixed_XA[i].unsqueeze(0)
                fixed_XB[i] = fixed_XB[i].unsqueeze(0)
                                
                mu_infA, std_infA, logvar_infA = encoderA(fixed_XA[i])
                mu_infB, std_infB, logvar_infB = encoderB(fixed_XB[i])
                fixed_zmu[i], _, _ = apply_poe( 
                    self.use_cuda, 
                    mu_infA, logvar_infA, mu_infB, logvar_infB
                )
                
            IMG = {}
            for i, idx in enumerate(fixed_idxs):
                IMG['fixed'+str(i)] = \
                    torch.cat([fixed_XA[i], fixed_XB[i]], dim=2)            
            for i in range(num_reps):
                IMG['random'+str(i)] = \
                    torch.cat([random_XA[i], random_XB[i]], dim=2)            

            Z = {}
            for i, idx in enumerate(fixed_idxs):
                Z['fixed'+str(i)] = fixed_zmu[i]
            for i in range(num_reps):
                Z['random'+str(i)] = random_zmu[i]
            
        else:
            
            raise NotImplementedError
            

        WS = torch.ones(IMG['fixed1'].shape)
        if self.use_cuda:
            WS = WS.cuda()


        # do traversal and collect generated images 
        gifs = []
        for key in Z:
            
            z_ori = Z[key]
            
            # traversal over z
            for val in interpolation:
                gifs.append(WS)
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:,row] = val
                    sampleA = torch.sigmoid(decoderA(z)).data
                    sampleB = torch.sigmoid(decoderB(z)).data
                    sample = torch.cat([sampleA, sampleB], dim=2)
                    gifs.append(sample)    
                    
        ####

        # save the generated files, also the animated gifs     
        out_dir = os.path.join(self.output_dir_trvsl, str(iters))
        mkdirs(self.output_dir_trvsl)
        mkdirs(out_dir)
        gifs = torch.cat(gifs)
        gifs = gifs.view( 
            len(Z), 1+self.z_dim, 
            len(interpolation), self.nc, 2*64, 64
        ).transpose(1,2)
        for i, key in enumerate(Z.keys()):
            for j, val in enumerate(interpolation):
                I = torch.cat([IMG[key], gifs[i][j]], dim=0)
                save_image(
                    tensor=I.cpu(),
                    filename=os.path.join(out_dir, '%s_%03d.jpg' % (key,j)),
                    nrow=1+1+self.z_dim, 
                    pad_value=1 )
            # make animated gif
            grid2gif(
                out_dir, key, str(os.path.join(out_dir, key+'.gif')), delay=10
            )

    
    ####
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
            

    ####
    def dump_to_record(self, prn_str):
        
        record = open(self.record_file, 'a')
        record.write('%s\n' % (prn_str,))
        record.close()


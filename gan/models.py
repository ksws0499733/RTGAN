from ast import Pass
import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import Discriminator
from .loss import AdversarialLoss
from .utils import Gradient

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self,gen_weight_path=None, dis_weight_path=None):
        gen_weight_path = gen_weight_path or self.gen_weights_path
        if os.path.exists(gen_weight_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(gen_weight_path)
            else:
                data = torch.load(gen_weight_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        dis_weight_path = dis_weight_path or self.dis_weights_path
        if self.config.MODE == 1 and os.path.exists(dis_weight_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(dis_weight_path)
            else:
                data = torch.load(dis_weight_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])
    

    def save(self, root = None):
        print('\nsaving %s...\n' % self.name)

        if root is not None:
            self.gen_weights_path = os.path.join(root,self.name+ '_gen.pth')
            self.dis_weights_path = os.path.join(root,self.name+ '_dis.pth')
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)

class RailModel(BaseModel):
    def __init__(self, config, generator,gen_dic=None, discriminator=None, numclass=2 ):
        super(RailModel, self).__init__('RailModel',config)
        # generator---moduleList
        # generator input: [image(3)]
        # discriminator input: (iamge(3)+ grad(2) + edge(1))
        # generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = discriminator or Discriminator(in_channels=6, 
                                                        use_sigmoid=config.GAN_LOSS != 'hinge')
        
        generator = generator.cuda()



        if len(config.GPU) > 0:
            generator = nn.DataParallel(generator, config.GPU).cuda()
            discriminator = nn.DataParallel(discriminator, config.GPU).cuda()

        if gen_dic is not None:
            print('----------------',gen_dic.keys())
            if 'generator' in gen_dic.keys():
                print('----------------generator-------')
                generator.load_state_dict(gen_dic['generator'])

        l1_loss = nn.L1Loss().cuda()
        print("\tgan loss:", config.GAN_LOSS)
        
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS).cuda()

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.add_module('softmax1', nn.Softmax(dim=1).cuda())

        self.grad = Gradient()

    def process(self, images, target):
        self.iteration += 1
        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()
        # process outputs
        target = torch.unsqueeze(target,1)

        out_cls, out_gx,out_gy, tar_gx,tar_gy = self(images, target)

        gen_loss = 0
        dis_loss = 0

        output_cls = out_cls[:,1:2]

        # discriminator loss
        dis_input_real = torch.cat(( images, tar_gx,tar_gy, target), dim=1)
        dis_input_fake = torch.cat(( images, out_gx,out_gy, output_cls.detach()), dim=1)
        
        dis_real, dis_real_midfeats = self.discriminator(dis_input_real)      
        dis_fake, dis_fake_midfeats = self.discriminator(dis_input_fake)        
        

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        
        dis_loss = dis_loss + (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = torch.cat((images, out_gx,out_gy, output_cls), dim=1)
        gen_fake, gen_fake_midfeats = self.discriminator(gen_input_fake) 
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss = gen_loss + gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_midfeats)):
            gen_fm_loss = gen_fm_loss + self.l1_loss(gen_fake_midfeats[i], dis_real_midfeats[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss =gen_loss + gen_fm_loss

        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]
        return out_cls, gen_loss, dis_loss, logs

    def forward(self, images, target):
        inputs = images
        out_cls = self.generator(inputs)['cls']
        out_gx,out_gy = self.grad.gauss_gradxy(out_cls[:,1:2])
        tar_gx,tar_gy = self.grad.gauss_gradxy(target)
        return out_cls, out_gx, out_gy, tar_gx, tar_gy

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            pass
            dis_loss.backward()
            self.dis_optimizer.step()

        if gen_loss is not None:
            pass
            gen_loss.backward()
            self.gen_optimizer.step()


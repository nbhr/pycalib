import unittest
import numpy as np
import cv2
import torch
import gzip
import pycalib

class TestPyCalibBa(unittest.TestCase):
    ITER = 20
    BAL_FILENAME = 'kp_dict.txt.gz'

    def test_bal(self):
        with gzip.open(self.BAL_FILENAME) as fp:
            model, masks, pt2ds = pycalib.ba.load_bal(fp)
        print(model)
    
        device = torch.device('cpu')
    #    device = torch.device('cuda')
    
        model = model.to(device)
        masks = masks.to(device)
        pt2ds = pt2ds.to(device)
    
        #optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-2)
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
        criterion = torch.nn.MSELoss()
    
        loss_initial = criterion(model.forward(masks), pt2ds)
        model.plot_fig()[0].savefig('before.png')

        model.train()
        for i in range(self.ITER):
            # 順伝搬
            x = model.forward(masks)
            # 再投影誤差を計算
            loss = criterion(x, pt2ds)
            # print('E_rep[%d] = %f px (mse)' % (i, loss))
            #print(x[0, :], pt2ds[0, :])
            # 勾配をゼロ初期化
            optimizer.zero_grad()
            # 勾配の計算
            loss.backward()
            # パラメータの更新
            optimizer.step()
    
        print('%e -> %e px (mse)' % (loss_initial, loss))
    
        print(model.cpu())
        print("OK: bal")
    

if __name__ == '__main__':
    unittest.main()

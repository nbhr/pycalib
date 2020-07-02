import unittest
import numpy as np
import cv2
import torch
import gzip
import bz2
import pycalib


def cam2torchA(self):
    return torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


def cam2d(self):
    distCoeffs = np.zeros(5)
    distCoeffs[0] = self.distCoeffs[0].data.numpy()
    distCoeffs[1] = self.distCoeffs[1].data.numpy()
    distCoeffs[2] = self.distCoeffs[2].data.numpy()
    distCoeffs[3] = self.distCoeffs[3].data.numpy()
    distCoeffs[4] = self.distCoeffs[4].data.numpy()
    return distCoeffs

def genX(Np):
    X_gt = np.random.rand(3*Np).reshape((3, Np)) + np.array([0, 0, 100]).reshape((3, 1))
    X = X_gt + np.random.normal(0, 0.01, 3 * Np).reshape((3, Np))
    return X_gt, X


def genCamera(X_gt):
    Np = X_gt.shape[1]

    rvec_gt = (np.random.rand(3) - 0.5) * 2
    tvec_gt = (np.random.rand(3) - 0.5) * 2
    a_gt = np.random.normal(1000, 100, 4)
    d_gt = np.random.rand(5)

    A_gt = np.array([[a_gt[0], 0, a_gt[2]], [  0., a_gt[1], a_gt[3]], [  0., 0., 1. ]])
    x_gt, _ = cv2.projectPoints(X_gt.T.reshape((1,Np,3)), rvec_gt, tvec_gt, A_gt, d_gt)
    x_gt = x_gt.reshape(-1, 2).T

    rvec = rvec_gt + (np.random.rand(3) - 0.5) * 0.001
    tvec = tvec_gt + (np.random.rand(3) - 0.5) * 0.001
    a = a_gt + np.random.normal(0, 10, 4)
    d = d_gt + np.random.normal(0, 0.001, 5)
    x = x_gt + np.random.normal(0, 3, x_gt.shape)

    return rvec_gt, tvec_gt, a_gt, d_gt, x_gt, rvec, tvec, a, d, x

class TestPyCalibBa(unittest.TestCase):
    N = 20
    BAL_FILENAME = 'problem-49-7776-pre.txt.bz2'
    ITER = 100

    def test_rodrigues(self):
        for i in range(self.N):
            # generate a random rodrigues vector
            x = np.random.rand(3).reshape((3, 1))
            # convert to 3x3 matrix form by OpenCV
            y, _ = cv2.Rodrigues(x)
            # and again back to vector form by OpenCV
            yx, _ = cv2.Rodrigues(y)
            # also convert to 3x3 matrix form by myself
            z = pycalib.ba.rvec2mat(torch.from_numpy(x))
            # the outputs shold be identical up to eps
            self.assertTrue(np.allclose(x, yx))
            self.assertTrue(np.allclose(y, z))

    def test_distort(self):
        for i in range(self.N):
            x = np.random.rand(3).reshape((1,1,3))
            x[0, 0, 2] = 1
            dist = np.random.rand(5)
            y, _ = cv2.projectPoints(x, np.zeros(3), np.zeros(3), np.eye(3), dist)
            #print(x, y)
            z = pycalib.ba.distort(torch.from_numpy(x), torch.from_numpy(dist))
            self.assertTrue(np.allclose(y, z))

    def test_projectPoints(self):
        # project a random single point
        for i in range(self.N):
            x = np.random.rand(3).reshape((3, 1))
            rvec = np.random.rand(3)
            tvec = np.random.rand(3)
            fx = np.random.rand(1)
            fy = np.random.rand(1)
            cx = np.random.rand(1)
            cy = np.random.rand(1)
            dist = np.random.rand(5)
            A = np.array([[fx[0], 0, cx[0]], [0, fy[0], cy[0]], [0, 0, 1]])
            #print(A)
            y, _ = cv2.projectPoints(x.reshape((1,1,3)), rvec, tvec, A, dist)
            y = y.reshape(-1, 2).T
            #print(x, y)
            z = pycalib.ba.projectPoints(torch.from_numpy(x), torch.from_numpy(rvec), torch.from_numpy(tvec), torch.from_numpy(fx), torch.from_numpy(fy), torch.from_numpy(cx), torch.from_numpy(cy), torch.from_numpy(dist))
            #print(y)
            #print(z)
            #print('--')
            self.assertTrue(np.allclose(y, z))


    def test_project2(self):
        # project a batch of points
        X = np.random.rand(3*self.N).reshape((3, self.N))
        rvec = np.random.rand(3)
        tvec = np.random.rand(3)
        a = np.random.rand(4) * 1000
        A = np.array([[a[0], 0, a[2]],
        [  0., a[1], a[3]],
        [  0., 0., 1. ]])
        d = np.random.rand(5)

        y, _ = cv2.projectPoints(X.T.reshape((1, self.N, 3)), rvec, tvec, A, d)
        y = y.reshape(-1, 2).T
        #print(y)
        z = pycalib.ba.projectPoints(torch.from_numpy(X), torch.from_numpy(rvec), torch.from_numpy(tvec), torch.from_numpy(np.array(A[0,0])), torch.from_numpy(np.array(A[1,1])), torch.from_numpy(np.array(A[0,2])), torch.from_numpy(np.array(A[1,2])), torch.from_numpy(d))
        #print(z)
        self.assertTrue(np.allclose(y, z))


    def test_camera(self):
        idx = [1, 3, 5, 6]
        X = np.random.rand(3*self.N).reshape((3, self.N))

        cam = pycalib.ba.Camera(np.random.rand(3), np.random.rand(3), np.random.rand(1), np.random.rand(1), np.random.rand(1), np.random.rand(1), np.random.rand(5))

        y, _ = cv2.projectPoints(X.T.reshape((1,self.N,3))[:,idx,:], cam.rvec.data.numpy(), cam.tvec.data.numpy(), cam2torchA(cam).data.numpy(), cam2d(cam))
        y = y.reshape(-1, 2).T
        z = cam.forward(torch.from_numpy(X)[:,idx])
        self.assertTrue(np.allclose(y, z.detach().numpy()))


    def test_torch(self):
        # generate 3D points with (initial value) / without noise (ground truth)
        X_gt, X = genX(self.N)

        # generate cameras and 2D projections
        Nc = 3
        cameras = []
        masks = []
        pt2ds = []
        for i in range(Nc):
            rvec_gt, tvec_gt, a_gt, d_gt, x_gt, rvec, tvec, a, d, x = genCamera(X_gt)
            cam = pycalib.ba.Camera(rvec, tvec, a[0], a[1], a[2], a[3], d)
            cameras.append(cam)

            # radomly generate a visibility mask (which points are visible from the camera)
            mask = (np.random.rand(self.N) < 0.5).astype(np.int)
            print('cam%d mask = %s' % (i, mask))
            masks.append(mask)
            # and also remove 2D points not visible from the camera
            pt2ds.append((x[:, mask.nonzero()]).squeeze().T)
        masks = torch.from_numpy(np.vstack(masks))
        pt2ds = torch.from_numpy(np.vstack(pt2ds))
        # we do not update the mask and the 2D observations
        masks.requires_grad = False
        pt2ds.requires_grad = False
        self.assertTrue(masks.shape == (Nc, self.N))

        # 投影モデル
        model = pycalib.ba.Projection(cameras, X)
        #print(model)

        optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()

        loss_initial = criterion(model.forward(masks), pt2ds)

        model.train()
        for i in range(self.ITER):
            # 順伝搬
            x = model.forward(masks)
            # 再投影誤差を計算
            loss = criterion(x, pt2ds)
            #print(loss)
            # 勾配をゼロ初期化
            optimizer.zero_grad()
            # 勾配の計算
            loss.backward()
            # パラメータの更新
            optimizer.step()

        print('%e -> %e px (mse)' % (loss_initial, loss))

        self.assertTrue(loss < loss_initial)



    def test_bal(self):
        with bz2.open(self.BAL_FILENAME) as fp:
            model, masks, pt2ds = pycalib.ba.load_bal(fp)
        #print(model)

        device = torch.device('cpu')
    #    device = torch.device('cuda')

        model = model.to(device)
        masks = masks.to(device)
        pt2ds = pt2ds.to(device)

        #optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-2)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        criterion = torch.nn.MSELoss()

        loss_initial = criterion(model.forward(masks), pt2ds)

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

        self.assertTrue(loss < loss_initial)
        #print(model.cpu())
        #print("OK: bal")


if __name__ == '__main__':
    unittest.main()

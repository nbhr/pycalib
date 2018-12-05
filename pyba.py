import torch
from torch.nn import Module, ModuleList, ParameterList
from torch.nn.parameter import Parameter
import torch.optim as optim
import cv2
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

def skew(x):
    """
    Return skew-symmetric matrix
    """
    return torch.tensor([[
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ]], device=x.device)

def rvec2mat(rvec):
    """
    Convert Rodrigues vector to rotation matrix.
    This is identical to vector-to-matrix convertion by cv2.Rodrigues().

    Parameters
    ----------
    rvec : 1x3 or 3x1 torch.tensor
        Rodrigues vector

    Returns
    -------
    rmat : 3x3 torch.tensor
        Rotation matrix
    """

    # https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209/2
    device = rvec.device
    rvec = rvec.reshape(3)
    theta = torch.norm(rvec, 2)
    if theta != 0:
        rvec = rvec / theta
    #else:
    #   No need to return here, since the following equation automatically yields
    #   eye(3) for rvec == 0. If returns, it disconnects the autograd graph.
    #   return torch.eye(3)

    c_th = torch.cos(theta)
    s_th = torch.sin(theta)
    return c_th * torch.eye(3).to(device) + (1-c_th) * torch.ger(rvec, rvec) + s_th * skew(rvec)


def distort(pt3d, d):
    """
    Distort points in normalized camera. Identical to cv2.projectPoints(pt3d, np.zeros(3),
    np.zeros(3), np.eye(3), distCoeffs).

    Parameters
    ----------
    pt3d : 3xN torch.tensor
        Ideal positions (not necessarily in normalized camera, i.e., pt3d[3,:] is
        not required to be 1 since this function first does x/=z and y/=z anyway)
    d : list of torch.tensor
        Distortion coeffs (k1, k2, p1, p2, k3)

    Returns
    -------
    n : 2xN torch.tensor
        Distorted points in normalized camera
    """
    pt3d = pt3d.reshape((3, -1))
    x1 = pt3d[0, :] / pt3d[2, :]
    y1 = pt3d[1, :] / pt3d[2, :]
    r2 = (x1*x1 + y1*y1)
    r4 = r2*r2
    r6 = r4*r2
    kdist = 1 + d[0]*r2 + d[1]*r4 + d[4]*r6
    pdist = 2*x1*y1
    x2 = x1*kdist + d[2]*pdist + d[3]*(r2+2*x1*x1)
    y2 = y1*kdist + d[3]*pdist + d[2]*(r2+2*y1*y1)
    return x2, y2


def projectPoints(pt3d, rvec, tvec, fx, fy, cx, cy, distCoeffs):
    """
    Project 3D points in WCS to image. Identical to cv2.projectPoints(pt3d, rvec, tvec,
    cameraMatrix, distCoeffs), where cameraMatrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].

    Parameters
    ----------
    pt3d : 3xN torch.tensor
        3D points in world coordinate system
    rvec : 1x3 or 3x1 torch.tensor
        Rodrigues vector representing world-to-camera conversion
    tvec : 1x3 or 3x1 torch.tensor
        Translation vector representing world-to-camera conversion
    fx : torch.tensor
        Focal length (aka alpha)
    fy : torch.tensor
        Focal length (aka beta)
    cx : torch.tensor
        principal point (aka u0)
    cy : torch.tensor
        principal point (aka v0)
    distCoeffs : 1x5 or 5x1 torch.tensor
        Distortion coeffs (k1, k2, p1, p2, k3)

    Returns
    -------
    pt2d : 2xN torch.tensor
        Image points
    """
    R = rvec2mat(rvec)
    u, v = distort(torch.matmul(R, pt3d) + tvec.unsqueeze(1), distCoeffs)
    return torch.stack([fx*u + cx, fy*v + cy])


class Camera(Module):
    """
    Camera

    Attributes
    ----------
    rvec : 1x3 or 3x1 torch.tensor
        Rodrigues vector representing world-to-camera conversion
    tvec : 1x3 or 3x1 torch.tensor
        Translation vector representing world-to-camera conversion
    fx : torch.tensor
        Focal length (aka alpha)
    fy : torch.tensor
        Focal length (aka beta)
    cx : torch.tensor
        principal point (aka u0)
    cy : torch.tensor
        principal point (aka v0)
    distCoeffs : 1x5 or 5x1 torch.tensor
        Distortion coeffs (k1, k2, p1, p2, k3)
    """

    def __init__(self, rvec, tvec, fx, fy, cx, cy, distCoeffs):
        """
        Parameters
        ----------
        rvec : 1x3 or 3x1 numpy.ndarray
            Rodrigues vector representing world-to-camera conversion
        tvec : 1x3 or 3x1 numpy.ndarray
            Translation vector representing world-to-camera conversion
        fx : double
            Focal length (aka alpha)
        fy : double or None
            Focal length (aka beta)
        cx : double
            principal point (aka u0)
        cy : double
            principal point (aka v0)
        distCoeffs : numpy.ndarray of size 1, 2, ..., or 5
            Distortion coeffs (k1, k2, p1, p2, k3). If the size is less than 5, the higher coeffs are fixed to zero.
        """
        super(Camera, self).__init__()

        def get_d(distCoeffs, i):
            if distCoeffs is not None and distCoeffs.size > i:
                return Parameter(torch.tensor(distCoeffs[i]))
            else:
                return Parameter(torch.tensor(0.0), requires_grad=False)

        self.rvec = Parameter(torch.from_numpy(rvec))
        self.tvec = Parameter(torch.from_numpy(tvec))
        self.fx = Parameter(torch.tensor(np.double(fx)))
        self.fy = Parameter(torch.tensor(np.double(fy))) if fy is not None else self.fx
        self.cx = Parameter(torch.tensor(np.double(cx))) if cx is not None else Parameter(torch.tensor(0.0, requires_grad=False), requires_grad=False)
        self.cy = Parameter(torch.tensor(np.double(cy))) if cy is not None else Parameter(torch.tensor(0.0, requires_grad=False), requires_grad=False)
        self.k1 = get_d(distCoeffs, 0)
        self.k2 = get_d(distCoeffs, 1)
        self.p1 = get_d(distCoeffs, 2)
        self.p2 = get_d(distCoeffs, 3)
        self.k3 = get_d(distCoeffs, 4)

        self.distCoeffs = [self.k1, self.k2, self.p1, self.p2, self.k3]

    def forward(self, pt3d):
        """
        Project 3D points in WCS to the image plane

        Parameters
        ----------
        pt3d : 3xN torch.tensor
            3D points in world coordinate system

        Returns
        -------
        pt2d : 2xN torch.tensor
            Image points
        """
        return projectPoints(pt3d, self.rvec, self.tvec, self.fx, self.fy, self.cx, self.cy, self.distCoeffs)

    def extra_repr(self):
        return 'rvec={},\ntvec={},\nfx={}, fy={}, cx={}, cy={},\ndist=[{}, {}, {}, {}, {}]'.format(
            self.rvec.data.numpy(), self.tvec.data.numpy(), self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2, self.k3
        )

class Projection(Module):
    """
    Projection of 3D points to multiple cameras with visibility mask
    """
    def __init__(self, cameras, pt3d):
        """
        Parameters
        ----------
        cameras : list of Camera objects
            Cameras observing the 3D points in WCS
        pt3d : 3xN torch.tensor
            3D points in WCS
        """
        super(Projection, self).__init__()
        self.Nc = len(cameras)
        self.Np = pt3d.shape[1]
        self.cameras = ModuleList(cameras)
        self.pt3d = Parameter(torch.from_numpy(pt3d))
        assert pt3d.ndim == 2
        assert pt3d.shape[0] == 3

    def forward(self, mask):
        """
        Project 3D points specified by visibility mask

        Parameters
        ----------
        mask : Nc x Np torch.tensor or numpy.ndarray
            Binary visibility mask. If (i, j) is non-zero, the j-th 3D point is projected to
            the i-th camera.

        Returns
        -------
        pt2d : Mx2 torch.tensor
            (u,v) positions of M projected points, where M = np.count_nonzero(mask).
            The size is Mx2, i.e., [[u0, v0], [u1, v1], [u2, v2], ...]
        """
        # mask is a binary Nc x Np matrix
        assert mask.dim() == 2
        assert mask.shape == (self.Nc, self.Np)
        rep = []
        for c in range(self.Nc):
            idx = mask[c, :].nonzero()
            X = self.pt3d[:, idx].squeeze()
            x = self.cameras[c](X)
            rep.append(x.transpose(0, 1))
            #rep.append(x)
        return torch.cat(rep)#.view(-1)


def load_bal(filename):
    # http://grail.cs.washington.edu/projects/bal/
    with open(filename, 'r') as fp:
        # load all lines
        lines = fp.readlines()

        # num of cameras / points / observations from the 1st line
        num_cameras, num_points, num_observations = [int(x) for x in lines[0].strip().split()]
        curr = 1

        # 2D observations
        observations = np.array([np.loadtxt(lines[i:i+1]) for i in np.arange(curr, curr+num_observations)])
        assert observations.shape == (num_observations, 4)
        curr += num_observations

        # Cameras
        cameras = np.array([np.loadtxt(lines[i:i+9]) for i in np.arange(curr, curr+num_cameras*9, 9)])
        assert cameras.shape == (num_cameras, 9)
        curr += num_cameras*9

        # 3D points
        points = np.array([np.loadtxt(lines[i:i+3]) for i in np.arange(curr, curr+num_points*3, 3)])
        assert points.shape == (num_points, 3)

    # Sort observations by cameras and then points
    observations = observations[np.lexsort((observations[:, 1], observations[:, 0]))]
    # Mask
    mask = np.zeros((num_cameras, num_points), dtype=np.int)
    mask[observations[:, 0].astype(np.int), observations[:, 1].astype(np.int)] = 1
    assert np.count_nonzero(mask) == num_observations

    cams = []
    for i in range(num_cameras):
        cam = Camera(cameras[i, 0:3], cameras[i, 3:6], cameras[i, 6], None, 0, 0, cameras[i, 7:9])
        cam.cx.requires_grad = True
        cam.cy.requires_grad = True
        cam.p1.requires_grad = True
        cam.p2.requires_grad = True
        cam.k3.requires_grad = True
        cams.append(cam)

    masks = torch.from_numpy(mask)
    pt2ds = torch.from_numpy(-observations[:, 2:])
    masks.requires_grad = False
    pt2ds.requires_grad = False
    assert masks.shape == (num_cameras, num_points)

    model = Projection(cams, points.T)

    return model, masks, pt2ds



###################


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

def test_rodrigues():
    for i in range(10):
        x = np.random.rand(3).reshape((3, 1))
        y, _ = cv2.Rodrigues(x)
        yx, _ = cv2.Rodrigues(y)
        z = rvec2mat(torch.from_numpy(x))
        #print(x)
        assert np.allclose(x, yx)
        assert np.allclose(y, z)
    print("OK: test_rodrigues")


def test_distort():
    for i in range(10):
        x = np.random.rand(3).reshape((1,1,3))
        x[0, 0, 2] = 1
        dist = np.random.rand(5)
        y, _ = cv2.projectPoints(x, np.zeros(3), np.zeros(3), np.eye(3), dist)
        #print(x, y)
        z = distort(torch.from_numpy(x), torch.from_numpy(dist))
        assert np.allclose(y, z)
        #print(z)
        #print('--')
    print("OK: test_distort")


def test_project():
    for i in range(10):
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
        z = projectPoints(torch.from_numpy(x), torch.from_numpy(rvec), torch.from_numpy(tvec), torch.from_numpy(fx), torch.from_numpy(fy), torch.from_numpy(cx), torch.from_numpy(cy), torch.from_numpy(dist))
        #print(y)
        #print(z)
        #print('--')
        assert np.allclose(y, z)
    print("OK: test_project")


def test_project2():
    Np = 10
    X = np.random.rand(3*Np).reshape((3, Np))
    rvec = np.random.rand(3)
    tvec = np.random.rand(3)
    a = np.random.rand(4) * 1000
    A = np.array([[a[0], 0, a[2]],
     [  0., a[1], a[3]],
     [  0., 0., 1. ]])
    d = np.random.rand(5)

    y, _ = cv2.projectPoints(X.T.reshape((1,Np,3)), rvec, tvec, A, d)
    y = y.reshape(-1, 2).T
    #print(y)
    z = projectPoints(torch.from_numpy(X), torch.from_numpy(rvec), torch.from_numpy(tvec), torch.from_numpy(np.array(A[0,0])), torch.from_numpy(np.array(A[1,1])), torch.from_numpy(np.array(A[0,2])), torch.from_numpy(np.array(A[1,2])), torch.from_numpy(d))
    #print(z)
    assert np.allclose(y, z)
    print("OK: test_project2")

def test_camera():
    Np = 10
    idx = [1, 3, 5, 6]
    X = np.random.rand(3*Np).reshape((3, Np))

    cam = Camera(np.random.rand(3), np.random.rand(3), np.random.rand(1), np.random.rand(1), np.random.rand(1), np.random.rand(1), np.random.rand(5))

    y, _ = cv2.projectPoints(X.T.reshape((1,Np,3))[:,idx,:], cam.rvec.data.numpy(), cam.tvec.data.numpy(), cam2torchA(cam).data.numpy(), cam2d(cam))
    y = y.reshape(-1, 2).T
    z = cam.forward(torch.from_numpy(X)[:,idx])
    assert np.allclose(y, z.detach().numpy())
    print("OK: test_camera")


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


def test_torch():
    # 真値と初期値を作る
    Np = 10
    X_gt, X = genX(Np)

    # カメラと観測２Ｄ
    Nc = 3
    cameras = []
    masks = []
    pt2ds = []
    for i in range(Nc):
        rvec_gt, tvec_gt, a_gt, d_gt, x_gt, rvec, tvec, a, d, x = genCamera(X_gt)
        cam = Camera(rvec, tvec, a[0], a[1], a[2], a[3], d)
        cameras.append(cam)

        # どの点を観測しているかをランダムに決める
        mask = (np.random.rand(Np) < 0.5).astype(np.int)
        print('cam%d mask = %s' % (i, mask))
        masks.append(mask)
        # 2Dの点もmaskに合わせて間引く
        pt2ds.append((x[:, mask.nonzero()]).squeeze().T)
    masks = torch.from_numpy(np.vstack(masks))
    pt2ds = torch.from_numpy(np.vstack(pt2ds))
    masks.requires_grad = False
    pt2ds.requires_grad = False
    assert masks.shape == (Nc, Np)

    # 投影モデル
    model = Projection(cameras, X)
    print(model)

    optimizer = optim.Adadelta(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()

    model.train()
    for i in range(10):
        # 順伝搬
        x = model.forward(masks)
        # 再投影誤差を計算
        loss = criterion(x, pt2ds)
        print(loss)
        # 勾配をゼロ初期化
        optimizer.zero_grad()
        # 勾配の計算
        loss.backward()
        # パラメータの更新
        optimizer.step()
    print("OK: torch")


def test_bal(filename):
    model, masks, pt2ds = load_bal(filename)
    print(model)

    device = torch.device('cpu')
#    device = torch.device('cuda')

    model = model.to(device)
    masks = masks.to(device)
    pt2ds = pt2ds.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()

    model.train()
    for i in range(1000):
        # 順伝搬
        x = model.forward(masks)
        # 再投影誤差を計算
        loss = criterion(x, pt2ds)
        print('E_rep[%d] = %f px (mse)' % (i, loss))
        #print(x[0, :], pt2ds[0, :])
        # 勾配をゼロ初期化
        optimizer.zero_grad()
        # 勾配の計算
        loss.backward()
        # パラメータの更新
        optimizer.step()
    print(model.cpu())
    print("OK: bal")


test_rodrigues()
test_distort()
test_project()
test_project2()
test_camera()
test_torch()
test_bal('problem-49-7776-pre.txt')

exit()


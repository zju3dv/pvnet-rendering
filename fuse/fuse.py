import pickle

import numpy as np
import os
import time
import cv2

from glob import glob
from PIL import ImageFile, Image
from plyfile import PlyData
from skimage.io import imread, imsave
from concurrent.futures import ProcessPoolExecutor

from config import cfg


class ModelAligner(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {
        # 'cat': np.array([-0.00577495, -0.01259045, -0.04062323])
    }
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]]),
        # 'blender': np.array([[280.0, 0.0, 128.0],
        #                      [0.0, 280.0, 128.0],
        #                      [0.0, 0.0, 1.0]]),
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])
    }

    def __init__(self, class_type,linemod_dir,linemod_orig_dir):
        self.class_type = class_type
        self.blender_model_path = os.path.join(linemod_dir,'{}/{}.ply'.format(class_type, class_type))
        self.orig_model_path = os.path.join(linemod_orig_dir,'{}/mesh.ply'.format(class_type))
        self.orig_old_model_path = os.path.join(linemod_orig_dir,'{}/OLDmesh.ply'.format(class_type))
        self.transform_dat_path = os.path.join(linemod_orig_dir,'{}/transform.dat'.format(class_type))

        self.R_p2w,self.t_p2w,self.s_p2w=self.setup_p2w_transform()

    @staticmethod
    def setup_p2w_transform():
        transform1 = np.array([[0.161513626575, -0.827108919621, 0.538334608078, -0.245206743479],
                               [-0.986692547798, -0.124983474612, 0.104004733264, -0.050683632493],
                               [-0.018740313128, -0.547968924046, -0.836288750172, 0.387638419867]])
        transform2 = np.array([[0.976471602917, 0.201606079936, -0.076541729271, -0.000718327821],
                               [-0.196746662259, 0.978194475174, 0.066531419754, 0.000077120210],
                               [0.088285841048, -0.049906700850, 0.994844079018, -0.001409600372]])

        R1 = transform1[:, :3]
        t1 = transform1[:, 3]
        R2 = transform2[:, :3]
        t2 = transform2[:, 3]

        # printer system to world system
        t_p2w = np.dot(R2, t1) + t2
        R_p2w = np.dot(R2, R1)
        s_p2w = 0.85
        return R_p2w,t_p2w,s_p2w

    def pose_p2w(self,RT):
        t,R=RT[:,3],RT[:,:3]
        R_w2c=np.dot(R, self.R_p2w.T)
        t_w2c=-np.dot(R_w2c,self.t_p2w)+self.s_p2w*t
        return np.concatenate([R_w2c,t_w2c[:,None]],1)

    @staticmethod
    def load_ply_model(model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        return np.stack([x, y, z], axis=-1)

    def read_transform_dat(self):
        transform_dat = np.loadtxt(self.transform_dat_path, skiprows=1)[:, 1]
        transform_dat = np.reshape(transform_dat, newshape=[3, 4])
        return transform_dat

    def load_orig_model(self):
        if os.path.exists(self.orig_model_path):
            return self.load_ply_model(self.orig_model_path) / 1000.
        else:
            transform = self.read_transform_dat()
            old_model = self.load_ply_model(self.orig_old_model_path) / 1000.
            old_model = np.dot(old_model, transform[:, :3].T) + transform[:, 3]
            return old_model

    def get_translation_transform(self):
        if self.class_type in self.translation_transforms:
            return self.translation_transforms[self.class_type]

        blender_model = self.load_ply_model(self.blender_model_path)
        orig_model = self.load_orig_model()
        blender_model = np.dot(blender_model, self.rotation_transform.T)
        translation_transform = np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0)
        self.translation_transforms[self.class_type] = translation_transform

        return translation_transform

class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}
    class_type_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010'
    }
    blender_models={}

    def __init__(self, class_type,linemod_dir,linemod_orig_dir):
        self.class_type = class_type
        self.blender_model_path = os.path.join(linemod_dir,'{}/{}.ply'.format(class_type, class_type))
        self.orig_model_path = os.path.join(linemod_orig_dir,'{}/mesh.ply'.format(class_type))
        self.model_aligner = ModelAligner(class_type,linemod_dir,linemod_orig_dir)

    def orig_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        tra = tra + np.dot(rot, self.model_aligner.get_translation_transform())
        rot = np.dot(rot, self.rotation_transform)
        return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def read_rgb_np(rgb_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img,np.uint8)
    return img

def read_mask_np(mask_path):
    mask = Image.open(mask_path)
    mask_seg = np.array(mask).astype(np.int32)
    return mask_seg

def read_pose(rot_path, tra_path):
    rot = np.loadtxt(rot_path, skiprows=1)
    tra = np.loadtxt(tra_path, skiprows=1) / 100.
    return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

def collect_train_val_test_info(linemod_dir,cls_name):
    with open(os.path.join(linemod_dir,cls_name,'test.txt'),'r') as f:
        test_fns=[line.strip().split('/')[-1] for line in f.readlines()]

    with open(os.path.join(linemod_dir,cls_name,'train.txt'),'r') as f:
        train_fns=[line.strip().split('/')[-1] for line in f.readlines()]

    return test_fns, train_fns

def collect_linemod_set_info(linemod_dir,linemod_cls_name,linemod_orig_dir,cache_dir='./'):
    database=[]
    if os.path.exists(os.path.join(cache_dir,'{}_info.pkl').format(linemod_cls_name)):
        return read_pickle(os.path.join(cache_dir,'{}_info.pkl').format(linemod_cls_name))

    _,train_fns=collect_train_val_test_info(linemod_dir,linemod_cls_name)
    print('begin generate database {}'.format(linemod_cls_name))
    rgb_dir=os.path.join(linemod_dir,linemod_cls_name,'JPEGImages')
    msk_dir=os.path.join(linemod_dir,linemod_cls_name,'mask')
    rt_dir = os.path.join(linemod_orig_dir, linemod_cls_name, 'data')
    img_num=len(os.listdir(rgb_dir))
    for k in range(img_num):
        data={}
        data['rgb_pth']=os.path.join(rgb_dir, '{:06}.jpg'.format(k))
        data['dpt_pth']=os.path.join(msk_dir, '{:04}.png'.format(k))
        if data['rgb_pth'].split('/')[-1] not in train_fns: continue

        pose=read_pose(os.path.join(rt_dir, 'rot{}.rot'.format(k)),
                       os.path.join(rt_dir, 'tra{}.tra'.format(k)))
        pose_transformer = PoseTransformer(linemod_cls_name, linemod_dir, linemod_orig_dir)
        data['RT'] = pose_transformer.orig_pose_to_blender_pose(pose).astype(np.float32)
        database.append(data)

    print('success generate database {} len {}'.format(linemod_cls_name,len(database)))
    save_pickle(database,os.path.join(cache_dir,'{}_info.pkl').format(linemod_cls_name))
    return database

def randomly_read_background(background_dir,cache_dir):
    if os.path.exists(os.path.join(cache_dir,'background_info.pkl')):
        fns=read_pickle(os.path.join(cache_dir,'background_info.pkl'))
    else:
        fns=glob(os.path.join(background_dir,'*.jpg'))+glob(os.path.join(background_dir,'*.png'))
        save_pickle(fns,os.path.join(cache_dir,'background_info.pkl'))
    return imread(fns[np.random.randint(0,len(fns))])

def prepare_dataset_parallel(output_dir, linemod_dir, linemod_orig_dir, fuse_num, background_dir, cache_dir, worker_num=8):
    exector=ProcessPoolExecutor(max_workers=worker_num)
    futures=[]

    for cls_name in linemod_cls_names:
        collect_linemod_set_info(linemod_dir,cls_name,linemod_orig_dir,cache_dir)
    randomly_read_background(background_dir,cache_dir)

    for idx in np.arange(fuse_num):
        seed=np.random.randint(5000)
        futures.append(exector.submit(
            prepare_dataset_single,output_dir,idx, linemod_cls_names, linemod_dir, linemod_orig_dir, background_dir,cache_dir, seed))

    for f in futures:
        f.result()


def prepare_dataset_single(output_dir,idx,linemod_cls_names,linemod_dir,linemod_orig_dir,background_dir,cache_dir,seed):
    time_begin=time.time()
    np.random.seed(seed)
    rgbs,masks,begins,poses=[],[],[],[]
    image_dbs={}
    for cls_id,cls_name in enumerate(linemod_cls_names):
        image_dbs[cls_id]=collect_linemod_set_info(linemod_dir,cls_name,linemod_orig_dir,cache_dir)

    for cls_id,cls_name in enumerate(linemod_cls_names):
        rgb, mask, begin, pose=randomly_sample_foreground(image_dbs[cls_id], linemod_dir)
        mask*=cls_id+1
        rgbs.append(rgb)
        masks.append(mask)
        begins.append(begin)
        poses.append(pose)

    background=randomly_read_background(background_dir,cache_dir)

    fuse_img, fuse_mask, fuse_begins= fuse_regions(rgbs, masks, begins, background, 480, 640)

    save_fuse_data(output_dir, idx, fuse_img, fuse_mask, fuse_begins, poses)
    print('{} cost {} s'.format(idx,time.time()-time_begin))

def fuse_regions(rgbs,masks,begins,background,th,tw):
    fuse_order=np.arange(len(rgbs))
    np.random.shuffle(fuse_order)
    fuse_img=background
    fuse_img=cv2.resize(fuse_img,(tw,th),interpolation=cv2.INTER_LINEAR)
    fuse_mask=np.zeros([fuse_img.shape[0],fuse_img.shape[1]],np.int32)
    for idx in fuse_order:
        rh,rw=masks[idx].shape
        bh=np.random.randint(0,fuse_img.shape[0]-rh)
        bw=np.random.randint(0,fuse_img.shape[1]-rw)

        silhouette=masks[idx]>0
        out_silhouette=np.logical_not(silhouette)
        fuse_mask[bh:bh+rh,bw:bw+rw]*=out_silhouette.astype(fuse_mask.dtype)
        fuse_mask[bh:bh+rh,bw:bw+rw]+=masks[idx]

        fuse_img[bh:bh+rh,bw:bw+rw]*=out_silhouette.astype(fuse_img.dtype)[:,:,None]
        fuse_img[bh:bh+rh,bw:bw+rw]+=rgbs[idx]

        begins[idx][0]=-begins[idx][0]+bh
        begins[idx][1]=-begins[idx][1]+bw

    return fuse_img,fuse_mask,begins

def randomly_sample_foreground(image_db,linemod_dir):
    idx=np.random.randint(0,len(image_db))
    rgb_pth=os.path.join(linemod_dir,image_db[idx]['rgb_pth'])
    dpt_pth=os.path.join(linemod_dir,image_db[idx]['dpt_pth'])
    rgb = read_rgb_np(rgb_pth)
    mask = read_mask_np(dpt_pth)
    mask=np.sum(mask,2)>0
    mask=np.asarray(mask,np.int32)

    hs,ws=np.nonzero(mask)
    hmin,hmax=np.min(hs),np.max(hs)
    wmin,wmax=np.min(ws),np.max(ws)

    mask=mask[hmin:hmax,wmin:wmax]
    rgb=rgb[hmin:hmax,wmin:wmax]

    rgb*=mask.astype(np.uint8)[:,:,None]
    begin=[hmin,wmin]
    pose=image_db[idx]['RT']

    return rgb, mask, begin, pose

def save_fuse_data(output_dir, idx, fuse_img, fuse_mask, fuse_begins, fuse_poses):
    os.makedirs(output_dir, exist_ok=True)
    imsave(os.path.join(output_dir,'{}_rgb.jpg'.format(idx)),fuse_img)
    fuse_mask=fuse_mask.astype(np.uint8)
    imsave(os.path.join(output_dir,'{}_mask.png'.format(idx)),fuse_mask)
    save_pickle([np.asarray(fuse_begins,np.int32), np.asarray(fuse_poses,np.float32)],
                os.path.join(output_dir,'{}_info.pkl'.format(idx)))

def randomly_sample_foreground_ycb(image_db, ycb_dir, ycb_cls_idx):
    idx=np.random.randint(0,len(image_db.train_real_set))
    rgb_pth=os.path.join(ycb_dir, image_db.train_real_set[idx]['rgb_pth'])
    msk_pth=os.path.join(ycb_dir, image_db.train_real_set[idx]['msk_pth'])

    rgb = read_rgb_np(rgb_pth)
    mask = read_mask_np(msk_pth)
    mask = mask == ycb_cls_idx
    if len(mask.shape)>2: mask=np.sum(mask,2)>0
    mask=np.asarray(mask,np.int32)

    hs,ws=np.nonzero(mask)
    if len(hs)==0:
        print('zero size')
        raise RuntimeError
    hmin,hmax=np.min(hs),np.max(hs)
    wmin,wmax=np.min(ws),np.max(ws)

    mask=mask[hmin:hmax,wmin:wmax]
    rgb=rgb[hmin:hmax,wmin:wmax]

    rgb*=mask.astype(np.uint8)[:,:,None]
    begin=[hmin,wmin]
    pose=image_db.train_real_set[idx]['pose']
    K=image_db.train_real_set[idx]['K']

    return rgb, mask, begin, pose, K

linemod_cls_names=['ape','cam','cat','duck','glue','iron','phone', 'benchvise','can','driller','eggbox','holepuncher','lamp']


def run():
    output_dir='tmp/'
    linemod_dir=cfg.LINEMOD
    linemod_orig_dir=cfg.LINEMOD_ORIG
    background_dir=os.path.join(cfg.SUN, "JPEGImages")
    cache_dir='./'
    fuse_num=10000
    worker_num=2
    prepare_dataset_parallel(output_dir, linemod_dir, linemod_orig_dir, fuse_num, background_dir, cache_dir, worker_num)


if __name__=="__main__":
    output_dir='tmp/'
    linemod_dir='/home/liuyuan/data/LINEMOD'
    linemod_orig_dir='/home/liuyuan/data/LINEMOD_ORIG'
    background_dir='/home/liuyuan/data/SUN2012pascalformat/JPEGImages'
    cache_dir='./'
    fuse_num=10000
    worker_num=2
    prepare_dataset_parallel(output_dir, linemod_dir, linemod_orig_dir, fuse_num, background_dir, cache_dir, worker_num)

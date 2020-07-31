import pdb
from InsightFace.data.config import get_config
import argparse
from InsightFace.models.Learner import face_learner
from InsightFace.data.data_pipe import get_val_pair
from torchvision import transforms as trans

conf = get_config(training=False)
learner = face_learner(conf, inference=True)

learner.load_state(conf, 'ir_se50.pth', model_only=True, from_save_folder=True)

vgg2_fp, vgg2_fp_issame = get_val_pair(conf.emore_folder, 'vgg2_fp')
accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, vgg2_fp, vgg2_fp_issame, nrof_folds=10, tta=True)
print('vgg2_fp - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
trans.ToPILImage()(roc_curve_tensor)


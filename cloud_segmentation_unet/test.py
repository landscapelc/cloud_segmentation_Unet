from seg_unet import unet
from dataProcess import testGenerator, saveResult, color_dict
import os
from keras.models import load_model
#  训练模型保存地址
model_path = "./Model/1.hdf5"
#  测试数据路径
test_iamge_path = "/media/liaolingcen/e25ccc15-5e77-4bf9-8ab3-057029fd14dd/landsat_cloud/dataset_seg_1_8/test/"
#  结果保存路径
save_path = "./predict/"
#  测试数据数目
test_num = len(os.listdir(test_iamge_path))
#  类的数目(包括背景)
classNum = 4
#  模型输入图像大小
input_size = (512, 512, 3)
#  生成图像大小
output_size = (512, 512)
#  训练数据标签路径
train_label_path = "/media/liaolingcen/e25ccc15-5e77-4bf9-8ab3-057029fd14dd/landsat_cloud/dataset_seg_1_8/labels/"
#  标签的颜色字典
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)

#model = unet(model_path)
model = load_model(model_path)



testGene = testGenerator(test_iamge_path, input_size)

#  预测值的Numpy数组
results = model.predict_generator(testGene,
                                  test_num,
                                  verbose = 1)

#  保存结果
saveResult(test_iamge_path, save_path, results, colorDict_RGB, output_size)
import argparse
import utils

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
default_img = '/home/workspace/aipnd-project/flowers/test/99/image_07838.jpg'

parser = argparse.ArgumentParser(description='predict-file')
parser.add_argument('image_path', default=default_img, nargs='*', action="store")
parser.add_argument('checkpoint', default="./checkpoint.pth", nargs='*', action="store")
parser.add_argument('--top_k', default=3, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', dest="gpu", default="gpu", action="store",)
    
parse = parser.parse_args()
image_path = parse.image_path
checkpoint = parse.checkpoint
top_k = parse.top_k
category = parse.category_names
gpu = parse.gpu
trainloader, validloader, testloader, _ = utils.data_loaders(data_dir)
#model = utils.build_network(architecture, learn_rate)
model, optimizer = utils.load_checkpoint(path="checkpoint.pth")


cat_to_name = utils.category(category)

utils.predict(image_path, model, cat_to_name, device = gpu, topk = top_k)



import argparse
import utils

print("OK")

parser = argparse.ArgumentParser(description='Setting train image classifer')
parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/") 
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--arch', dest="arch", action="store", default="vgg16")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)  
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")

parse = parser.parse_args()
data_dir = parse.data_dir
save_dir = parse.save_dir
arch = parse.arch
learn_rate = parse.learning_rate
epochs = parse.epochs
hidden_units = parse.hidden_units
gpu = parse.gpu

trainloader, validloader, testloader, train_data = utils.data_loaders(data_dir)
model, input_size = utils.build_network(arch)
print(learn_rate)

model, optimizer = utils.trainer(trainloader, validloader, model, epochs = epochs, steps = 0,  learnrate = learn_rate, print_every = 5, gpu = gpu)

utils.save_checkpoint(model, 
                      train_data, 
                      optimizer, 
                      input_size, 
                      output_size = 102, 
                      hidden_layers =[102], 
                      learn_rate = learn_rate, 
                      epochs = epochs, 
                      steps = 0)


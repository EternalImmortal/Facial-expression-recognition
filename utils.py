import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')


def save_checkpoint(epoch, model, optimizer, best_acc):
    '''
        Save model checkpoint
    '''
    print('saving model with acc of ' + str(best_acc))
    state = {'epoch': epoch, "model_weights": model, "optimizer": optimizer}
    filename = "model_state.pth.tar"
    torch.save(state, filename)


def save_whole_model(model, best_acc):
    print('saving whole model')
    model_path = 'VGG19'
    print('finished saving model of acc ' + str(best_acc))
    torch.save(model, model_path)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def eval(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            # batch_size * crops_number * channel * height * weight
            bs, ncrops, c, h, w = np.shape(images)
            images = images.view(-1, c, h, w)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.view(bs, ncrops, -1).mean(1)
            # return values, indices
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %2f %%' % (100 * correct / total))


def detail_eval(model, test_loader):
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    with torch.no_grad():
        for images, labels in test_loader:
            bs, ncrops, c, h, w = np.shape(images)
            images = images.view(-1, c, h, w)
            images = images.to(device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            outputs = model(images)
            outputs = outputs.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(7):
        print('Accuracy of %5s : %2f (%d / %d) %%' % (
            classes[i], 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))


def save_data(model, test_loader):
    total_predicted = []
    total_labels = []
    total_predicted = np.array(total_predicted)
    total_labels = np.array(total_labels)
    with torch.no_grad():
        for images, labels in test_loader:
            bs, ncrops, c, h, w = np.shape(images)
            images = images.view(-1, c, h, w)
            images = images.to(device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            outputs = model(images)
            outputs = outputs.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(outputs, 1)
            total_predicted = np.append(total_predicted, predicted.cpu())
            total_labels = np.append(total_labels, labels.cpu())
    print(total_predicted.shape)
    print(total_labels.shape)
    np.save(total_predicted, 'total_predicted.npy')
    np.save(total_labels, 'total_labels.npy')

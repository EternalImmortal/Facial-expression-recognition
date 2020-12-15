import torch
from vgg import VGG
from datasets import FER2013, FER2013_MASK
from utils import eval, detail_eval, save_data
import argparse
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root",
                default="/Users/WeiJoseph/mystuff/Independent_Project_2020Fall/data/masked_fer2013.csv",
                help="Path to the data folder")
ap.add_argument("--bs", default=8, type=int, help="Batch size for evaluating")
ap.add_argument("--num_workers", default=4, type=int, help="Number of workers")
ap.add_argument("--trained_model", default="/home/renjie/code/model_state.pth.tar", type=str,
                help="Trained state_dict file path to open")
ap.add_argument("--model_name", default="VGG19", type=str, help="name model")
args = ap.parse_args()

data_path = args.dataset_root
batch_size = args.bs
model_path = args.trained_model
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
crop_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(crop_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
])

# publictest_dataset = FER2013(data_path, split="PUBLIC_TEST", transform=transform_test)
publictest_dataset = FER2013_MASK(data_path, split="PUBLIC_TEST", transform=transform_test)
publictest_dataloader = torch.utils.data.DataLoader(publictest_dataset, batch_size=batch_size,
                                                    num_workers=4)
# trained_model = torch.load(model_path)
# print("Load weight model with {} epoch".format(trained_model["epoch"]))

# model = VGG(args.model_name)
# model.load_state_dict(trained_model["model_weights"])
model = torch.load('VGG19')
model.to(device)
model.eval()

# publictest_dataset = FER2013(data_path, split="PUBLIC_TEST", transform=transform_test)
publictest_dataset = FER2013_MASK(data_path, split="PUBLIC_TEST", transform=transform_test)
publictest_dataloader = torch.utils.data.DataLoader(publictest_dataset, batch_size=batch_size,
                                                    num_workers=4)

# private_data = FER2013(data_path, split="PRIVATE_TEST", transform=transform_test)
private_data = FER2013_MASK(data_path, split="PRIVATE_TEST", transform=transform_test)
private_dataloader = torch.utils.data.DataLoader(private_data, batch_size=batch_size,
                                                 num_workers=4)

# print("Evaluation validation (public test) dataset...")
# eval(model, publictest_dataloader)
# detail_eval(model, publictest_dataloader)
# print("-" * 10)
#
# print("Evaluation public private dataset...")
# eval(model, private_dataloader)
# detail_eval(model, private_dataloader)
# print("-" * 10)

save_data(model, private_dataloader)

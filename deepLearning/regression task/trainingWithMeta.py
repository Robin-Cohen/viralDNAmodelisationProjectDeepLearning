from model import *
import time
import matplotlib.pyplot as plt
# import tqdm
from torch.cuda.amp import autocast, GradScaler
import csv
# from torch.utils.tensorboard import SummaryWriter

def save_losses_to_csv(train_losses, valid_losses, file_path='losses.csv')->None:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
            for epoch, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
                writer.writerow([epoch + 1, train_loss, valid_loss])



def plot_training_losses(train_losses, valid_losses, save_path='training3D.png'):
    """
    Plot training and validation losses and save to file.
    
    Args:
        train_losses (list): List of training losses
        valid_losses (list): List of validation losses
        save_path (str): Path to save the plot image
    """
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

#set randomness
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

deviceName= torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Current cuda device: {deviceName}")



#--------------------------------------Fine Tunnig
fineTunning = False
pretrained_model_path = "path/to/model.pth"

#--------------------------------------Data directory
noisyDataDirectoryPath = "path/to/directory"
noNoiseDataDirectoryPath = "/path/to/directory/"
metafile ="/path/to/csv/file"
#-------------------------------------- Data loading
#Noisy data and non noisy data are loaded separately
#After that  a part of the noisy are split in order to make test dataset
#The rest of the noisy Dataset and the noNoiseData #are concatenated 

torch.cuda.empty_cache() #maybe not necessary just in case
#loading class
dataset = MrcDataset2vMetaData(metaFile=metafile,noiseDirectory=noisyDataDirectoryPath,noNoiseDirectory=noNoiseDataDirectoryPath)
#spliting
trainDataset, testDataset = random_split(dataset , [0.8, 0.2])


#no augmentation/denoising



print(f"number of data points in training: {len(trainDataset)}")
print(f"number of data points in testing: {len(testDataset)}")
#data loader
batchSize= 200 # => must be max GPU node memory 
#worker are max CPU core of the GPU node
trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, pin_memory=True, num_workers=16)
testDataloader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, pin_memory=True, num_workers=16)



#--------------------------------------model initiaitsation & learning parameters

model= Model4().to(device) #last model for regrssion of 2 varaible
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.005,
    betas=(0.9, 0.9999),
    weight_decay=0.01,
    eps=1e-7,
    amsgrad=True
)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, 
    base_lr=0.0001,
    max_lr=0.01,
    step_size_up=2000,
    cycle_momentum=False
)

scaler = GradScaler()
# Load the pre-trained model
if fineTunning:
    try:
        model.load_state_dict(torch.load(pretrained_model_path))
        print(f"Loaded pre-trained model from {pretrained_model_path}")
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")

running_loss = 0.
last_loss = 0.

for inputs, labels in trainDataloader:
    print(f"Input shape: {inputs.shape}")
    print(f"Label shape: {labels.shape}")
    break

#learning variable
running_losses = []
losses = []
best_vloss = 1_000_000.
best_tloss = 1_000_000.
trainLosess=[]
validLosess=[]
patience = 200
waited=0

#--------------------------------------Training 
print("Start training")
timeTraining = time.time()
for epoch in range(1000):
    timeEpoch = time.time()
    print(f"Epoch {epoch}")
    model.train(True)
    print(f"number of batch: {len(trainDataloader)}")
    for i, data in enumerate(trainDataloader,0):
        inputs, labels = data
        inputs = inputs.float().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)#gradiant clipping
        optimizer.step()
        running_loss += loss.item()
        if i == len(trainDataloader) - 1:
            print("------------------------------------------------")
            last_loss = running_loss/len(trainDataloader)
            trainLosess.append(last_loss)
            print(f"Epoch {epoch}, loss: {last_loss:.4f}")
            running_loss = 0.0
    print(f"LR actuel: {optimizer.param_groups[0]['lr']}")
    print("evaluating   ")
    model.eval()
    vrunning_loss = 0.
    predictions = []
    actuals = []
    with torch.no_grad(): # disable gradient calculation for testing
        for i, vdata in enumerate(testDataloader, 0):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            predictions.extend(voutputs.cpu().numpy())  
            actuals.extend(vlabels.cpu().numpy())
            vloss = loss_fn(voutputs, vlabels)
            vrunning_loss += vloss.item()
    avg_vloss = vrunning_loss / len(testDataloader)
    validLosess.append(avg_vloss)
    scheduler.step()
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_tloss = last_loss 
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        waited+=1
        if waited > patience:

            print(f"LOSS train {last_loss} valid {avg_vloss}")
            print(f"Time for epoch {epoch}: {time.time()-timeEpoch}")
            print("Early stopping")
            break
    
    print(f"LOSS train {last_loss} valid {avg_vloss}")
    print(f"Time for epoch {epoch}: {time.time()-timeEpoch}")

    plot_training_losses(trainLosess, validLosess)

    print(f"Time for training over {epoch} epoch: {time.time()-timeTraining}")
save_losses_to_csv(trainLosess, validLosess)
print("====================================================================")
print("end of the training")
print(f"LOSS train {best_tloss} valid {best_vloss}")
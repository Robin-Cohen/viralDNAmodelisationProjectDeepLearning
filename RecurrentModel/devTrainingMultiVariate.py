from devModel import *
import time
import matplotlib.pyplot as plt
import tqdm
import logging
import csv
#previous best model: best_modelEpoch37.pth
# from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(42)
torch.set_num_threads(20)

fineTunning = False
dataDirectoryPath="/home/robin/viralDNAmodelisationProjectDeepLearning/spiralMidlePitchNoNoises"
# dataDirectoryPath="/home/robin/viralDNAmodelisationProjectDeepLearning/datasetCylinder"
# Configure logging
log_file_path = 'training_log.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Replace all print statements with logging.info
# print = logging.info
dataset = MrcDataset(getMrcFeatures(dataDirectoryPath))
# dataset = MrcDatasetMulti(getMrcFeatures("/home/robin/viralDNAmodelisationProjectDeepLearning/spiralMidlePitchNoNoises"))
print(f"Number of data points: {len(dataset)}")

trainDataset, testDataset = random_split(dataset , [0.9, 0.1])

trainDataloader = DataLoader(trainDataset, batch_size=256, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=256, shuffle=False)

# print(trainDataloader)
 
model= ModelMultiReg()
loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    betas=(0.9, 0.9999),
    weight_decay=0.01,
    eps=1e-7,
    amsgrad=True
)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, 
    base_lr=0.0001,
    max_lr=0.01,
    step_size_up=2000
)

# Load the pre-trained model
if fineTunning:
    pretrained_model_path = '/home/robin/viralDNAmodelisationProjectDeepLearning/RecurrentModel/twoVariableTrainingNoNoise/best_modelEpoch37.pth'

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


running_losses = []
losses = []
best_vloss = 1_000_000.
trainLosess=[]
validLosess=[]
patience = 50
waited=0
print("Start training")

timeTraining = time.time()
for epoch in range(100):
    timeEpoch = time.time()
    print(f"Epoch {epoch}")
    model.train(True)
    print(f"number of batch: {len(trainDataloader)}")
    for i, data in enumerate(trainDataloader,0):
        inputs, labels = data
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)#gradiant clipping
        optimizer.step()
        running_loss += loss.item()
        #print(f"Batch: Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
        if i == len(trainDataloader) - 1:
            print("------------------------------------------------")
            last_loss = running_loss/len(trainDataloader)
            trainLosess.append(last_loss)
            print(f"Epoch {epoch}, loss: {last_loss:.4f}")
            print("------------------------------------------------")
            running_loss = 0.0
    print(f"LR actuel: {optimizer.param_groups[0]['lr']}")
    print("evaluating model")
    model.eval()
    vrunning_loss = 0.
    predictions = []
    actuals = []
    with torch.no_grad():
        for i, vdata in enumerate(testDataloader, 0):
            vinputs, vlabels = vdata
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
        torch.save(model.state_dict(), 'best_model.pth')
        for i in range(5):
            print(f"prediction: {predictions[i]}, actual: {actuals[i]}")
        waited=0
    else:
        waited+=1
        if waited > patience:
            print(f"LOSS train {last_loss} valid {avg_vloss}")
            print(f"Time for epoch {epoch}: {time.time()-timeEpoch}")
            print("Early stopping")
            break
    
    print(f"LOSS train {last_loss} valid {avg_vloss}")
    print(f"Time for epoch {epoch}: {time.time()-timeEpoch}")

    plt.plot(trainLosess, label='Training Loss')
    plt.plot(validLosess, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training3D-RT.png')
    plt.close()

    

        
    print(f"Time for training over {epoch} epoch: {time.time()-timeTraining}")

    # Save train losses and validation losses to a CSV file
    csv_file_path = 'losses.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        for epoch, (train_loss, valid_loss) in enumerate(zip(trainLosess, validLosess)):
            writer.writerow([epoch + 1, train_loss, valid_loss])

    print(f"Losses saved to {csv_file_path}")
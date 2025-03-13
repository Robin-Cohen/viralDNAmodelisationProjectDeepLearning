from devModel import *

torch.manual_seed(42)
dataset = MrcDataset(getMrcFeatures("/home/robin/viralDNAmodelisationProjectDeepLearning/dataset"))

trainDataset, testDataset = random_split(dataset, [0.8, 0.2])

trainDataloader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=64, shuffle=False)

input_size = 35*35*35
hidden_sizes = [512, 256, 128, 64]  
output_size = 1  


model = FullyConnectedRegressor()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
num_epochs = 100
print("training on fully connected model")
print("training started: on ", num_epochs, " epochs")
print("iteration per epoch: ", len(trainDataloader))
best_vloss = 1_000_000.
for epoch in range(num_epochs):
   
    model.train()
    for i, data in enumerate(trainDataloader):
        loss = 0
        images, labels= data
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss += loss.item()
        print(f'Epoch [{epoch}], Iteration {i} Loss: {loss.item():.4f}')
        if i == len(trainDataloader) - 1:
            print("------------------------------------------------")
            print(f'Epoch [{epoch}], Training Loss: {loss/len(trainDataloader):.4f}')
            print("------------------------------------------------")
    # evaluation in each epoch
    model.eval()

    predictions = []
    actuals = []
    with torch.no_grad():
        val_loss = 0
        counter = 0
        for images, labels in testDataloader:
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            if counter < 10:
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(testDataloader)
        #print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

 
    counter = 0
    # for pred, actual in zip(predictions, actuals): #take time : remove it later
    #     #print(f"Prediction: {pred}, Actual: {actual}")
    #     if counter < 10:
    #         break
    #     counter += 1

    if avg_val_loss < best_vloss:
        best_vloss = avg_val_loss
        model_path = 'model_{}.pth'.format(epoch)
        torch.save(model.state_dict(), model_path)
        with open('training_log.txt', 'a') as f:
            f.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss/len(trainDataloader):.4f}, Validation Loss: {avg_val_loss:.4f}\n')
            for pred, actual in zip(predictions, actuals):
                f.write(f'Prediction: {pred}, Actual: {actual}\n')
                print(f"Prediction: {pred}, Actual: {actual}")
                f.write("-----------------------------\n")
                
    
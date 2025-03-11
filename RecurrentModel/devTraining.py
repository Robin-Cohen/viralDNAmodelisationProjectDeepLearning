from devModel import *
# from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(42)
dataset = MrcDataset(getMrcFeatures())
# writer = SummaryWriter()
# print("------------------------------------------------")
# print(dataset.shape()))
# print("------------------------------------------------")
# print(dataset[0][0].size())
# devTestLoader = DataLoader(dataset, batch_size=1, shuffle=True)


trainDataset, testDataset = random_split(dataset, [0.8, 0.2])

trainDataloader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=64, shuffle=False)

# print(trainDataloader)
print("------------------------------------------------")
print(len(trainDataloader))
print("------------------------------------------------")

model= Model()
loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

running_loss = 0.
last_loss = 0.

for inputs, labels in trainDataloader:
    print(f"Input shape: {inputs.shape}")
    print(f"Label shape: {labels.shape}")
    break
print("------------------------------------------------")
print("training started")
print(len(trainDataloader))

running_losses = []
losses = []
best_vloss = 1_000_000.
for epoch in range(30):
    
    print(f"Epoch {epoch}")
    model.train(True)
    for i, data in enumerate(trainDataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_losses.append(running_loss)
        losses.append(loss.item()/len(trainDataloader))
        print(f"Batch: Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
        if i == len(trainDataloader) - 1:
            print("------------------------------------------------")
            last_loss = running_loss/len(trainDataloader)
            print(f"Epoch {epoch}, loss: {last_loss:.4f}")
            losses.append(running_loss/len(trainDataloader))
            running_loss = 0.0
            running_losses.append(running_loss)
            
    #test the model
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
    avg_vloss = vrunning_loss / (i + 1)
    print('LOSS train {} valid {}'.format(last_loss, avg_vloss))

    for pred, actual in zip(predictions, actuals):
        print(f"Prediction: {pred}, Actual: {actual}")
    # writer.add_scalars('Training vs. Validation Loss',
    #                 { 'Training' : last_loss, 'Validation' : avg_vloss },
    #                 epoch)
    # writer.flush()
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}.pth'.format(epoch)
        torch.save(model.state_dict(), model_path)
print("Finished Training")
plt.plot(losses)
plt.savefig("loss.png")
plt.plot(running_losses)
plt.savefig("running_loss.png")
# Save the model
torch.save(model.state_dict(), "./model.pth")
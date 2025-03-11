from devModel import *

pthName= "/home/robin/viralDNAmodelisationProjectDeepLearning/model.pth"
mrcRep="/home/robin/viralDNAmodelisationProjectDeepLearning/RecurrentModel/testmrc"

model=Model()
model.load_state_dict(torch.load(pthName))
model.eval()
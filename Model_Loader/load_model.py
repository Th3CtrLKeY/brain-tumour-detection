from Log_Writer.logger import App_Logger
import torchvision , torch
import torch.nn as nn
import dill as dill

def load(PATH):
    try:
        logger = App_Logger()

        # # Load Model
        device = torch.device('cpu')
        # model = torchvision.models.squeezenet1_1(pretrained=False).to(device)
        # model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1)).to(device)
        # # if we load model from its original checkpoints and then load best model weights
        # model.load_state_dict(torch.load(PATH, map_location=device))
        # model.eval()

        #if we use saved model directly
        model=torch.load(PATH, pickle_module=dill, encoding='utf-8' ,map_location=torch.device('cpu'))
        model.eval()
        logger.log("Model Loaded Successfully")
        return model
    except Exception as e:
        logger = App_Logger()
        logger.log("ERROR : Model Loading Unsuccessful\n")
        return print(e)
from torch import nn
import torch
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        #super(NeuralNetwork, self).__init__()
        super().__init__()

        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(1000, 1200),
            nn.Sigmoid(),
            nn.Linear(1200,1200),
            nn.Sigmoid(),
            nn.Linear(1200, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #x = self.flatten(x)
        return self.linear_sigmoid_stack(x)


class CustomMailDataset(Dataset):
    def __init__(self,mail_content,spam_labels):
        self.spam_labels = spam_labels.to(torch.float32)
        self.mail_content = mail_content.to(torch.float32)

    def __len__(self):
        return len(self.spam_labels)

    def __getitem__(self, idx):
        mail_content = self.mail_content[idx]
        spam = self.spam_labels[idx]
        return mail_content, spam

def load_set() -> tuple[torch.Tensor,torch.Tensor]:
    mail_data = torch.load('mail_data.pt')
    dataset = CustomMailDataset(mail_data['X'],mail_data['y'])

    train_set,test_set = torch.utils.data.random_split(dataset, [0.8,0.2], generator=torch.Generator().manual_seed(42))

    return train_set, test_set

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(torch.squeeze(pred, 0), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()

        if batch % 2500 == 0:
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return loss

def test(testloader, trainloader, model, loss_fn):
    num_batches = len(testloader)
    model.eval()
    test_loss, train_correct, test_correct = 0, 0, 0

    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(torch.squeeze(test_pred, 0), y)

            if test_pred >= .5 and y == 1 or test_pred < .5 and y == 0:
                # correct predicted via sigmoid
                test_correct += 1

        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            train_pred = model(X)        
            if train_pred >= .5 and y == 1 or train_pred < .5 and y == 0:
                # correct predicted via sigmoid   
                train_correct += 1

    
    test_loss /= num_batches
    test_correct /= len(testloader.dataset)
    train_correct /= len(trainloader.dataset)
    print(f"Test Error: \n Accuracy: {(100*test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss.item(), test_correct, train_correct


def train_the_model(epochs):
    print('train the model')

    test_dataloader = DataLoader(test_set)
    train_dataloader = DataLoader(train_set)

    model = NeuralNetwork().to(device)
    #print(model)

    loss_fn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-6)
    test_losses,train_losses, test_acc, train_acc = [],[],[],[]

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses.append(train(train_dataloader, model, loss_fn, optimizer))
        loss,test_correct,train_correct = test(test_dataloader, train_dataloader, model, loss_fn)
        test_losses.append(loss)
        test_acc.append(test_correct)
        train_acc.append(train_correct)

    # 50 Epochs SGD learning rate 1e-3: Accuracy: 86.5%, Avg loss: 0.369147
    # 25 Epoch Adam lr 1e-4 wd 1e-7: Accuracy: 90%

    save_the_model(model,f'Adam_spam_filter')

    return train_losses, train_acc, test_losses, test_acc 

    


def save_the_model(model,name:str):
    """save the model 'model' to name.pth \n
    To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method"""

    # saves the weights
    torch.save(model.state_dict(), f"{name}.pth")
    print(f"Saved PyTorch Model State to {name}.pth")

def print_the_learning_curve(train_losses,train_acc,test_losses,test_acc,x):    
    fig,ax = plt.subplots()
    
    ax.plot(x,train_losses,label='training')
    ax.plot(x,test_losses,label='test')
    ax.plot(x,train_acc,label='training acc.')
    ax.plot(x,test_acc,label='test acc.')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_ylim((0,1))
    ax.set_title('Learning curve for 25 epochs with Adam')
    plt.xticks(np.arange(0,len(x)+1,5))
    plt.legend(loc='upper right')
    plt.savefig('Adam_tinker_spam_filter.png')


if __name__ == "__main__":

    # more than 30 is useless with adam
    epochs = 25

    train_set,test_set= load_set()
   
    train_losses,train_acc, test_losses, test_acc = train_the_model(epochs)

    print("Training Done!")    

    x = [x for x in range(epochs)]

    print_the_learning_curve(train_losses,train_acc,test_losses,test_acc,x)
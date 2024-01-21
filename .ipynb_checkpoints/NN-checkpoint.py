import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
from tqdm.notebook import tqdm
import optuna
from optuna.trial import TrialState
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle
from optuna import trial
warnings.filterwarnings('ignore')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
pd.set_option('mode.chained_assignment',None)


def get_dataframe(path):
    df=pd.read_csv(path)
    return df

def split_data(X,y):
    return train_test_split(X,y.to_numpy(),test_size=0.1,random_state=42)

def split_trainval(X,y):
    return train_test_split(X,y,test_size=0.5,random_state=42)

def normalize_dataset(X_train,X_val,X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train,X_val,X_test

def convert_to_torch(value):
    return torch.tensor(data=value,dtype=torch.float32,requires_grad=True)

data = get_dataframe("/lustre/work/guoqi/code/result/DM_NN_z1.csv")                    
X = data[['Omega_m','sigma_8','A_SN1','A_AGN1','A_SN2','A_AGN2','redshift','halo_mass']]            #feature
y = np.log10(-data['C0'])
X_train,X_val_,y_train,y_val_  = split_data(X,y)
X_val,X_test,y_val,y_test  = split_trainval(X_val_,y_val_)

X_train,X_val,X_test=normalize_dataset(X_train,X_val,X_test)
X_train=np.array(X_train)
X_val=np.array(X_val)
y_train=np.array(y_train)
y_val=np.array(y_val)
print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class FluxData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        esp = 1e-6
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x,y)) + esp
        loss = torch.sqrt(criterion(x,y))
        return loss

X_data = convert_to_torch(X_train)
y_data = convert_to_torch(y_train)
X_val = convert_to_torch(X_val)
y_val = convert_to_torch(y_val)
train_data = FluxData(X_data,y_data)
test_data = FluxData(X_val,y_val)

def RegressionFluxNet(trial,in_features,n_layers,dropout,n_output):
    
    layers = []
    fc_layer = in_features
    
    for i in range(n_layers):
        
        out_features = trial.suggest_int("n_units_l{}".format(i),30, 35)
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features

    layers.append(torch.nn.Linear(in_features, fc_layer)) 
    layers.append(torch.nn.ReLU())

    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(fc_layer,n_output)) 
    
    return torch.nn.Sequential(*layers)
def get_dataloaders(dataset_type,batch,shuffle):
    if shuffle:
         return DataLoader(dataset=dataset_type, batch_size=batch, shuffle=True)
    else:
        return DataLoader(dataset=dataset_type, batch_size=batch,shuffle=False)


def train_net(trial, params,model):
    trainloader = get_dataloaders(FluxData(X_data,y_data), batch=20, shuffle=True)
    testloader = get_dataloaders(FluxData(X_val,y_val), batch=1, shuffle=False)
    
    loss_function = RMSELoss()
    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
  
    for epoch in range(0, 10): 
        
        total_test_loss = []
  
        
        for i, data in enumerate(trainloader, 0):
        
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))
                
                test_outputs = model(inputs)
                test_loss = loss_function(test_outputs, targets)
                total_test_loss.append(test_loss.item())

    return total_test_loss

def objective(trial):

    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate',1e-6,1e-3), 
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              'weight_decay': trial.suggest_loguniform('weight_decay',1e-2,1),
              "n_layers" : trial.suggest_int("n_layers", 5,10),
              "dropout" : trial.suggest_float('dropout',0.1,0.2,step = 0.1)
              }
    
    model = RegressionFluxNet(trial=trial,
            in_features= X_train.shape[1],
            n_layers=params['n_layers'] ,
            dropout=params['dropout'],
            n_output= 1).to(device)
    test_loss = train_net(trial,params,model) 
    return np.mean(test_loss)


study = optuna.create_study(study_name='Pytorch-PS3E15',direction ="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials= 100)

    
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))
best_trial = study.best_trial

print("Best trial:", best_trial)
print("  Value: ", best_trial.value)

print("Best Trail Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

model = RegressionFluxNet(trial=best_trial,
        in_features= X_train.shape[1],
        n_layers=best_trial.params['n_layers'] ,
        dropout=best_trial.params['dropout'],
        n_output= 1).to(device)
params = {
          'learning_rate': best_trial.suggest_loguniform('learning_rate',1e-6,1e-3), 
          'optimizer': best_trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
          'weight_decay': best_trial.suggest_loguniform('weight_decay',1e-6,1e-1),
          "n_layers" : best_trial.suggest_int("n_layers", 7,12),
          "dropout" : best_trial.suggest_float('dropout',0.1,0.2,step = 0.1)
          }

trainloader = get_dataloaders(FluxData(X_data,y_data), batch=40, shuffle=True)
testloader = get_dataloaders(FluxData(X_val,y_val), batch=1, shuffle=False)
loss_function = RMSELoss()
optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
loss_list=[]

for epoch in range(0, 8000):

    total_test_loss = []

    for i, data in enumerate(trainloader, 0):

        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            test_outputs = model(inputs)
            test_loss = loss_function(test_outputs, targets)
            total_test_loss.append(test_loss.item())
    loss_list.append(np.mean(total_test_loss))

X_test = convert_to_torch(X_test)
y_test = convert_to_torch(y_test)
plt.figure(figsize=(5, 5))
yp_train = model(X_data)
yp_val_pre  = model(X_val)
yp_test_pre = model(X_test)
plt.scatter(yp_train.detach().numpy(),y_data.detach().numpy(), s=3, label="Train Set")
plt.scatter(yp_val_pre.detach().numpy(), y_val.detach().numpy(), alpha=0.5, label="Validation Set")
plt.scatter(yp_test_pre.detach().numpy(), y_test.detach().numpy(), alpha=0.5, label="Test Set")
plt.title("log(C0)")
plt.xlabel("obs")
plt.ylabel("pre")
plt.legend()
# plt.savefig("./C0_diffz.pdf")
plt.show()
plt.close()
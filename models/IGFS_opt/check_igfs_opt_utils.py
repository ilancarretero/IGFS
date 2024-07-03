# Integrated Gradients Feature Selection auxiliary classes and functions

# Import modules 
import time
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, roc_auc_score
from captum.attr import IntegratedGradients
from sklearn.preprocessing import StandardScaler
import mlflow

# Reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Scaler
def data_scaler(X_train):
    # Normalize train  and test with MinMaxScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns.tolist())
    # Return partitions and normalized data
    return X_train_scaled
    
# General settings for pytorch
def transfer_device(X_train, y_train, X_val, y_val, criterion_rec, criterion_clf, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device) 
    criterion_rec = criterion_rec.to(device)
    criterion_clf = criterion_clf.to(device)
    model = model.to(device)
    return X_train, y_train, X_val, y_val, criterion_rec, criterion_clf, model

def torch_tensors(X_t_cv, y_t_cv, X_v_cv, y_v_cv):
    X_t_cv = torch.FloatTensor(X_t_cv.values)
    y_t_cv = torch.FloatTensor(y_t_cv.values)
    X_v_cv = torch.FloatTensor(X_v_cv.values)
    y_v_cv = torch.FloatTensor(y_v_cv.values)
    return X_t_cv, y_t_cv, X_v_cv, y_v_cv

def get_params(args):
    # Get parameters and hyperparameters for each of the databases models
    # Extracted from Optuna results
    if args.db == 'colon':
        params = {
            'learning_rate': 0.05,
            'optimizer': 'adamW',
            'ae_arch': 'small',
            'latent_space': 24,
            'lambda_1': 0.0001,
            'lambda_2': 0.0001,
            'activation_function': 'relu',
            'batch_normalization': 'true',
            'classifier_arch': 'hidden',
            'batch_size': 32,
            'weight_recon': 97,
            'weight_cl': 2,
            'cl_weights': 'true'}
    elif args.db == 'leukemia':
        params = {
            'learning_rate': 0.05,
            'optimizer': 'adam',
            'ae_arch': 'small',
            'latent_space': 32,
            'lambda_1': 0.00001,
            'lambda_2': 0.0001,
            'activation_function': 'prelu',
            'batch_normalization': 'true',
            'classifier_arch': 'no_hidden',
            'batch_size': 16,
            'weight_recon': 27,
            'weight_cl': 93,
            'cl_weights': 'true'}
    elif args.db == 'crohn':
        params = {
            'learning_rate': 0.0001,
            'optimizer': 'adamW',
            'ae_arch': 'small',
            'latent_space': 32,
            'lambda_1': 0,
            'lambda_2': 0.00001,
            'activation_function': 'relu',
            'batch_normalization': 'true',
            'classifier_arch': 'hidden',
            'batch_size': 32,
            'weight_recon': 52,
            'weight_cl': 1,
            'cl_weights': 'true'
        }
    return params

# NN Architecture and parameters/hiperparameters
def param_hparam_nn(params, args, X, y):
    # Loss functions
    criterion_rec = nn.MSELoss()
    if params['cl_weights'] == 'true':
        pos_weight_number = np.sum(y[y.columns[0]] == 0) / np.sum(y[y.columns[0]] == 1)
        criterion_clf = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight_number]))
    else:
        criterion_clf = nn.BCEWithLogitsLoss()
    # Layers and neurons inside each layer
    if args.db == 'colon':
        if params['ae_arch'] == 'small':
            layers = [X.shape[1]] + [2**7] 
        else:
            layers = [X.shape[1]] + [2**9, 2**7] 
    elif args.db == 'leukemia':
        if params['ae_arch'] == 'small':
            layers = [X.shape[1]] + [2**9, 2**6] 
        else:
            layers = [X.shape[1]] + [2**11, 2**9, 2**7] 
    elif args.db == 'crohn':
        if params['ae_arch'] == 'small':
            layers = [X.shape[1]] + [2**10, 2**6]  
        else:
            layers = [X.shape[1]] + [2**12, 2**10, 2**8, 2**6] 
    # Layers + latent space
    layers = layers + [params['latent_space']]
    # Activation function
    if params['activation_function'] == 'relu':
        act_funct = nn.ReLU()
    elif params['activation_function'] == 'silu':
        act_funct = nn.SiLU()
    elif params['activation_function'] == 'prelu':
        act_funct = nn.PReLU()
    # Return
    return criterion_rec, criterion_clf, layers, act_funct
    
class autoencoder_E2E(torch.nn.Module):
    def __init__(self, layers, afunc, params):
        super(autoencoder_E2E, self).__init__()
        
        # Define variables
        batch_norm = True if params['batch_normalization'] == 'true' else False
        hidden = True if params['classifier_arch'] == 'hidden' else False
        
        self.encoder_loop = nn.Sequential()
        self.decoder_loop = nn.Sequential()
        self.classifier = nn.Sequential()
        
        # Encoder layers
        for i in range(len(layers)-1):
            in_features = layers[i]
            out_features = layers[i+1]
            layer = nn.Linear(in_features, out_features)
            self.encoder_loop.add_module(f"encoder_layer_{i}", layer)
            if batch_norm:
                self.encoder_loop.add_module(f"batch_norm_layer_{i}", nn.BatchNorm1d(out_features))
            self.encoder_loop.add_module(f"encoder_activation_{i}", afunc)
            # if i  == (len(layers) - 2): # PENSAR LA CONDICIÓN QUE DEBERÍA PONER
                #self.encoder_loop.add_module(f"encoder_normalization_{i}", nn.BatchNorm1d(neurons[-1]))
            #    pass
              
        # Decoder layers
        for i in range(len(layers)-1, 0, -1):
            in_features = layers[i]
            out_features = layers[i-1]
            layer = nn.Linear(in_features, out_features)
            self.decoder_loop.add_module(f"decoder_layer_{i}", layer)
            if batch_norm:
                self.decoder_loop.add_module(f"batch_norm_layer_{i}", nn.BatchNorm1d(out_features))
            if i != 1: # 1 porque en el range vamos al revés, por lo que es en esa capa donde nos interesa hacer la sigmoide
                self.decoder_loop.add_module(f"decoder_activation_{i}", afunc)
            else: # PENSAR LA CONDICIÓN QUE DEBERÍA PONER
                self.decoder_loop.add_module(f"decoder_sigmoid_{i}", nn.Sigmoid()) 
                pass 
                
        # Classifier layers
        if hidden:
            layer = nn.Linear(layers[-1], int(layers[-1]/2))
            self.classifier.add_module(f'hidden_layer', layer)
            self.classifier.add_module(f'hidden_activation', afunc)
            layer = nn.Linear(int(layers[-1]/2), 1)
            self.classifier.add_module(f'classification_layer', layer)
            #self.classifier.add_module(f'cl_layer_activation', nn.Sigmoid())
        else:
            layer = nn.Linear(layers[-1], 1)
            self.classifier.add_module(f'classification_layer', layer)
            #self.classifier.add_module(f'cl_layer_activation', nn.Sigmoid())
     
    def forward(self, x):
        encoded_loop = self.encoder_loop(x)
        decoded_loop = self.decoder_loop(encoded_loop)
        out_loop = self.classifier(encoded_loop)
        return decoded_loop, out_loop, encoded_loop 

# Train and evaluation functions
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=0, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        # Assert and self definitions
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        # Full batch size
        if batch_size == 0:
            batch_size = self.tensors[0].shape[0]
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def calculate_accuracy(out, labels):
            round_pred = torch.round(out)
            similarity = round_pred == labels
            correct = similarity.sum()
            accuracy = correct/labels.shape[0] * 100
            return accuracy, round_pred
        
def train(train_tensor, y_train_tensor, model, optimizer, criterion1, criterion2, params, weight_rec=70, weight_bce=1):
            inputs = train_tensor
            labels = torch.reshape(y_train_tensor, (y_train_tensor.shape[0], 1))
            optimizer.zero_grad()
            decoded, out, _ = model(inputs)
            loss1 = criterion1(decoded, inputs)
            loss2 = criterion2(out, labels)
            weight1 = weight_rec#70#30 #30 #70 peso optimo para metilacion #1
            weight2 = weight_bce#1 #0.002
            # L1 and L2 regularization term
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            l1_reg = torch.tensor(0.).to(device=device)
            l2_reg = torch.tensor(0.).to(device=device)
            for param in model.parameters():
                    l1_reg += torch.norm(param, p=1)
                    l2_reg += torch.norm(param, p=2)
            initial_loss = weight1 * loss1 + weight2 * loss2 
            loss = initial_loss + float(params['lambda_1']) * l1_reg + float(params['lambda_2'])* l2_reg
            acc, _ = calculate_accuracy(torch.sigmoid(out), labels)
            loss.backward()
            optimizer.step()
            return loss.item(), acc.item(), labels, out
        
def evaluate(val_tensor, y_val_tensor, model, criterion1, criterion2):
    with torch.no_grad():
        inputs = val_tensor
        labels = torch.reshape(y_val_tensor, (y_val_tensor.shape[0], 1))
        decoded, out, _ = model(inputs)
        loss1 = criterion1(decoded, inputs)
        loss2 = criterion2(out, labels)
        weigth1 = 1
        weigth2 = 1
        loss = weigth1 * loss1 + weigth2 * loss2
        acc, _ = calculate_accuracy(torch.sigmoid(out), labels)
    return loss.item(), acc.item(), labels, out
              
# Train and evaluation loop
def train_eval_loop(epochs, train_tensor, y_train_tensor, val_tensor, y_val_tensor, model, optimizer, criterion_rec, criterion_clf, params, i, classes):
    best_v_loss = 1000000
    epochs_no_improve = 0
    max_epochs_stop = 500
    for epoch in range(epochs):
        # Define class Dataloader class
        train_batches = FastTensorDataLoader(train_tensor, y_train_tensor,
                                         batch_size=params['batch_size'],
                                         shuffle=True)
        # Train
        model.train(True)
        running_loss = 0.0
        running_acc = 0.0
        counter = 0
        y_pred_total_t = []
        y_true_total_t = []
        n_metrics = 7
        results_total_t = np.zeros(n_metrics)
        for x_train_batch, y_train_batch in train_batches:
            train_loss, train_acc, labels, out = train(x_train_batch, y_train_batch, model, optimizer, criterion_rec, criterion_clf, params, weight_rec=params['weight_recon'], weight_bce=params['weight_cl'])
            running_loss += train_loss
            running_acc += train_acc
            counter += 1
            if np.all(labels.cpu().numpy().astype(int)== labels.cpu().numpy().astype(int)[0]):
                continue
            y_pred_t, y_true_t, results_t = clf_results(labels, torch.sigmoid(out), classes)
            y_pred_total_t += y_pred_t.squeeze().tolist()
            y_true_total_t += y_true_t.squeeze().tolist()
            results_total_t += results_t
        loss_per_epoch = running_loss / counter
        acc_per_epoch = running_acc / counter
        results_t_per_epoch = results_total_t / counter
        mlflow.log_metric('train loss', loss_per_epoch, step=epoch)
        mlflow.log_metric('train accuracy', acc_per_epoch, step=epoch)
        mlflow.log_metric('train ACC', results_t_per_epoch[0], step=epoch)
        mlflow.log_metric('train SEN', results_t_per_epoch[1], step=epoch)
        mlflow.log_metric('train SPE', results_t_per_epoch[2], step=epoch)
        mlflow.log_metric('train PPV', results_t_per_epoch[3], step=epoch)
        mlflow.log_metric('train NPV', results_t_per_epoch[4], step=epoch)
        mlflow.log_metric('train F1', results_t_per_epoch[5], step=epoch)
        mlflow.log_metric('train AUC', results_t_per_epoch[6], step=epoch)
        # Evaluation
        model.eval()
        valid_loss, valid_acc, labels, out = evaluate(val_tensor, y_val_tensor, model, criterion_rec, criterion_clf)
        y_pred_v, y_true_v, results_v = clf_results(labels, torch.sigmoid(out), classes)
        mlflow.log_metric('val loss', valid_loss, step=epoch)
        mlflow.log_metric('val accuracy', valid_acc, step=epoch)
        mlflow.log_metric('val ACC', results_v[0], step=epoch)
        mlflow.log_metric('val SEN', results_v[1], step=epoch)
        mlflow.log_metric('val SPE', results_v[2], step=epoch)
        mlflow.log_metric('val PPV', results_v[3], step=epoch)
        mlflow.log_metric('val NPV', results_v[4], step=epoch)
        mlflow.log_metric('val F1', results_v[5], step=epoch)
        mlflow.log_metric('val AUC', results_v[6], step=epoch)
        # Save best validation loss
        if valid_loss < best_v_loss:
            best_v_loss = valid_loss
            best_t_loss = loss_per_epoch
            epochs_no_improve = 0
            best_results_t = results_t_per_epoch
            best_results_v = results_v
            best_y_pred_total_t = y_pred_total_t
            best_y_pred_v = y_pred_v
            best_epochs = epoch
        else:
            epochs_no_improve += 1
        # Early stopping
        if epochs_no_improve == max_epochs_stop:
            print(f'Early stopping after {max_epochs_stop} with no improvement in epoch {epoch+1}')
            break
        # Prints results per epoch
        if epoch % 100 == 0:
            print(f'-------------- Epoch: {epoch+1:02} ----------------')
            print(f'\tTrain Loss: {loss_per_epoch:.3f} | Train Acc: {acc_per_epoch:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc:.2f}%')  
    print(f'---------------------------- CV: {i} -------------------------------------')    
    return best_v_loss, best_t_loss, best_results_t, best_results_v, y_true_total_t, best_y_pred_total_t, y_true_v, best_y_pred_v, best_epochs

# Obtain metrics
def report_classification_results(y_true, y_pred, classes):
    """ Print confusion matrix and show metrics """
    # Extract confusion matrix
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)

    # Compute metrics
    headers = []
    g_TP, g_TN, g_FN, g_FP = [],[],[],[]
    ss, ee, pp, nn, ff, aa = [],[],[],[],[],[]
    S, E, PPV, NPV, F, ACC = ['Sensitivity'], ['Specificity'], ['PPV (precision)'], ['NPV'], ['F1-score'], ['Accuracy'] #Positive Predictive Value/Negative Predictive Value
    i = 1
    headers.append(classes[i])

    # Extract indicators per class
    TP = conf_mat[i,i]
    #TN = sum(sum(conf_mat))-sum(conf_mat[i,:])-sum(conf_mat[:,i])
    FN = sum(conf_mat[i,:])-conf_mat[i,i]
    FP = sum(conf_mat[:,i])-conf_mat[i,i]
    TN = sum(sum(conf_mat))-TP-FP-FN

    # Extract metrics per class
    sen = TP/(TP+FN)
    spe = TN/(TN+FP)
    ppv = TP/(TP+FP)
    npv = TN/(TN+FN)
    f1_s = 2*(ppv*sen/(ppv+sen))
    acc = (TP+TN)/(TP+TN+FP+FN)

    # Create table for printing
    S.append(str(round(sen,4)))
    E.append(str(round(spe,4)))
    PPV.append(str(round(ppv,4)))
    NPV.append(str(round(npv,4)))
    F.append(str(round(f1_s,4)))
    ACC.append(str(round(acc,4)))
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    metrics_array = np.array([acc, sen, spe, ppv, npv, f1_s, auc])
    return metrics_array

def clf_results(labels, out, classes):
    labels_cpu = torch.Tensor.cpu(labels)
    predictions = torch.round(out)
    predictions_cpu = torch.Tensor.cpu(predictions)
    y_pred = predictions_cpu.detach().numpy().astype(int)
    y_true = labels_cpu.numpy().astype(int)
    results = report_classification_results(y_true, y_pred, classes)
    return y_pred, y_true, results
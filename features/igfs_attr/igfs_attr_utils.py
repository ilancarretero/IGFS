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

# NN Architectures
class autoencoder_E2E(torch.nn.Module):
    def __init__(self, neurons):
        super(autoencoder_E2E, self).__init__()
        
        self.encoder_loop = nn.Sequential()
        self.decoder_loop = nn.Sequential()
        
        # Encoder layers
        for i in range(len(neurons)-1):
            in_features = neurons[i]
            out_features = neurons[i+1]
            layer = nn.Linear(in_features, out_features)
            self.encoder_loop.add_module(f"encoder_layer_{i}", layer)
            self.encoder_loop.add_module(f"encoder_activation_{i}", nn.ReLU())
            if i  == (len(neurons) - 2): # PENSAR LA CONDICIÓN QUE DEBERÍA PONER
                self.encoder_loop.add_module(f"encoder_normalization_{i}", nn.BatchNorm1d(neurons[-1]))
              
        # Decoder layers
        for i in range(len(neurons)-1, 0, -1):
            in_features = neurons[i]
            out_features = neurons[i-1]
            layer = nn.Linear(in_features, out_features)
            self.decoder_loop.add_module(f"decoder_layer_{i}", layer)
            if i != 1: # 1 porque en el range vamos al revés, por lo que es en esa capa donde nos interesa hacer la sigmoide
                self.decoder_loop.add_module(f"decoder_activation_{i}", nn.ReLU())
            else: # PENSAR LA CONDICIÓN QUE DEBERÍA PONER
                self.decoder_loop.add_module(f"decoder_sigmoid_{i}", nn.Sigmoid())  
        
        self.classifier = nn.Sequential(
            nn.Linear(neurons[-1], 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(5, 1),
            nn.Sigmoid()
        ) # HACERLO GENERALIZABLE O QUE SE PUEDA DEFINIR FUERA Y ASÍ NO TENER QUE CAMBIAR LA ESTRUCTURA DEL CLASIFICADOR
        
        
    def forward(self, x):
        encoded_loop = self.encoder_loop(x)
        decoded_loop = self.decoder_loop(encoded_loop)
        out_loop = self.classifier(encoded_loop)
        return decoded_loop, out_loop, encoded_loop 
 
class SubModel(nn.Module):
    def __init__(self, model):
        super(SubModel, self).__init__()
        self.og_net = model
        
    def forward(self, x):
        _, out_probs, _ = self.og_net(x)
        return out_probs
    
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight_pos, weight_neg):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def forward(self, inputs, targets):
        loss = -(self.weight_pos * targets * torch.log(inputs) + self.weight_neg * (1 - targets) * torch.log(1 - inputs))
        return torch.mean(loss)
    
# Train and evaluation functions
def calculate_accuracy(out, labels):
            round_pred = torch.round(out)
            similarity = round_pred == labels
            correct = similarity.sum()
            accuracy = correct/labels.shape[0] * 100
            return accuracy, round_pred
        
def epoch_time(start_time, end_time):
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
            return elapsed_mins, elapsed_secs

def train(train_tensor, y_train_tensor, model, optimizer, criterion1, criterion2, weight_rec=70, weight_bce=1):
            model.train()
            inputs = train_tensor
            labels = torch.reshape(y_train_tensor, (y_train_tensor.shape[0], 1))
            optimizer.zero_grad()
            decoded, out, _ = model(inputs)
            loss1 = criterion1(decoded, inputs)
            loss2 = criterion2(out, labels)
            weight1 = weight_rec#70#30 #30 #70 peso optimo para metilacion #1
            weight2 = weight_bce#1 #0.002
            loss = weight1 * loss1 + weight2 * loss2 
            acc, _ = calculate_accuracy(out, labels)
            loss.backward()
            optimizer.step()
            return loss.item(), acc.item(), loss1.item(), loss2.item(), labels, out
        
def evaluate(val_tensor, y_val_tensor, model, criterion1, criterion2):
            model.eval()
            with torch.no_grad():
                inputs = val_tensor
                labels = torch.reshape(y_val_tensor, (y_val_tensor.shape[0], 1))
                decoded, out, _ = model(inputs)
                loss1 = criterion1(decoded, inputs)
                loss2 = criterion2(out, labels)
                weigth1 = 1
                weigth2 = 1
                loss = weigth1 * loss1 + weigth2 * loss2
                acc, _ = calculate_accuracy(out, labels)
            return loss.item(), acc.item(), loss1.item(), loss2.item(), labels, out
        
def final_evaluation(val_tensor, y_val_tensor, model, criterion1, criterion2):
            model.eval()
            with torch.no_grad():
                inputs = val_tensor
                labels = torch.reshape(y_val_tensor, (y_val_tensor.shape[0], 1))
                decoded, out, encoded = model(inputs)
                loss1 = criterion1(decoded, inputs)
                loss2 = criterion2(out, labels)
                weigth1 = 1
                weigth2 = 1
                loss = weigth1 * loss1 + weigth2 * loss2
                acc, sim = calculate_accuracy(out, labels)       
            return labels, out, encoded, acc.item(), sim, labels, out
        
# Train and evaluation loop
def train_eval_loop(epochs, train_tensor, y_train_tensor, val_tensor, y_val_tensor, model, optimizer, criterion_rec, criterion_clf, i):
    for epoch in range(epochs):
        start_time = time.monotonic()
        train_loss, train_acc, t_loss1, t_loss2, t_labels, t_out = train(train_tensor, y_train_tensor, model, optimizer, criterion_rec, criterion_clf, weight_rec=1, weight_bce=1)
        valid_loss, valid_acc, v_loss1, v_loss2, v_labels, v_out = evaluate(val_tensor, y_val_tensor, model, criterion_rec, criterion_clf)
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc:.2f}%')
        print(f'\tTrain. Loss AE: {t_loss1:.3f} |  Train. Loss MLP: {t_loss2:.2f}')
        print(f'\tVal. Loss AE: {v_loss1:.3f} |  Val. Loss MLP: {v_loss2:.2f}')
        
        
    print(f'---------------------------- CV: {i} -------------------------------------')    
    labels, _, latent_space, acc, predictions, label, out = final_evaluation(val_tensor, y_val_tensor, model, criterion_rec, criterion_clf)
    return labels, out

# NN hyperparameters
def optim_param_hparam_nn(args, sheet_n='hyp_optim'):
    # Read excel with optimal hyperparameters per seed
    path_optim_df = args.root_data_path + 'opt_hyper.xlsx'
    optim_df = pd.read_excel(path_optim_df, sheet_name= sheet_n)
    # Select optimal hyperparameters for the seed
    epochs = optim_df.loc[optim_df['DB'] == args.db, 'EPOCHS'].values[0]
    learning_rate = optim_df.loc[optim_df['DB'] == args.db, 'LR'].values[0]
    latent_space = optim_df.loc[optim_df['DB'] == args.db, 'LATENT_SPACE'].values[0]
    # Specific parameters per database
    if args.db == 'colon':
        # Loss functions
        criterion_rec = nn.MSELoss()
        criterion_clf = nn.BCELoss()
        # Neurons and classes
        neurons = [2000, 500, 70] + [latent_space]
        classes = ['normal', 'colon']
    elif args.db == 'leukemia':
        # Loss functions
        criterion_rec = nn.MSELoss()
        criterion_clf = nn.BCELoss()
        # Neurons and classes
        neurons = [7129, 2000, 500, 70] + [latent_space]
        classes = ['ALL', 'AML']
    elif args.db == 'crohn':
        # Loss functions
        criterion_rec = nn.MSELoss()
        criterion_clf = nn.BCELoss()
        # Neurons and classes
        neurons = [22283, 6000, 2000, 500, 70] + [latent_space]
        classes = ['normal', 'crohn']
    else:
        # Loss functions
        criterion_rec = nn.MSELoss()
        criterion_clf = nn.BCELoss()
        # Optimizer
        learning_rate = 0.0001#0.0001
        # Epochs
        epochs = 2500
        # Neurons
        neurons = [4946, 2000, 500, 70, 10]
    return criterion_rec, criterion_clf, learning_rate, epochs, neurons, classes

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
    y_pred = predictions_cpu.numpy().astype(int)
    y_true = labels_cpu.numpy().astype(int)
    results = report_classification_results(y_true, y_pred, classes)
    return y_pred, y_true, results
    
# Integrated gradients relevance feature method
def IG_attributions(submodel, val_cv, y_val, round_pred, val_samples=7): # OJO: NECESARIO DEFINIR CORRECT_IDX_PREDS EN IGFS_METH
    """Obtain input attributions and correct indices"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Obtener muestras predichas correctamente
    res_var = y_val[y_val.columns[0]].values
    pred = round_pred.squeeze().astype(int)
    correct_samples = res_var == pred
    y_val_correct = y_val.loc[correct_samples]
    correct_indices = y_val_correct.index.values
    if val_samples != correct_indices.shape[0]: # correct_idx_preds.shape[1]
        diff = val_samples - correct_indices.shape[0]
        nans_array = np.full(diff, np.nan)
        correct_indices = np.append(correct_indices, nans_array)
    correct_indices = np.expand_dims(correct_indices, axis=0)
    
    # Apply integrated gradients
    val_tensor = torch.FloatTensor(val_cv.values).to(device)
    ig = IntegratedGradients(submodel)
    attr, delta = ig.attribute(val_tensor, n_steps=900, return_convergence_delta=True) #optimal number of steps
    attr = attr.to('cpu').detach().numpy()
    delta = delta.to('cpu').numpy()
    
    # Convert to dataframe and include the name of the regions 
    attr_df = pd.DataFrame(attr)
    attr_df.columns = val_cv.columns
    attr_df = attr_df.T
    
    # Compute mean
    attr_df_mean = attr_df.mean(axis=1)
    attr_df_mean.columns = y_val.index.to_list() # revisar porque probablemente quiera cambiar los nombres de las columnas de attr_df y no de attr_df_mean
    attr_df.insert(0, 'attr_mean', attr_df_mean)
    attr_df.insert(0, 'variable', val_cv.columns)
    
    # Sort dataframe with absolute values of the mean attributions for the subjects
    sorted_attr_df = attr_df.loc[attr_df['attr_mean'].sort_values(ascending=False).index]
    
    # Return dataframe and boolean array with correct val samples predictions
    return sorted_attr_df, correct_indices

# DATA WRANGLING OF IG RESULTS AND SELECTION OF CORRECT VARIABLES
def selected_correct_vars_attr(y, sorted_attr_df, correct_indices, class_1=True, cv=0, full=True):
    '''Select samples correctly classify and mantain the attributions'''
    # Filter indices of melanoma and nevus samples
    if class_1:
        indices = y.loc[y[y.columns[0]] == 1].index.to_list()
        indices_str = [str(element) for element in indices]
    else:
        indices = y.loc[y[y.columns[0]] == 0].index.to_list()
        indices_str = [str(element) for element in indices]

    # Filter attr and select cols
    if full:
        select_cols = list(set(indices_str).intersection(sorted_attr_df.columns.to_list()))
        select_cols.append('variable')
        samples_attr = sorted_attr_df.loc[:, select_cols]
    else:
        select_cols = list(set(indices_str).intersection(sorted_attr_df.columns.to_list()))
        select_cols = list(set(select_cols).intersection(list(map(str, correct_indices.squeeze().astype(int).tolist())))) 
        select_cols.append('variable')
        samples_attr = sorted_attr_df.loc[:, select_cols]
    
    # Change samples names
    samples_names = ['obs_' + str(samples_attr.columns[i]) + '_cv_' + str(cv) for i in range(len(samples_attr.columns.to_list()) - 1)]
    samples_names.append('variable')
    samples_attr.columns = samples_names
    
    # Return attributions values for the fold correct samples
    return samples_attr
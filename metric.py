import yaml
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader

from rna_model import *
from metric import dRMAE, align_svd_mae



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries
    def print(self):
        print(self.entries)

def get_xyz(train_sequences, train_labels):
    all_xyz = []
    for pdb_id in tqdm(train_sequences['target_id']):
        df             = train_labels[train_labels["pdb_id"]==pdb_id]
        xyz            = df[['x_1','y_1','z_1']].to_numpy().astype('float32')
        xyz[xyz<-1e17] = float('Nan');
        all_xyz.append(xyz)
    return all_xyz

def split_data(data):
    all_index = np.arange(len(data['sequence']))
    
    # Shuffle the indices to ensure a random 90-10 split
    np.random.shuffle(all_index)
    
    # Calculate the split point for 90% of the data
    split_point = int(0.9 * len(all_index))
    
    # Divide the shuffled indices into training and testing sets
    train_index = all_index[:split_point].tolist()
    test_index = all_index[split_point:].tolist()
    
    print(f"Total data points: {len(all_index)}")
    print(f"Training data points: {len(train_index)}")
    print(f"Testing data points: {len(test_index)}")

    return train_index, test_index

def get_ct(bp,s):
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0]-1,b[1]-1]=1
    return ct_matrix

class RNA3D_Dataset(Dataset):
    def __init__(self,indices,data):
        self.indices = indices
        self.data    = data
        self.tokens  = {nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        idx      = self.indices[idx]
        sequence = [self.tokens[nt] for nt in (self.data['sequence'][idx])]
        sequence = np.array(sequence)
        sequence = torch.tensor(sequence)

        #get C1' xyz
        xyz      = self.data['xyz'][idx]
        xyz      = torch.tensor(np.array(xyz))

        if len(sequence)>configs.max_len:
            crop_start = np.random.randint(len(sequence)-configs.max_len)
            crop_end   = crop_start+configs.max_len

            sequence=sequence[crop_start:crop_end]
            xyz=xyz[crop_start:crop_end]
        

        return {'sequence':sequence,
                'xyz':xyz}
    
def training_loop(logger, configs, model, train_loader, val_loader):

    epochs        = configs.epochs
    cos_epoch     = configs.cos_epoch
    batch_size    = configs.batch_size
    best_val_loss = 99999999999


    optimizer   = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=0.0001) #no weight decay following AF
    schedule    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs-cos_epoch)*len(train_loader)//batch_size)

    model.to('cuda')

    logger.info(f"Training with {len(train_loader)} batches per epoch, {epochs} epochs, batch size {batch_size}, cos_epoch {cos_epoch}")
    for epoch in range(epochs):

        logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        model.train()
        tbar       = tqdm(train_loader)
        total_loss = 0
        oom        = 0

        for idx, batch in enumerate(tbar):
            sequence = batch['sequence'].cuda()
            gt_xyz   = batch['xyz'].cuda().squeeze()
            pred_xyz = model(sequence).cuda().squeeze()
            
            loss     = dRMAE(pred_xyz,pred_xyz,gt_xyz,gt_xyz) + align_svd_mae(pred_xyz, gt_xyz)

            (loss/batch_size).backward()

            if (idx+1)%batch_size==0 or idx+1 == len(tbar):

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                if (epoch+1)>cos_epoch:
                    schedule.step()

            total_loss += loss.item()
            tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)} OOMs: {oom}")

        model.eval()
        val_preds = []
        val_loss  = 0

        tbar = tqdm(val_loader)
        for idx, batch in enumerate(tbar):
            sequence = batch['sequence'].cuda()
            gt_xyz   = batch['xyz'].cuda().squeeze()

            with torch.no_grad():
                pred_xyz = model(sequence).cuda().squeeze()
                loss     = dRMAE(pred_xyz,pred_xyz,gt_xyz,gt_xyz)
                
            val_loss += loss.item()
            val_preds.append([gt_xyz.cpu().numpy(),pred_xyz.cpu().numpy()])
        
        val_loss = val_loss/len(tbar)
        logger.warning(f"val loss: {val_loss}")
        
        if val_loss<best_val_loss:
            best_val_loss = val_loss

    model_save_path = configs.model_save_path + configs.experiment_name +'.pt'
    torch.save(model.state_dict(), model_save_path)   
    logger.info(f"Model saved to {model_save_path}") 

def train(configs):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if configs.mode == 'test':
        logger.info("Running in test mode, using only 1 sequence for training.")
        train_sequences = pd.read_parquet(configs.train_sequences_path)[0:10]
        train_labels    = pd.read_parquet(configs.train_labels_path)
    else:
        logger.info("Running in train mode, using 10000 sequences for training.")
        train_sequences = pd.read_parquet(configs.train_sequences_path)[0:10000]
        train_labels    = pd.read_parquet(configs.train_labels_path)

    train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])

    logger.info(f"Number of training sequences: {len(train_sequences)}")
    all_xyz    = get_xyz(train_sequences, train_labels)


    filter_nan = []
    max_len    = 0
    
    for xyz in all_xyz:
        if len(xyz) > max_len:
            max_len = len(xyz)
        filter_nan.append((np.isnan(xyz).mean() <= 0.5) & \
                          (len(xyz)<configs.max_len_filter) & \
                          (len(xyz)>configs.min_len_filter))
    print(f"Longest sequence in train: {max_len}")
    
    filter_nan      = np.array(filter_nan)
    non_nan_indices = np.arange(len(filter_nan))[filter_nan]
    
    train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
    all_xyz         = [all_xyz[i] for i in non_nan_indices]

    logger.warning(f"columns in train_sequences : {train_sequences.columns}")

    data = {
              "sequence"        : train_sequences['sequence'].to_list(),
              "xyz"             : all_xyz
        }

    logger.info(f"Number of sequences after filtering: {len(data['sequence'])}")
    train_index, test_index = split_data(data)
    logger.info(f"Train index length: {len(train_index)}, Test index length: {len(test_index)}")

    train_dataset = RNA3D_Dataset(train_index , data)
    val_dataset   = RNA3D_Dataset(test_index  , data)

    train_loader  = DataLoader(train_dataset,batch_size=configs.batch_size,shuffle=True)
    val_loader    = DataLoader(val_dataset,batch_size=configs.batch_size,shuffle=False)
    
    model         = RNAModel(logger, configs)

    training_loop(logger, configs, model, train_loader, val_loader)
      
    logger.info("Training complete. Model saved.")

if __name__=='__main__':

    with open(r"configs.yaml") as f:
        configs = yaml.safe_load(f)

    configs = Config(**configs)

    train(configs)
